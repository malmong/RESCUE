from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.method.mlp import LearnedPolicyConfig, LearnedRanker, normalize_positions


class LearnedScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu"):
        payload = torch.load(str(checkpoint), map_location="cpu")
        cfg_data: dict[str, Any] = payload["config"]
        self.cfg = LearnedPolicyConfig.from_dict(cfg_data)
        self.model = LearnedRanker(self.cfg)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.inference_mode()
    def score_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
        attn_features: dict[str, torch.Tensor] | None = None,
        q_group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # key/value: [batch, kv_heads, seq, dim], positions: [kv_heads, seq]
        key = key.detach()
        value = value.detach()
        kv_heads, seq_len, head_dim = int(key.shape[1]), int(key.shape[2]), int(key.shape[3])
        if head_dim != int(self.cfg.head_dim):
            raise ValueError(f"Checkpoint head_dim={self.cfg.head_dim}, runtime head_dim={head_dim}")
        xs = []
        for h in range(kv_heads):
            token_pos = positions[h].to(key.device).float()
            checkpoint_t = torch.full_like(token_pos, float(token_pos.max().item() + 1 if token_pos.numel() else seq_len))
            layer = torch.full_like(token_pos, float(layer_id))
            head = torch.full_like(token_pos, float(h))
            pos = normalize_positions(token_pos, checkpoint_t, layer, head).to(key.device)
            parts = [key[0, h].float(), value[0, h].float(), pos.float()]
            if self.cfg.feature_family == "kv_pos_attn":
                parts.append(self._runtime_attn_features(attn_features, h, seq_len, key.device))
            elif self.cfg.feature_family == "kv_pos_q":
                parts.extend(self._runtime_query_features(q_group, key[0, h].float()))
            x = torch.cat(parts, dim=-1)
            xs.append(x)
        features = torch.cat(xs, dim=0).to(self.device)
        scores = self.model(features).float().cpu()
        return scores.view(kv_heads, seq_len).to(key.device)

    def _runtime_attn_features(
        self,
        attn_features: dict[str, torch.Tensor] | None,
        kv_head: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        names = [
            "current_attn_prob",
            "current_qk_logit",
            "attention_row_sum",
            "hist_sum_all",
            "hist_max_all",
            "hist_mean",
            "hist_variance",
            "last_attended_distance",
            "hit_count",
            "hist_sum_last_1",
            "hist_sum_last_4",
            "hist_sum_last_16",
            "hist_sum_last_64",
            "hist_sum_last_256",
            "hist_max_last_256",
        ]
        cols = []
        for name in names:
            if attn_features and name in attn_features:
                val = attn_features[name]
                if val.ndim == 2:
                    col = val[kv_head, :seq_len]
                else:
                    col = val[:seq_len]
            else:
                col = torch.zeros(seq_len, dtype=torch.float32, device=device)
            col = torch.nan_to_num(col.float().to(device), nan=0.0, posinf=1e6, neginf=-1e6)
            cols.append(torch.sign(col) * torch.log1p(col.abs()))
        return torch.stack(cols, dim=-1)

    def _runtime_query_features(
        self,
        q_group: torch.Tensor | None,
        key_h: torch.Tensor,
    ) -> list[torch.Tensor]:
        seq_len, head_dim = key_h.shape
        if q_group is None:
            return [
                torch.zeros(seq_len, head_dim, dtype=torch.float32, device=key_h.device),
                torch.zeros(seq_len, 5, dtype=torch.float32, device=key_h.device),
            ]
        q = q_group.float().to(key_h.device)
        if q.ndim == 2:
            q = q.unsqueeze(0)
        q_mean = q.mean(dim=0).view(1, -1).expand(seq_len, -1)
        qn = torch.nn.functional.normalize(q, dim=-1)
        kn = torch.nn.functional.normalize(key_h.view(1, seq_len, head_dim), dim=-1)
        cos = (qn.unsqueeze(1) * kn.unsqueeze(0)).sum(dim=-1)
        logits = (q.unsqueeze(1) * key_h.view(1, seq_len, head_dim)).sum(dim=-1) / max(1.0, head_dim ** 0.5)
        sims = torch.stack([cos.mean(dim=0), cos.max(dim=0).values, cos.min(dim=0).values, logits.mean(dim=0), logits.max(dim=0).values], dim=-1)
        return [q_mean, sims]

