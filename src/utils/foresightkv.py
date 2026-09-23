from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd
import pyarrow.parquet as pq
import torch
from safetensors.torch import safe_open
from torch import nn
from torch.utils.data import Dataset

from src.method.dataset import ATTN_COLUMNS


@dataclass
class ForesightConfig:
    model_id: str
    head_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    target: str = "future_max_128"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "ForesightConfig":
        return cls(**dict(data))

    @property
    def input_dim(self) -> int:
        return self.head_dim * 2 + 6 + len(ATTN_COLUMNS)


class ForesightScorer(nn.Module):
    def __init__(self, cfg: ForesightConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for _ in range(int(cfg.num_layers)):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.LayerNorm(cfg.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


def _pos_features(token_pos: torch.Tensor, checkpoint_t: torch.Tensor, layer: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    token_pos = token_pos.float()
    checkpoint_t = checkpoint_t.float().clamp_min(1.0)
    age = (checkpoint_t - token_pos).clamp_min(0.0)
    return torch.stack([token_pos / checkpoint_t, age / checkpoint_t, torch.log1p(age), torch.log1p(token_pos), layer / 100.0, head / 100.0], dim=-1)


class ForesightRankingTraceDataset(Dataset):
    def __init__(
        self,
        trace_root: Path,
        model_id: str,
        pattern: str = "by_source_200_*_ckpt128_replay512",
        target: str = "future_max_128",
        limit_runs: int | None = None,
        max_groups: int | None = None,
        max_seq_len: int = 1024,
        seed: int = 0,
    ):
        needed = {"sample_id", "checkpoint_t", "layer_id", "kv_head_id", "token_position", "vector_index", target, *ATTN_COLUMNS}
        frames = []
        for run_dir in sorted((Path(trace_root) / model_id).glob(pattern))[: limit_runs or None]:
            meta_path = run_dir / "metadata.parquet"
            if not meta_path.exists() or not (run_dir / "vectors.safetensors").exists():
                continue
            cols = [c for c in pq.read_schema(meta_path).names if c in needed]
            df = pd.read_parquet(meta_path, columns=cols)
            df["run_dir"] = str(run_dir)
            frames.append(df)
        if not frames:
            raise ValueError(f"No ForesightKV trace rows found for {model_id}")
        df = pd.concat(frames, ignore_index=True)
        groups = []
        for key, group in df.groupby(["run_dir", "sample_id", "checkpoint_t", "layer_id", "kv_head_id"], sort=False):
            group = group.sort_values("token_position")
            if len(group) > 1:
                groups.append((key, group))
        rng = random.Random(int(seed))
        rng.shuffle(groups)
        if max_groups:
            groups = groups[: int(max_groups)]
        self.groups = groups
        self.target = target
        self.max_seq_len = int(max_seq_len)
        self._safe_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.groups)

    def _safe(self, run_dir: str):
        opener = self._safe_cache.get(run_dir)
        if opener is None:
            opener = safe_open(str(Path(run_dir) / "vectors.safetensors"), framework="pt", device="cpu")
            self._safe_cache[run_dir] = opener
        return opener

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        (run_dir, _sample, checkpoint_t, layer_id, kv_head_id), group = self.groups[int(idx)]
        group = group.iloc[: self.max_seq_len]
        vector_index = torch.tensor(group["vector_index"].to_numpy("int64"), dtype=torch.long)
        f = self._safe(str(run_dir))
        k = f.get_tensor("k").index_select(0, vector_index).float()
        v = f.get_tensor("v").index_select(0, vector_index).float()
        token_pos = torch.tensor(group["token_position"].to_numpy("float32"))
        n = int(token_pos.numel())
        pos = _pos_features(
            token_pos,
            torch.full((n,), float(checkpoint_t)),
            torch.full((n,), float(layer_id)),
            torch.full((n,), float(kv_head_id)),
        )
        attn_vals = []
        for col in ATTN_COLUMNS:
            x = torch.tensor(group[col].to_numpy("float32")) if col in group else torch.zeros(n)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            attn_vals.append(torch.sign(x) * torch.log1p(x.abs()))
        attn = torch.stack(attn_vals, dim=-1)
        x = torch.cat([k, v, pos, attn], dim=-1)
        y = torch.tensor(group[self.target].to_numpy("float32"))
        return {"x": x, "y": y, "seq_len": torch.tensor(n, dtype=torch.long)}


def foresight_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(x["seq_len"]) for x in batch)
    dim = int(batch[0]["x"].shape[-1])
    xs = torch.zeros(len(batch), max_len, dim)
    ys = torch.zeros(len(batch), max_len)
    seq_lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, row in enumerate(batch):
        n = int(row["seq_len"])
        xs[i, :n] = row["x"]
        ys[i, :n] = row["y"]
        seq_lengths[i] = n
    return {"x": xs, "y": ys, "seq_lengths": seq_lengths}


def pairwise_ranking_loss(scores: torch.Tensor, target: torch.Tensor, seq_lengths: torch.Tensor, max_pairs: int = 2048) -> torch.Tensor:
    losses = []
    for b in range(scores.shape[0]):
        n = int(seq_lengths[b].item())
        if n < 2:
            continue
        y = target[b, :n]
        s = scores[b, :n]
        pairs = torch.nonzero(y.view(-1, 1) > y.view(1, -1), as_tuple=False)
        if pairs.numel() == 0:
            continue
        if pairs.shape[0] > max_pairs:
            idx = torch.randperm(pairs.shape[0], device=pairs.device)[:max_pairs]
            pairs = pairs[idx]
        diff = s[pairs[:, 0]] - s[pairs[:, 1]]
        losses.append(torch.nn.functional.softplus(-diff).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


class ForesightPaperScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu"):
        payload = torch.load(str(checkpoint), map_location="cpu")
        self.cfg = ForesightConfig.from_dict(payload["config"])
        self.model = ForesightScorer(self.cfg)
        self.model.load_state_dict(payload["model"])
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
        attn_features: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        kv_heads, seq_len = int(key.shape[1]), int(key.shape[2])
        out = torch.empty(kv_heads, seq_len, dtype=torch.float32, device=key.device)
        for h in range(kv_heads):
            token_pos = positions[h].float().to(key.device)
            checkpoint_t = torch.full_like(token_pos, float(token_pos.max().item() + 1 if token_pos.numel() else seq_len))
            pos = _pos_features(
                token_pos,
                checkpoint_t,
                torch.full_like(token_pos, float(layer_id)),
                torch.full_like(token_pos, float(h)),
            ).to(key.device)
            attn_cols = []
            for col in ATTN_COLUMNS:
                if attn_features and col in attn_features:
                    raw = attn_features[col][h, :seq_len] if attn_features[col].ndim == 2 else attn_features[col][:seq_len]
                else:
                    raw = torch.zeros(seq_len, dtype=torch.float32, device=key.device)
                raw = torch.nan_to_num(raw.float().to(key.device), nan=0.0, posinf=1e6, neginf=-1e6)
                attn_cols.append(torch.sign(raw) * torch.log1p(raw.abs()))
            x = torch.cat([key[0, h].float(), value[0, h].float(), pos, torch.stack(attn_cols, dim=-1)], dim=-1).to(self.device)
            out[h] = self.model(x).float().to(key.device)
        return out
