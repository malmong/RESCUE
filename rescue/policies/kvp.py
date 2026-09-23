from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random

import pandas as pd
import pyarrow.parquet as pq
import torch
from safetensors.torch import safe_open
from torch import nn
from torch.utils.data import Dataset


@dataclass
class KVPAgentConfig:
    model_id: str
    layer_id: int
    kv_head_id: int
    head_dim: int = 128
    max_seq_length: int = 6000
    processing_dim: int = 256
    mlp_hidden_size: int = 256
    pos_embedding_dim: int = 32
    num_mlp_layers: int = 2
    n_sinks: int = 4
    n_running_window: int = 16

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "KVPAgentConfig":
        return cls(**dict(data))


class KVPRankingAgent(nn.Module):
    """KVP-style per-layer/per-KV-head scoring agent.

    This mirrors the released Apple KVP agent's default paper setup:
    key projection + value projection + learned absolute position embedding,
    followed by a small MLP. Query and attention features are intentionally not
    used.
    """

    def __init__(self, cfg: KVPAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.key_projection = nn.Linear(cfg.head_dim, cfg.processing_dim)
        self.value_projection = nn.Linear(cfg.head_dim, cfg.processing_dim)
        self.pos_embedding = nn.Embedding(cfg.max_seq_length, cfg.pos_embedding_dim)
        self.norm_key = nn.LayerNorm(cfg.processing_dim)
        self.norm_value = nn.LayerNorm(cfg.processing_dim)
        self.norm_pos = nn.LayerNorm(cfg.pos_embedding_dim)
        layers: list[nn.Module] = []
        in_dim = cfg.processing_dim * 2 + cfg.pos_embedding_dim
        for _ in range(int(cfg.num_mlp_layers)):
            layers.append(nn.Linear(in_dim, cfg.mlp_hidden_size))
            layers.append(nn.LayerNorm(cfg.mlp_hidden_size))
            layers.append(nn.SiLU())
            in_dim = cfg.mlp_hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, keys: torch.Tensor, values: torch.Tensor, seq_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # keys/values: [B, S, D]
        bsz, seq_len, _ = keys.shape
        device = keys.device
        pos = torch.arange(seq_len, device=device).clamp(max=self.cfg.max_seq_length - 1).view(1, -1).expand(bsz, -1)
        x = torch.cat(
            [
                self.norm_key(self.key_projection(keys.float())),
                self.norm_value(self.value_projection(values.float())),
                self.norm_pos(self.pos_embedding(pos)),
            ],
            dim=-1,
        )
        scores = self.mlp(x).squeeze(-1)
        if seq_lengths is not None:
            mask = torch.arange(seq_len, device=device).view(1, -1) >= seq_lengths.view(-1, 1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        return scores


def kvp_auc_reward(scores: torch.Tensor, future_importance: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Continuous future-attention AUC reward from the KVP paper/code.

    For every possible cache size, the top-b prefix of the ranking should retain
    as much future attention as the oracle ranking. This computes the normalized
    AUC over all b using the same weighted-ranking idea as Apple's
    FutureAttentionAucNormalizedReward.
    """
    bsz, seq_len = scores.shape
    order = scores.argsort(dim=-1, descending=True)
    ranked_imp = future_importance.gather(1, order).float()
    mask = torch.arange(seq_len, device=scores.device).view(1, -1) < seq_lengths.view(-1, 1)
    ranked_imp = ranked_imp.masked_fill(~mask, 0.0)
    weights = torch.arange(1, seq_len + 1, device=scores.device, dtype=torch.float32).view(1, -1)
    agent_auc = (ranked_imp * weights).sum(dim=-1)
    oracle_imp = future_importance.float().masked_fill(~mask, 0.0).sort(dim=-1, descending=True).values
    oracle_auc = (oracle_imp * weights).sum(dim=-1).clamp_min(1e-8)
    return agent_auc / oracle_auc


def list_trace_runs(trace_root: Path, model_id: str, pattern: str) -> list[Path]:
    return sorted((trace_root / model_id).glob(pattern))


class KVPRankingTraceDataset(Dataset):
    def __init__(
        self,
        trace_root: Path,
        model_id: str,
        layer_id: int,
        kv_head_id: int,
        pattern: str = "by_source_200_*_ckpt128_replay512",
        target: str = "future_sum_all",
        limit_runs: int | None = None,
        max_groups: int | None = None,
        max_seq_len: int = 4096,
        seed: int = 0,
    ):
        self.trace_root = Path(trace_root)
        self.model_id = model_id
        self.layer_id = int(layer_id)
        self.kv_head_id = int(kv_head_id)
        self.target = target
        self.max_seq_len = int(max_seq_len)
        run_dirs = list_trace_runs(self.trace_root, model_id, pattern)
        if limit_runs:
            run_dirs = run_dirs[: int(limit_runs)]
        cols_needed = {
            "sample_id",
            "checkpoint_t",
            "layer_id",
            "kv_head_id",
            "token_position",
            "vector_index",
            target,
        }
        frames = []
        for run_dir in run_dirs:
            meta_path = run_dir / "metadata.parquet"
            vec_path = run_dir / "vectors.safetensors"
            if not meta_path.exists() or not vec_path.exists():
                continue
            cols = [c for c in pq.read_schema(meta_path).names if c in cols_needed]
            df = pd.read_parquet(meta_path, columns=cols)
            df = df[(df["layer_id"] == self.layer_id) & (df["kv_head_id"] == self.kv_head_id)]
            if df.empty:
                continue
            df["run_dir"] = str(run_dir)
            frames.append(df)
        if not frames:
            raise ValueError(f"No KVP groups found for {model_id} layer={layer_id} head={kv_head_id}")
        df = pd.concat(frames, ignore_index=True)
        groups = []
        for key, group in df.groupby(["run_dir", "sample_id", "checkpoint_t"], sort=False):
            group = group.sort_values("token_position")
            if len(group) > 1:
                groups.append((key, group[["vector_index", target]].to_numpy()))
        rng = random.Random(int(seed))
        rng.shuffle(groups)
        if max_groups:
            groups = groups[: int(max_groups)]
        self.groups = groups
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
        (run_dir, _sample_id, _checkpoint_t), arr = self.groups[int(idx)]
        if len(arr) > self.max_seq_len:
            arr = arr[: self.max_seq_len]
        vector_index = torch.tensor(arr[:, 0].astype("int64"), dtype=torch.long)
        future = torch.tensor(arr[:, 1].astype("float32"), dtype=torch.float32)
        f = self._safe(str(run_dir))
        keys = f.get_tensor("k").index_select(0, vector_index).float()
        values = f.get_tensor("v").index_select(0, vector_index).float()
        return {"keys": keys, "values": values, "future": future, "seq_len": torch.tensor(len(vector_index), dtype=torch.long)}


def kvp_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(x["seq_len"]) for x in batch)
    head_dim = int(batch[0]["keys"].shape[-1])
    keys = torch.zeros(len(batch), max_len, head_dim, dtype=torch.float32)
    values = torch.zeros_like(keys)
    future = torch.zeros(len(batch), max_len, dtype=torch.float32)
    seq_lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, row in enumerate(batch):
        n = int(row["seq_len"])
        keys[i, :n] = row["keys"]
        values[i, :n] = row["values"]
        future[i, :n] = row["future"]
        seq_lengths[i] = n
    return {"keys": keys, "values": values, "future": future, "seq_lengths": seq_lengths}


def save_kvp_agent(path: Path, cfg: KVPAgentConfig, model: KVPRankingAgent, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "history": history}, path)


def load_kvp_agent(path: Path, device: torch.device | str = "cpu") -> KVPRankingAgent:
    payload = torch.load(str(path), map_location="cpu")
    cfg = KVPAgentConfig.from_dict(payload["config"])
    model = KVPRankingAgent(cfg)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


class KVPPaperScorer:
    """Each (layer, kv_head) has its own SEPARATELY trained agent (256 of them
    for a 32-layer/8-kv-head backbone) -- so scoring a layer can't just be one
    shared-weight batched call. But every agent shares the identical
    architecture (same KVPAgentConfig shapes), which is exactly the case
    torch.func.vmap over stacked per-model parameters is for: one vmapped
    call runs all of a layer's kv_head agents together instead of a Python
    loop issuing kv_heads separate small forward passes (measured to matter:
    this loop runs on EVERY decode step, every layer)."""

    def __init__(self, root: str | Path, device: torch.device | str = "cpu"):
        self.root = Path(root)
        self.device = torch.device(device)
        self.agents: dict[tuple[int, int], KVPRankingAgent] = {}
        if self.root.is_file():
            agent = load_kvp_agent(self.root, self.device)
            self.agents[(agent.cfg.layer_id, agent.cfg.kv_head_id)] = agent
        else:
            for path in sorted(self.root.glob("layer_*/kv_head_*/agent.pt")):
                agent = load_kvp_agent(path, self.device)
                self.agents[(agent.cfg.layer_id, agent.cfg.kv_head_id)] = agent
        if not self.agents:
            raise ValueError(f"No KVP paper agents found under {self.root}")
        self._layer_batches: dict[int, dict] = {}
        self._build_layer_batches()

    def _build_layer_batches(self) -> None:
        from torch.func import functional_call, stack_module_state, vmap

        by_layer: dict[int, dict[int, KVPRankingAgent]] = {}
        for (layer_id, kv_head_id), agent in self.agents.items():
            by_layer.setdefault(layer_id, {})[kv_head_id] = agent
        for layer_id, heads in by_layer.items():
            kv_head_ids = sorted(heads)
            models = [heads[h] for h in kv_head_ids]
            params, buffers = stack_module_state(models)
            base = models[0]  # architecture template only; functional_call ignores its own weights

            def call_one(p, b, k, v, sl, _base=base):
                return functional_call(_base, (p, b), (k, v, sl))

            self._layer_batches[layer_id] = {
                "kv_head_ids": torch.tensor(kv_head_ids, dtype=torch.long),
                "params": params,
                "buffers": buffers,
                # Built once here rather than re-wrapped on every score_layer
                # call (called on every decode step, every layer) -- vmap's
                # own dispatch/tracing setup has real per-call cost that a
                # fresh wrapper pays again and again for no reason.
                "vmapped": vmap(call_one),
            }

    @torch.inference_mode()
    def score_layer(self, key: torch.Tensor, value: torch.Tensor, positions: torch.Tensor, layer_id: int) -> torch.Tensor:
        kv_heads, seq_len = int(key.shape[1]), int(key.shape[2])
        out = torch.full((kv_heads, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32, device=key.device)
        batch = self._layer_batches.get(int(layer_id))
        if batch is None:
            # No agents trained for this layer at all -- matches the old
            # per-head "missing agent" fallback (deliberately low priority
            # rather than silently sharing another head's policy).
            return out

        kv_head_ids = batch["kv_head_ids"].to(key.device)
        keys_in = key[0].index_select(0, kv_head_ids).unsqueeze(1).float().to(self.device)    # [H, 1, seq, dim]
        values_in = value[0].index_select(0, kv_head_ids).unsqueeze(1).float().to(self.device)  # [H, 1, seq, dim]
        seq_lengths_in = torch.full((kv_head_ids.numel(), 1), seq_len, dtype=torch.long, device=self.device)

        scores = batch["vmapped"](batch["params"], batch["buffers"], keys_in, values_in, seq_lengths_in)  # [H, 1, seq]
        scores = scores.squeeze(1).float().to(key.device)  # [H, seq]
        out[kv_head_ids] = scores
        return out
