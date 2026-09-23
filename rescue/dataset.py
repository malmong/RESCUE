from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd
import pyarrow.parquet as pq
import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset

from rescue.mlp import LearnedPolicyConfig, normalize_positions, transform_label


ATTN_COLUMNS = [
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


@dataclass
class TraceShard:
    run_dir: Path
    metadata: pd.DataFrame


def find_trace_runs(trace_root: Path, model_id: str, pattern: str = "by_source_200_*_ckpt128_replay512") -> list[Path]:
    return sorted((trace_root / model_id).glob(pattern))


def load_trace_rows(
    trace_root: Path,
    model_id: str,
    pattern: str = "by_source_200_*_ckpt128_replay512",
    limit_runs: int | None = None,
    rows_per_run: int | None = None,
    seed: int = 0,
) -> list[TraceShard]:
    run_dirs = find_trace_runs(trace_root, model_id, pattern)
    if limit_runs:
        run_dirs = run_dirs[: int(limit_runs)]
    shards: list[TraceShard] = []
    needed = {
        "sample_id",
        "task_id",
        "checkpoint_t",
        "layer_id",
        "kv_head_id",
        "token_position",
        "vector_index",
        "future_sum_all",
        "future_max_all",
        "future_discounted",
        "future_sum_128",
        "future_max_128",
        *ATTN_COLUMNS,
    }
    rng = random.Random(int(seed))
    for run_dir in run_dirs:
        meta_path = run_dir / "metadata.parquet"
        vec_path = run_dir / "vectors.safetensors"
        if not meta_path.exists() or not vec_path.exists():
            continue
        schema_cols = pq.read_schema(meta_path).names
        cols = [c for c in schema_cols if c in needed]
        df = pd.read_parquet(meta_path, columns=cols)
        if rows_per_run and len(df) > int(rows_per_run):
            take = rng.sample(range(len(df)), int(rows_per_run))
            df = df.iloc[take].sort_index().reset_index(drop=True)
        df["run_dir"] = str(run_dir)
        shards.append(TraceShard(run_dir=run_dir, metadata=df))
    return shards


class TraceFeatureDataset(Dataset):
    def __init__(self, shards: list[TraceShard], cfg: LearnedPolicyConfig):
        self.cfg = cfg
        frames = [s.metadata for s in shards]
        if not frames:
            raise ValueError("TraceFeatureDataset received no shards")
        self.df = pd.concat(frames, ignore_index=True)
        self._safe_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _vectors(self, run_dir: str, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        opener = self._safe_cache.get(run_dir)
        if opener is None:
            opener = safe_open(str(Path(run_dir) / "vectors.safetensors"), framework="pt", device="cpu")
            self._safe_cache[run_dir] = opener
        k = opener.get_tensor("k")[index].float()
        v = opener.get_tensor("v")[index].float()
        q = opener.get_tensor("q_group")[index].float() if "q_group" in opener.keys() else None
        return k, v, q

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[int(idx)]
        k, v, q_group = self._vectors(str(row["run_dir"]), int(row["vector_index"]))
        pos = normalize_positions(
            torch.tensor(float(row["token_position"])),
            torch.tensor(float(row["checkpoint_t"])),
            torch.tensor(float(row["layer_id"])),
            torch.tensor(float(row["kv_head_id"])),
        )
        parts = [k, v, pos]
        if self.cfg.feature_family == "kv_pos_attn":
            vals = []
            for col in ATTN_COLUMNS:
                vals.append(float(row[col]) if col in row and pd.notna(row[col]) else 0.0)
            attn = torch.tensor(vals, dtype=torch.float32)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=1e6, neginf=-1e6)
            attn = torch.sign(attn) * torch.log1p(attn.abs())
            parts.append(attn)
        elif self.cfg.feature_family == "kv_pos_q":
            if q_group is None:
                q_mean = torch.zeros_like(k)
                sims = torch.zeros(5, dtype=torch.float32)
            else:
                q_mean = q_group.mean(dim=0)
                qn = torch.nn.functional.normalize(q_group, dim=-1)
                kn = torch.nn.functional.normalize(k.view(1, -1), dim=-1)
                cos = (qn * kn).sum(dim=-1)
                logits = (q_group * k.view(1, -1)).sum(dim=-1) / max(1.0, k.numel() ** 0.5)
                sims = torch.stack([cos.mean(), cos.max(), cos.min(), logits.mean(), logits.max()])
            parts.extend([q_mean, sims])
        x = torch.cat([p.flatten() for p in parts]).float()
        y = transform_label(torch.tensor(float(row[self.cfg.target])), self.cfg.label_transform)
        return x, y.float()
