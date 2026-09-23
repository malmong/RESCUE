from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


LEARNED_POLICY_FEATURES = {
    "kvp": "kv_pos",
    "foresightkv": "kv_pos_attn",
    "lookaheadkv": "kv_pos_q",
}


@dataclass
class LearnedPolicyConfig:
    policy: str
    model_id: str
    head_dim: int
    q_group_size: int = 1
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    target: str = "future_sum_all"
    feature_family: str | None = None
    label_transform: str = "log1p"
    score_activation: str = "identity"
    notes: str = ""

    def __post_init__(self) -> None:
        self.policy = str(self.policy).lower()
        if self.feature_family is None:
            self.feature_family = LEARNED_POLICY_FEATURES.get(self.policy, "kv_pos")

    @property
    def input_dim(self) -> int:
        base = 2 * int(self.head_dim) + 6
        if self.feature_family == "kv_pos":
            return base
        if self.feature_family == "kv_pos_attn":
            return base + 15
        if self.feature_family == "kv_pos_q":
            return base + int(self.head_dim) + 5
        raise ValueError(f"Unknown feature_family={self.feature_family}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedPolicyConfig":
        return cls(**dict(data))


class LearnedRanker(nn.Module):
    def __init__(self, cfg: LearnedPolicyConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = int(cfg.input_dim)
        hidden = int(cfg.hidden_dim)
        depth = max(1, int(cfg.num_layers))
        for layer_idx in range(depth):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden, hidden))
            layers.append(nn.SiLU())
            if float(cfg.dropout) > 0:
                layers.append(nn.Dropout(float(cfg.dropout)))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features.float()).squeeze(-1)
        if self.cfg.score_activation == "softplus":
            out = torch.nn.functional.softplus(out)
        return out


def normalize_positions(
    token_position: torch.Tensor,
    checkpoint_t: torch.Tensor,
    layer_id: torch.Tensor,
    kv_head_id: torch.Tensor,
) -> torch.Tensor:
    token_position = token_position.float()
    checkpoint_t = checkpoint_t.float().clamp_min(1.0)
    age = (checkpoint_t - token_position).clamp_min(0.0)
    return torch.stack(
        [
            token_position / checkpoint_t,
            age / checkpoint_t,
            torch.log1p(age),
            torch.log1p(token_position.clamp_min(0.0)),
            layer_id.float() / 100.0,
            kv_head_id.float() / 100.0,
        ],
        dim=-1,
    )


def transform_label(label: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return label.float()
    if transform == "log1p":
        return torch.log1p(label.float().clamp_min(0.0))
    if transform == "binary":
        return (label.float() > 0).float()
    raise ValueError(f"Unknown label transform={transform}")

