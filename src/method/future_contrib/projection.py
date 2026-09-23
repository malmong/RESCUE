"""Frozen random projection shared by candidate (z_i) and current-state (c_t) features.

Spec ref: implementation-spec.md sections 3-5. A single [proj_dim, hidden_size]
Gaussian matrix, generated once from a fixed seed and never trained, applied
identically to every token/layer/model. Stored as a plain tensor (+ the seed
that produced it) so checkpoints stay reproducible without re-deriving RNG state.
"""
from __future__ import annotations

import torch


class FrozenRandomProjection:
    def __init__(self, hidden_size: int, proj_dim: int = 32, seed: int = 0, device=None, dtype=torch.float32):
        self.hidden_size = int(hidden_size)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        # N(0, 1/proj_dim) so ||Pz|| stays O(||h||) regardless of proj_dim.
        mat = torch.randn(self.proj_dim, self.hidden_size, generator=gen) / (self.proj_dim ** 0.5)
        self.matrix = mat.to(device=device, dtype=dtype)
        self.matrix.requires_grad_(False)

    def to(self, device=None, dtype=None) -> "FrozenRandomProjection":
        self.matrix = self.matrix.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def project(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., hidden_size] -> [..., proj_dim]."""
        return torch.matmul(h.to(self.matrix.dtype), self.matrix.transpose(0, 1))

    def state_dict(self) -> dict:
        return {"hidden_size": self.hidden_size, "proj_dim": self.proj_dim, "seed": self.seed, "matrix": self.matrix.clone()}

    @classmethod
    def from_state_dict(cls, sd: dict, device=None, dtype=torch.float32) -> "FrozenRandomProjection":
        obj = cls(sd["hidden_size"], sd["proj_dim"], sd["seed"], device=device, dtype=dtype)
        # Reproduced from seed above; overwrite with the saved matrix in case
        # torch's RNG algorithm ever changes across versions, so an old
        # checkpoint keeps working bit-for-bit rather than silently drifting.
        obj.matrix = sd["matrix"].to(device=device, dtype=dtype)
        return obj
