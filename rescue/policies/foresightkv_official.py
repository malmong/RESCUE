from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace

import torch

# ForesightKV is a third-party checkout we cannot redistribute. Point
# RESCUE_FORESIGHTKV_ROOT at a clone of the authors' release; only
# --method foresightkv needs it.
_FORESIGHTKV_EVAL_ROOT = Path(
    os.environ.get("RESCUE_FORESIGHTKV_ROOT",
                   Path(__file__).resolve().parents[2] / "third_party" / "ForesightKV")
) / "evaluation"


def _selection_model():
    """Import the authors' SelectionModel lazily.

    Importing at module scope would make every policy -- including the ones
    that need nothing third-party -- fail on a checkout without ForesightKV,
    because this module is imported by the generator unconditionally.
    """
    if str(_FORESIGHTKV_EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(_FORESIGHTKV_EVAL_ROOT))
    try:
        from rkv.modeling import SelectionModel
    except ImportError as exc:
        raise SystemExit(
            f"ForesightKV not found at {_FORESIGHTKV_EVAL_ROOT.parent}: {exc}\n"
            "Clone the authors' repository there or set RESCUE_FORESIGHTKV_ROOT. "
            "Only --method foresightkv needs it."
        ) from exc
    return SelectionModel


class ForesightKVJudgeScorer:
    """Loads judge_model checkpoints produced by scripts/train_foresightkv_paper.py
    or scripts/train_foresightkv_rl.py and scores candidate tokens with the
    genuine ForesightKV mechanism (SelectionModel judge over K/V + 8 causal
    attention statistics), replacing the earlier generic MLP proxy
    (rescue.policies.foresightkv.ForesightPaperScorer) that this project's
    own fidelity audit found architecturally unrelated to the paper.

    Unlike the official eval-time class (reference/official/ForesightKV/
    evaluation/rkv/compression/foresightkv.py), which only re-scores every
    `divide_length` decoding steps and keeps persistent attn_acc/attn_acc_decay
    state across calls, this recomputes the 8 statistics fresh from whatever
    attention this call was given every time score_layer runs -- consistent
    with how kvbench already re-scores every layer at every prefill/decode
    step for its other learned baselines (H2O, KVP, ...), and with how
    training computed cumsum/cumsum_decay (see train_foresightkv_paper.py's
    documented simplification: those two channels are recomputed per call
    rather than carried across calls).
    """

    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu"):
        payload = torch.load(str(checkpoint), map_location="cpu")
        if "judge_models" not in payload or "model_arch" not in payload:
            raise ValueError(
                f"{checkpoint} is not a ForesightKV judge checkpoint "
                "(expected keys 'judge_models' and 'model_arch' -- produced by "
                "scripts/train_foresightkv_paper.py or scripts/train_foresightkv_rl.py)"
            )
        arch = payload["model_arch"]
        self.head_dim = int(arch["head_dim"])
        self.num_key_value_heads = int(arch["num_key_value_heads"])
        self.num_attention_heads = int(arch["num_attention_heads"])
        self.kv_repeat = self.num_attention_heads // self.num_key_value_heads
        self.device = torch.device(device)

        cfg = SimpleNamespace(
            head_dim=self.head_dim,
            num_key_value_heads=self.num_key_value_heads,
            num_attention_heads=self.num_attention_heads,
        )
        self.judge_models: list[SelectionModel] = []
        for state_dict in payload["judge_models"]:
            jm = _selection_model()(cfg)
            jm.load_state_dict(state_dict)
            jm.float().eval().to(self.device)
            for p in jm.parameters():
                p.requires_grad = False
            self.judge_models.append(jm)

    @torch.inference_mode()
    def score_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
        raw_attention: torch.Tensor | None,
    ) -> torch.Tensor:
        """key/value: [kv_heads, seq_len, head_dim]. raw_attention: the layer's
        real attention weights for whatever queries this call observed
        (prefill's observation window, or a single decode-step query) --
        shape [num_attention_heads or num_kv_heads, q_len, seq_len]. Returns
        an evict-priority score per (kv_head, position) -- higher means more
        evictable, matching ForesightKV's own convention.
        """
        kv_heads, seq_len, _ = key.shape
        if layer_id >= len(self.judge_models):
            return torch.zeros(kv_heads, seq_len, device=key.device)
        judge = self.judge_models[layer_id]

        if raw_attention is None:
            # No attention observed this call (e.g. sdpa/flash attn_implementation
            # without output_attentions): fall back to a neutral all-tied score,
            # matching what KVPPaperScorer does when an agent is missing.
            attn_info = torch.zeros(kv_heads, seq_len, self.kv_repeat * 8, device=key.device, dtype=torch.float32)
        else:
            attn = raw_attention.float().to(key.device)
            if attn.dim() == 2:
                attn = attn.unsqueeze(0)
            if attn.shape[0] == self.num_attention_heads and self.kv_repeat > 1:
                attn = attn.view(kv_heads, self.kv_repeat, attn.shape[-2], attn.shape[-1]).mean(dim=1)
            elif attn.shape[0] != kv_heads:
                attn = attn.mean(dim=0, keepdim=True).expand(kv_heads, -1, -1)
            attn = attn[:, :, :seq_len]  # [kv_heads, q_len, seq_len]

            obs_max = attn.max(dim=1).values
            obs_min = attn.min(dim=1).values
            obs_sum = attn.sum(dim=1)
            n_q = attn.shape[1]
            sum1 = attn[:, -min(8, n_q):, :].sum(dim=1)
            sum2 = attn[:, -min(16, n_q):, :].sum(dim=1)
            sum3 = attn[:, -min(32, n_q):, :].sum(dim=1)
            # cumsum/cumsum_decay: recomputed fresh per call (see class docstring).
            stats = torch.stack([obs_max, obs_min, obs_sum, sum2, sum1, sum3, obs_sum, obs_sum], dim=-1)
            attn_info = stats.unsqueeze(2).expand(-1, -1, self.kv_repeat, -1).reshape(kv_heads, seq_len, self.kv_repeat * 8)

        key_in = key.float().to(self.device).unsqueeze(0)
        value_in = value.float().to(self.device).unsqueeze(0)
        attn_info_in = attn_info.unsqueeze(0).to(self.device)  # [1, kv_heads, seq_len, feat]

        out = judge(key_in, value_in, attn_info_in).squeeze(0).squeeze(-1).transpose(0, 1)  # [kv_heads, seq_len]
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out.to(key.device)
