"""Trace-based feature/label extraction. Spec ref: sections 2, 6, 10, 13, 20-22.

Unlike LookaheadKV/ForesightKV, this method needs no custom Attention/Model
classes and no live cache-eviction injection: we only need the real Q/K/V a
plain teacher-forced full-KV forward pass produces. So we monkeypatch only the
pure `eager_attention_forward` function each backbone's modeling module calls
(capturing its inputs right before repeat_kv/softmax, then letting it run
unmodified), plus a forward-pre-hook per decoder layer to grab the residual
hidden_states flowing into it. attn_implementation="eager" forces this path.

Candidate/protected definition mirrors kvbench's own BaselineConfig
(src/baselines/scoring.py): sink_tokens=4 leading tokens + recent_tokens=64
trailing window are protected; everything else currently cached is an
evictable candidate (spec section 2's requirement to match the real
framework's protected-window definition).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

MODEL_TYPE_TO_MODULE = {
    "llama": "transformers.models.llama.modeling_llama",
    "mistral": "transformers.models.mistral.modeling_mistral",
    "qwen3": "transformers.models.qwen3.modeling_qwen3",
}


@dataclass
class CandidateConfig:
    sink_tokens: int = 4
    recent_tokens: int = 64
    budget: int = 256          # first eviction step trigger (cache "fills up")
    step_stride: int = 64      # eviction cadence thereafter
    max_candidates: int = 256  # subsample cap (spec section 23)
    horizon: int = 256         # H, spec section 13


class SequenceCapture:
    """Holds per-layer captured tensors for one forward pass. Populated by hooks
    installed via install_capture_hooks; freed by clear()."""

    def __init__(self):
        self.qkv: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.hidden_in: dict[int, torch.Tensor] = {}

    def clear(self):
        self.qkv.clear()
        self.hidden_in.clear()


def install_capture_hooks(model, model_type: str, capture: SequenceCapture) -> Callable[[], None]:
    """Returns a cleanup() callable that restores the original function and
    removes hooks. Must be called with model.config._attn_implementation ==
    'eager' (checked by the caller) so the patched function is actually used.
    """
    import importlib

    module_path = MODEL_TYPE_TO_MODULE[model_type]
    module = importlib.import_module(module_path)
    original_eager = module.eager_attention_forward

    def traced_eager(attn_module, query, key, value, attention_mask, **kwargs):
        layer_idx = getattr(attn_module, "layer_idx", None)
        if layer_idx is not None:
            capture.qkv[layer_idx] = (query.detach(), key.detach(), value.detach())
        return original_eager(attn_module, query, key, value, attention_mask, **kwargs)

    module.eager_attention_forward = traced_eager

    handles = []
    layers = model.model.layers

    def make_pre_hook(layer_idx):
        def pre_hook(_module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is not None:
                capture.hidden_in[layer_idx] = hidden_states.detach()
            return None
        return pre_hook

    for layer_idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx), with_kwargs=True))

    def cleanup():
        module.eager_attention_forward = original_eager
        for h in handles:
            h.remove()

    return cleanup


@torch.no_grad()
def capture_sequence(model, input_ids: torch.Tensor, model_type: str) -> SequenceCapture:
    """Runs one plain forward pass and returns per-layer (query, key, value,
    hidden_in), each squeezed to batch size 1: query [num_heads, seq, head_dim],
    key/value [num_kv_heads, seq, head_dim] (not GQA-repeated), hidden_in
    [seq, hidden_size]."""
    assert model.config._attn_implementation == "eager", (
        "capture_sequence requires attn_implementation='eager' -- otherwise "
        "eager_attention_forward (the function this module patches) is never called."
    )
    capture = SequenceCapture()
    cleanup = install_capture_hooks(model, model_type, capture)
    try:
        model(input_ids=input_ids, use_cache=False, output_attentions=False, output_hidden_states=False)
    finally:
        cleanup()

    out = SequenceCapture()
    for layer_idx, (q, k, v) in capture.qkv.items():
        out.qkv[layer_idx] = (q[0], k[0], v[0])
    for layer_idx, h in capture.hidden_in.items():
        out.hidden_in[layer_idx] = h[0]
    return out


@torch.no_grad()
def capture_sequence_chunked(model, input_ids: torch.Tensor, model_type: str, chunk_size: int = 4096,
                             keep_hidden: bool = True) -> SequenceCapture:
    """Same contract/output shape as capture_sequence, but processes input_ids
    in QUERY-dimension chunks with a real KV cache instead of one single
    forward call. eager_attention_forward's own softmax still materializes a
    full [heads, chunk_len, seen_so_far] fp32 matrix per chunk (unavoidable --
    that IS the function this module patches to capture Q/K, see
    install_capture_hooks), but that is O(chunk_len * total_len) instead of
    capture_sequence's O(total_len^2), which OOMs on long documents (e.g.
    qasper's outlier ~17K-token samples under this project's uncapped
    max_seq_len=131072: 32 heads * 17000^2 * 4 bytes ~= 37GB in one shot).
    Only use this when the caller (currently just
    src.method.oracle_future.OracleFutureScorer.set_reference, which
    needs the full sequence's real rotary-applied Q/K to look up a ground-
    truth future attention distribution) can tolerate the O(chunk*total)
    memory profile -- existing training-time callers keep using plain
    capture_sequence unchanged."""
    from transformers import DynamicCache

    assert model.config._attn_implementation == "eager", (
        "capture_sequence_chunked requires attn_implementation='eager' -- otherwise "
        "eager_attention_forward (the function this module patches) is never called."
    )
    device = input_ids.device
    seq_len = int(input_ids.shape[1])
    # Peak is one chunk's attention matrix: heads * chunk * seq_len * 4 bytes.
    # A fixed 4096 is fine at 8K but asks for 18GB at 37K (narrativeqa's tail),
    # which OOMs on an otherwise EMPTY 95GB card -- it is a single allocation,
    # not accumulated pressure. Size the chunk to a fixed memory budget instead,
    # so long documents get more, smaller chunks and short ones are unaffected.
    _heads = int(getattr(model.config, "num_attention_heads", 32))
    _budget = 4 * 1024 ** 3
    chunk_size = max(256, min(int(chunk_size), _budget // max(1, _heads * seq_len * 4)))
    accum_query: dict[int, torch.Tensor] = {}
    accum_hidden: dict[int, torch.Tensor] = {}
    latest_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    capture = SequenceCapture()
    cleanup = install_capture_hooks(model, model_type, capture)
    try:
        past = DynamicCache()
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            capture.clear()
            position_ids = torch.arange(start, end, device=device).unsqueeze(0)
            model(
                input_ids=input_ids[:, start:end],
                past_key_values=past,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            for layer_idx, (q, k, v) in capture.qkv.items():
                accum_query[layer_idx] = q if layer_idx not in accum_query else torch.cat([accum_query[layer_idx], q], dim=2)
                latest_kv[layer_idx] = (k, v)  # already the full accumulated cache-to-date
            if keep_hidden:
                for layer_idx, h in capture.hidden_in.items():
                    accum_hidden[layer_idx] = h if layer_idx not in accum_hidden else torch.cat([accum_hidden[layer_idx], h], dim=1)
    finally:
        cleanup()

    out = SequenceCapture()
    for layer_idx, q in accum_query.items():
        k, v = latest_kv[layer_idx]
        out.qkv[layer_idx] = (q[0], k[0], v[0])
    for layer_idx, h in accum_hidden.items():
        out.hidden_in[layer_idx] = h[0]
    return out


def protected_positions(t: int, cfg: CandidateConfig) -> set[int]:
    """Positions protected at decision point t (0-indexed, t itself included in cache)."""
    sink = set(range(min(cfg.sink_tokens, t + 1)))
    recent_start = max(0, t + 1 - cfg.recent_tokens)
    recent = set(range(recent_start, t + 1))
    return sink | recent


def build_candidate_indices(t: int, cfg: CandidateConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """Evictable candidates at decision point t: cached positions [0, t] minus
    protected sink/recent-window (spec section 2). Subsampled to
    max_candidates if larger (spec section 23; V1 uses uniform sampling)."""
    protected = protected_positions(t, cfg)
    pool = [i for i in range(t + 1) if i not in protected]
    if len(pool) > cfg.max_candidates:
        idx = torch.randperm(len(pool), generator=generator)[: cfg.max_candidates]
        idx, _ = torch.sort(idx)
        pool = [pool[i] for i in idx.tolist()]
    return torch.tensor(pool, dtype=torch.long)


def iter_eviction_steps(seq_len: int, cfg: CandidateConfig):
    """Eviction decision points t (spec section 10: not every token, only
    every step_stride once the cache first exceeds budget)."""
    first = max(cfg.budget, cfg.sink_tokens + cfg.recent_tokens)
    t = first
    while t < seq_len - 1:  # need at least 1 future step to form a horizon
        yield t
        t += cfg.step_stride
