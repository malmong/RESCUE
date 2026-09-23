"""Recent-query state construction for FutureQueryPredictor -- shared between
training (src/method/future_contrib/train.py, computed from a full teacher-forced
capture_sequence pass) and eval-time eviction (src/dense_eviction_hf.py,
computed from a ring buffer of the last `recent_window` query vectors per
layer). Both call sites MUST use the identical grouping convention: GQA
means the GT future-attention distribution (and eval-time candidate scoring)
is per-kv-head, but query vectors come per-query-head, so query heads within
a kv-group are averaged down to one vector per kv-head before anything else
touches them. Any drift between train-time and eval-time grouping here would
silently corrupt eval-time scores without ever showing up in training loss.
"""
from __future__ import annotations

import torch


def grouped_query(q: torch.Tensor, pos_idx: list[int], num_kv_heads: int) -> torch.Tensor:
    """q: [num_heads, seq_len, head_dim] -> [num_kv_heads, len(pos_idx), head_dim],
    averaging query heads within each kv-group (matches how each kv-head's K/V is
    shared across its query group)."""
    q_group = q.shape[0] // num_kv_heads
    q_sel = q[:, pos_idx, :]
    return q_sel.view(num_kv_heads, q_group, q_sel.shape[1], -1).mean(dim=1)


def recent_query_features(q: torch.Tensor, t: int, recent_window: int, num_kv_heads: int) -> torch.Tensor:
    """Flattened recent-query window ending at (and including) position t, per
    kv-head: [num_kv_heads, recent_window*head_dim]. Left-pads by repeating the
    earliest available position when t < recent_window - 1 (early in a
    sequence). Training-time convenience: computes the window from a full
    already-captured Q tensor by slicing; eval-time uses
    flatten_query_ring_buffer below instead, since eval never has future
    positions materialized to slice from -- both must produce bit-identical
    output for the same underlying queries."""
    start = max(0, t - recent_window + 1)
    q_recent = grouped_query(q, list(range(start, t + 1)), num_kv_heads).float()  # [num_kv_heads, R', head_dim]
    return _left_pad_and_flatten(q_recent, recent_window)


def _left_pad_and_flatten(q_recent: torch.Tensor, recent_window: int) -> torch.Tensor:
    """q_recent: [num_kv_heads, R', head_dim], R' <= recent_window -> [num_kv_heads, recent_window*head_dim]."""
    r_actual = q_recent.shape[1]
    if r_actual < recent_window:
        pad = q_recent[:, :1, :].expand(-1, recent_window - r_actual, -1)
        q_recent = torch.cat([pad, q_recent], dim=1)
    return q_recent.reshape(q_recent.shape[0], -1)


def recent_query_last_and_mean(q: torch.Tensor, t: int, recent_window: int, num_kv_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-time counterpart to QueryRingBuffer.last_and_mean, computed
    from a full already-captured Q tensor (see recent_query_features for the
    flattened-window sibling used by FutureQueryPredictor). Returns
    (q_last, q_mean), each [num_kv_heads, head_dim]. Same left-pad convention
    as recent_query_features -- both MUST agree bit-for-bit with the
    eval-time ring buffer for the same underlying queries."""
    start = max(0, t - recent_window + 1)
    q_recent = grouped_query(q, list(range(start, t + 1)), num_kv_heads).float()  # [num_kv_heads, R', head_dim]
    r_actual = q_recent.shape[1]
    if r_actual < recent_window:
        pad = q_recent[:, :1, :].expand(-1, recent_window - r_actual, -1)
        q_recent = torch.cat([pad, q_recent], dim=1)
    return q_recent[:, -1, :], q_recent.mean(dim=1)


def recent_query_last_m(q: torch.Tensor, t: int, recent_window: int, m: int, num_kv_heads: int) -> torch.Tensor:
    """Training-time counterpart to QueryRingBuffer.last_m. Returns the last
    m individual (left-padded) recent queries: [num_kv_heads, m, head_dim]."""
    start = max(0, t - recent_window + 1)
    q_recent = grouped_query(q, list(range(start, t + 1)), num_kv_heads).float()
    r_actual = q_recent.shape[1]
    if r_actual < recent_window:
        pad = q_recent[:, :1, :].expand(-1, recent_window - r_actual, -1)
        q_recent = torch.cat([pad, q_recent], dim=1)
    return q_recent[:, -m:, :]


class QueryRingBuffer:
    """Eval-time recent-query state, ring-buffer-style: O(num_kv_heads *
    recent_window * head_dim) per layer, independent of sequence length (does
    NOT grow with the cache, unlike the old hidden-state z_cache this
    replaces for the reuse branch). One instance per layer."""

    def __init__(self, recent_window: int):
        self.recent_window = recent_window
        self._buf: torch.Tensor | None = None  # [num_kv_heads, <=recent_window, head_dim]

    def push(self, q_new: torch.Tensor, num_kv_heads: int) -> None:
        """q_new: [num_heads, num_new_positions, head_dim] (or [num_heads, head_dim]
        for a single new position) -- the model's own query-head-resolution
        output for whatever positions were just processed (prefill's
        observation window, or a single decode step)."""
        if q_new.dim() == 2:
            q_new = q_new.unsqueeze(1)
        grouped = grouped_query(q_new, list(range(q_new.shape[1])), num_kv_heads).float()  # [num_kv_heads, P, head_dim]
        self._buf = grouped if self._buf is None else torch.cat([self._buf, grouped], dim=1)
        if self._buf.shape[1] > self.recent_window:
            self._buf = self._buf[:, -self.recent_window :, :]

    def flatten(self) -> torch.Tensor:
        """[num_kv_heads, recent_window*head_dim], left-padded if fewer than
        recent_window positions have been pushed yet (matches training's
        left-pad convention exactly)."""
        assert self._buf is not None, "QueryRingBuffer.push must be called before flatten"
        return _left_pad_and_flatten(self._buf, self.recent_window)

    def last_and_mean(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Eval-time counterpart to recent_query_last_and_mean. Returns
        (q_last, q_mean), each [num_kv_heads, head_dim]."""
        assert self._buf is not None, "QueryRingBuffer.push must be called before last_and_mean"
        buf = self._buf
        r_actual = buf.shape[1]
        if r_actual < self.recent_window:
            pad = buf[:, :1, :].expand(-1, self.recent_window - r_actual, -1)
            buf = torch.cat([pad, buf], dim=1)
        return buf[:, -1, :], buf.mean(dim=1)

    def last_m(self, m: int) -> torch.Tensor:
        """Eval-time counterpart to recent_query_last_m: [num_kv_heads, m, head_dim]."""
        assert self._buf is not None, "QueryRingBuffer.push must be called before last_m"
        buf = self._buf
        r_actual = buf.shape[1]
        if r_actual < self.recent_window:
            pad = buf[:, :1, :].expand(-1, self.recent_window - r_actual, -1)
            buf = torch.cat([pad, buf], dim=1)
        return buf[:, -m:, :]
