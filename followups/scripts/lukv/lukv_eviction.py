"""Overlay that puts LU-KV's head-budget allocation into the ragged-base slot.

One patch, at import, touching nothing in the checkout:

  * ``rescue.eviction.RaggedLaVaState`` -> ``LUKVState``.  The generator looks
    that name up as a module global when it builds the state for
    ``policy == "lava"`` / ``ragged_base == "lava"``, so rebinding the
    attribute is enough -- and it buys two things for free that would
    otherwise have to be re-plumbed: the ragged decode path (``ragged_decoder``
    is switched on for policy "lava") and ``RescueInjection``, the hook the
    release already uses to stack RESCUE on top of a ragged base.

So ``--base lava`` runs LU-KV here, and ``--method rescue --base lava`` runs
LU-KV + RESCUE.  Every run name is tagged ``lukv`` so no output can be
mistaken for a real LAVa cell.  Nothing else in the pipeline changes: the
SnapKV score, the protected window and the retained total are the release's
own, so the only difference from the uniform arm is which head gets how many
slots.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = Path(os.environ.get("RESCUE_RELEASE_ROOT",
                          "/path/to/kv_cache/rescue-release"))
for _p in (str(HERE), str(REL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rescue.eviction as _ev  # noqa: E402
from lukv_state import LUKVState  # noqa: E402

from scripts.oc_adapter import KVCacheEvictionHF  # noqa: E402

from opencompass.registry import MODELS  # noqa: E402

if not hasattr(_ev, "RaggedLaVaState"):
    raise RuntimeError("rescue.eviction.RaggedLaVaState is gone; this overlay "
                       "rebinds that name and must be updated with the release.")
_ev.RaggedLaVaState = LUKVState


@MODELS.register_module()
class LUKVEvictionHF(KVCacheEvictionHF):
    """The shipped adapter. Everything that differs is the patch above."""
