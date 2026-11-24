"""Stage 3 of the "why isn't the probe a feature?" question -- the inference side.

The release already decodes a probe token off the dense prompt cache and keeps
its attention over the prompt: that is the RESCUE_PROBE_SNAPKV control arm,
which folds the row into SnapKV's own vote. The same row is what the probe
feature needs. So this overlay does not add a capture; it redirects the one
that exists.

Three patches, all at module import, none of them touching the checkout:

  * ``rescue.scorer.RescueScorer`` -> ``ProbeRescueScorer``. The generator
    imports that name INSIDE the branch that builds the scorer, so rebinding
    the module attribute is enough and the fidelity selector's per-lambda
    states pick it up for free.
  * ``set_probe_rows`` -> hand the rows to the scorer and pass weight 0.0 on
    to the release's own global. Weight 0 is what switches the s_recent fold
    off (scoring.py only folds when the weight is non-zero), so this arm is
    the FEATURE alone, not the feature plus the control it is compared with.
  * ``clear_probe_rows`` -> also drop our copy, so a document can never score
    on the previous document's probe.

Honest note on cost. This arm RE-decodes the probe token rather than reusing
the one the fidelity selector already decoded, because the selector's decode
happens after the prune this feature has to inform. That makes the arm's
measured latency an upper bound: what it answers is whether the scorer can use
the row, not what the row costs. The free-at-inference claim rests on the
selector's decode, which Appendix C prices separately.
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

import rescue.scorer as _scorer_mod  # noqa: E402
import rescue.policies.scoring as _scoring_mod  # noqa: E402
from probe_scorer import (ProbeRescueScorer, clear_probe_feature_rows,  # noqa: E402
                          set_probe_feature_rows)

from scripts.oc_adapter import KVCacheEvictionHF  # noqa: E402

from opencompass.registry import MODELS  # noqa: E402

# The capture is gated on both variables: N probe steps, and a non-zero
# "fold into s_recent" weight. One step is the horizon the feature was
# trained at; the weight only has to be non-zero to open the gate, because
# the patched setter below replaces it with 0.0 before the release sees it.
os.environ.setdefault("RESCUE_PROBE_SNAPKV", "1")
os.environ.setdefault("RESCUE_PROBE_IN_RECENT", "1")

_scorer_mod.RescueScorer = ProbeRescueScorer

_orig_set = _scoring_mod.set_probe_rows
_orig_clear = _scoring_mod.clear_probe_rows


def _set_probe_rows(rows, weight):          # noqa: ARG001 -- weight is deliberately dropped
    set_probe_feature_rows(rows)
    _orig_set(rows, 0.0)


def _clear_probe_rows():
    clear_probe_feature_rows()
    _orig_clear()


_scoring_mod.set_probe_rows = _set_probe_rows
_scoring_mod.clear_probe_rows = _clear_probe_rows


@MODELS.register_module()
class ProbeFeatureKVCacheEvictionHF(KVCacheEvictionHF):
    """The shipped adapter. Everything that differs is the import above."""
