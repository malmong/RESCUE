"""RESCUE: a policy-conditioned residual scorer for KV cache eviction, and a
fidelity-guided selector that decides per input whether to apply it.

    eviction    the generation loop and every eviction policy's prefill hook,
                including the selector
    features    the 18 features the scorer reads (raw 9 + their within-layer ranks)
    scorer      the residual scorer itself
    policies    the base policies RESCUE is ported onto, and their scoring
    models      the model registry, backed by configs/*.yaml
"""
from rescue.scorer_inference import LearnedScorer
from rescue.mlp import LearnedPolicyConfig, LearnedRanker

__all__ = ["LearnedPolicyConfig", "LearnedRanker", "LearnedScorer"]
