"""RESCUE: policy-conditioned residual correction with a fidelity-guided selector.

``eviction`` holds the generation loop and every eviction policy's prefill hook;
``scorer``/``mlp`` hold the residual scorer; the selector lives inside
``eviction._select_lambda_by_fidelity``.
"""
from src.method.scorer_inference import LearnedScorer
from src.method.mlp import LearnedPolicyConfig, LearnedRanker

__all__ = ["LearnedPolicyConfig", "LearnedRanker", "LearnedScorer"]
