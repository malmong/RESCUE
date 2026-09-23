"""Base eviction policies and their scoring functions.

SnapKV, LaProx, H2O, LAVa and R-KV are reimplementations written against each
method's released code and defaults, not copies of it; where the code and the
paper disagree, the code's behaviour is followed and the divergence is noted at
the call site.
"""
from rescue.policies.scoring import BaselineConfig, supported_policies

__all__ = ["BaselineConfig", "supported_policies"]
