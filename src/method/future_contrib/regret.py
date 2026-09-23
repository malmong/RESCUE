"""Eviction regret via gradient-based sensitivity (objective R, priority 6:
"train against the actual downstream cost of eviction instead of an
attention-magnitude proxy"). Every other objective in this package (A-H)
targets some function of real future ATTENTION -- the oracle experiment
proved that target is the right one in principle (45.82 ceiling), but three
separate predictors that improved offline attention-recall metrics (h32,
active-pattern H, and global_layer's own recency formula at lambda=0) all
failed to move the downstream qasper score. This objective instead targets
each candidate's actual marginal effect on generation LOSS.

The literal target -- re-run teacher-forcing once per candidate with that
candidate's KV removed, measure the NLL increase -- costs one forward pass
per candidate (hundreds per document), intractable at this project's scale.
Instead: a single forward+backward pass per document, reading the gradient
of the whole-sequence causal-LM loss w.r.t. every layer's raw (post-softmax)
attention weights. First-order Taylor expansion: zeroing out attn_weight
a_{q,i} (the effect of literally evicting key i, from query q's perspective)
changes the loss by approximately grad(a_{q,i}) * (0 - a_{q,i}) =
-grad(a_{q,i}) * a_{q,i}. Summing this over every query q that could attend
to key i (the causal mask already restricts this to q >= i) gives a
per-(layer, kv_head, key-position) regret estimate for the ENTIRE rest of
the document, in one backward pass instead of one per candidate.

Base model weights stay frozen (no .grad allocated for them, no optimizer
state) -- only the input embeddings are given requires_grad=True, purely to
give autograd a path to build the graph up to the attention-weight tensors."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_regret_via_grad(
    model,
    input_ids: torch.Tensor,
    num_layers: int,
    num_kv_heads: int,
    kv_repeat: int,
) -> list[torch.Tensor]:
    """input_ids: [1, seq_len]. Returns num_layers tensors, each
    [num_kv_heads, seq_len] non-negative regret scores in absolute sequence
    position order (query-group heads already summed into their kv_head,
    exactly like future_attention_distribution's head aggregation
    convention) -- caller slices by candidate position same as
    compute_output_contribution's output."""
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        embeds = model.get_input_embeddings()(input_ids).detach()
        embeds.requires_grad_(True)
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions  # tuple of [1, num_heads, seq_len, seq_len], graph-connected
        for attn in attentions:
            attn.retain_grad()
        logits = outputs.logits
        loss = F.cross_entropy(logits[0, :-1, :].float(), input_ids[0, 1:])
        loss.backward()

    regrets: list[torch.Tensor] = []
    with torch.no_grad():
        for layer_idx in range(num_layers):
            attn = attentions[layer_idx][0].float()  # [num_heads, seq_len, seq_len]
            grad = attentions[layer_idx].grad
            if grad is None:
                regrets.append(torch.zeros(num_kv_heads, attn.shape[-1], device=attn.device))
                continue
            grad = grad[0].float()
            sensitivity = (-grad * attn).clamp_min(0.0)  # [num_heads, q_len, k_len]
            per_key = sensitivity.sum(dim=1)  # sum over query positions -> [num_heads, k_len]
            seq_len = per_key.shape[-1]
            per_key = per_key.view(num_kv_heads, kv_repeat, seq_len).sum(dim=1)  # group query heads -> [num_kv_heads, k_len]
            regrets.append(per_key)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return regrets
