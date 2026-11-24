"""Delayed eviction: let the base policy see w real decode steps before it prunes.

The same question arrives from two directions. One asks for a
baseline that simply postpones compression, the other observation is that adding the
probe token's own attention row to SnapKV's vote already beats RESCUE on
average. Both are really asking whether RESCUE's gain is anything more than
"having seen some of the future", and the control that answers it is this: run
the model on the UNCOMPRESSED cache for w decode steps, then evict with those
w steps' attention included in the evidence, then carry on.

Implementation. The eviction machinery already does exactly the right thing
when the w tokens are part of the prompt: it prefills, scores every position
from the attention over the whole sequence, prunes once, and decodes. So a
delay of w is a prompt of (prompt + the w tokens the dense model would have
emitted), evaluated for (max_new_tokens - w) further steps. The policy's
observation window then covers real generated tokens instead of the prompt's
tail, which is the whole point of the control. Nothing in the shipped
generator changes -- this subclass only chooses what sequence it is handed,
which is why the arm is comparable with the paper's own numbers.

Two honest consequences, both of which belong in the answer rather than being
engineered away:

  * This does not save memory. Holding the full cache for w decode steps makes
    peak KV usage the dense peak, which is the cost eviction exists to avoid.
    Delayed eviction is a probe of where the gain comes from, not a rival
    method.
  * On a task whose answer is shorter than w, generation ends inside the
    window and no eviction ever happens: the document ran dense. Six of the
    sixteen LongBench tasks the paper reports have a 32-token output budget,
    so a w of 32 is dense on all of them. Every document reports which case it
    fell into on a [DELAY] line, so the degenerate share is measured rather
    than assumed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import torch

REL_ROOT = Path(os.environ.get("RESCUE_RELEASE_ROOT",
                               "/path/to/kv_cache/rescue-release"))
if str(REL_ROOT) not in sys.path:
    sys.path.insert(0, str(REL_ROOT))

from rescue.eviction import DenseEvictionGenerator  # noqa: E402
from scripts.oc_adapter import KVCacheEvictionHF  # noqa: E402

from opencompass.registry import MODELS  # noqa: E402


class _ForcedIds:
    """Hands the wrapped generator one exact id sequence, then steps aside.

    ``generate_one`` tokenizes its prompt itself, applying the chat template
    and the middle-truncation rule. The delayed arm has already done both --
    the lead tokens were generated from precisely those ids -- and re-deriving
    them from a decoded string would not round-trip. So the tokenizer is
    swapped for this proxy across the one call: the first ``__call__`` returns
    the prepared ids and every later call, and every attribute, goes to the
    real tokenizer.
    """

    def __init__(self, inner, input_ids: torch.Tensor):
        self._inner = inner
        self._ids = input_ids
        self._used = False

    def __call__(self, *args, **kwargs):
        if not self._used:
            self._used = True
            return {"input_ids": self._ids}
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class DelayedEvictionGenerator(DenseEvictionGenerator):
    """DenseEvictionGenerator that compresses w decode steps late."""

    def __init__(self, *args, delay_window: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_window = int(delay_window)
        self.delay_records: list[dict] = []

    @torch.inference_mode()
    def _lead(self, prompt: str, w: int, max_new_tokens: int, max_seq_len: int):
        """The first w tokens, decoded on the uncompressed cache.

        Tokenization and truncation copy ``generate_one``'s own rules under the
        ORIGINAL max_new_tokens, so the prompt this returns is the one the
        undelayed arm would have evicted on, position for position.
        """
        device = next(self.model.parameters()).device
        input_budget = max(1, int(max_seq_len) - int(max_new_tokens))
        if bool(self.cfg.use_chat_template) and hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
            enc = self.tokenizer(text, return_tensors="pt", truncation=False,
                                 add_special_tokens=False)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        ids = enc["input_ids"]
        if int(ids.shape[1]) > input_budget:
            half = input_budget // 2
            ids = torch.cat([ids[:, :half], ids[:, -(input_budget - half):]], dim=1)
        ids = ids.to(device)

        # The model is loaded with eager attention because the policies need raw
        # attention weights, but this pass never reads them and eager's full
        # [heads, seq, seq] softmax OOMs on a 64K document. Same swap, and same
        # reason, as _prime_oracle_reference's.
        prev_impl = self.model.config._attn_implementation
        try:
            self.model.config._attn_implementation = "sdpa"
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=int(w),
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=(self.tokenizer.eos_token_id
                              if self.tokenizer.pad_token_id is None
                              else self.tokenizer.pad_token_id),
            )
        finally:
            self.model.config._attn_implementation = prev_impl
        return ids, out[:, int(ids.shape[1]):]

    def generate_one(self, prompt: str, max_new_tokens: int, max_seq_len: int) -> str:
        w = self.delay_window
        if w <= 0:
            return super().generate_one(prompt, max_new_tokens, max_seq_len)
        # At least one step has to happen on the compressed cache, or the arm
        # is not measuring eviction at all.
        w = min(w, max(0, int(max_new_tokens) - 1))
        if w <= 0:
            return super().generate_one(prompt, max_new_tokens, max_seq_len)

        prompt_ids, lead = self._lead(prompt, w, max_new_tokens, max_seq_len)
        lead_ids = lead[0].tolist()
        eos = self.tokenizer.eos_token_id
        stopped = eos is not None and eos in lead_ids
        rec = dict(w=w, lead=len(lead_ids), prompt_len=int(prompt_ids.shape[1]),
                   dense=bool(stopped))
        self.delay_records.append(rec)
        print(f"[DELAY] w={w} prompt={rec['prompt_len']} lead={rec['lead']} "
              f"dense={int(stopped)}", flush=True)

        if stopped:
            # Generation finished inside the window: the cache was never pruned,
            # so this document is a dense one. Reported, not hidden.
            return self.tokenizer.decode(lead_ids[:lead_ids.index(eos)],
                                         skip_special_tokens=True)

        ids = torch.cat([prompt_ids, lead], dim=1)
        # input_budget in the call below is max_seq_len - (max_new_tokens - w),
        # i.e. w longer than the one _lead truncated to, so these ids pass
        # through generate_one's truncation untouched.
        # BaselineConfig is frozen, so the template is turned off by swapping
        # the generator's whole config for a copy -- the same handle the
        # fidelity selector already swaps when it walks its candidate lambdas.
        prev_tok, prev_cfg = self.tokenizer, self.cfg
        self.tokenizer = _ForcedIds(prev_tok, ids)
        self.cfg = replace(prev_cfg, use_chat_template=False)
        try:
            rest = super().generate_one(prompt, int(max_new_tokens) - w, max_seq_len)
        finally:
            self.tokenizer, self.cfg = prev_tok, prev_cfg
        return self.tokenizer.decode(lead_ids, skip_special_tokens=True) + rest


@MODELS.register_module()
class DelayedKVCacheEvictionHF(KVCacheEvictionHF):
    """The shipped adapter with the delayed generator in place of the plain one."""

    def __init__(self, kv_delay_window: int = 8, **kwargs):
        super().__init__(**kwargs)
        if getattr(self, "generator", None) is not None:
            self.generator = DelayedEvictionGenerator(
                self.model, self.tokenizer, self.eviction_cfg,
                delay_window=int(kv_delay_window),
            )
