#!/usr/bin/env python
"""Bit-exact regression check for edits to the eviction path, on CPU.

Every number in the paper comes out of ``src/method/eviction.py``, so a change
there has to be shown not to change what the code selects. Running a real
benchmark cell needs a GPU and the better part of an hour; this drives the same
code path with a tiny randomly initialised model, which catches a changed
selection because the comparison is bit-exact rather than statistical.

    python tools/regression_check.py --save baseline.json    # before editing
    python tools/regression_check.py --check baseline.json   # after editing

What it pins, per policy: a hash of the compressed cache itself -- every
retained key and value tensor, layer by layer -- plus its shape. Generated text
would be a weak fingerprint here, because an untrained model emits the same
tokens whatever the cache holds; the selection is what matters and it is what
is compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Pin to CPU before torch is imported, so the check is runnable while the GPUs
# are busy -- which is exactly when a refactor is convenient to make.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# The base policies, plus RESCUE on each of them. The scorer is a per-entry
# MLP over 18 features, so a shipped checkpoint drives the tiny model here
# unchanged -- which matters, because the rfc path is the one a refactor is
# most likely to touch.
BASE_POLICIES = ["full", "snapkv", "laprox", "h2o", "lava", "rkv", "streamingllm"]
RESCUE_BASES = ["snapkv", "laprox", "h2o", "lava", "rkv"]
POLICIES = BASE_POLICIES + [f"rescue:{b}" for b in RESCUE_BASES]
CKPT = REPO_ROOT / "results" / "checkpoints"
SEED = 0
PROMPT = " ".join(f"token{i % 97}" for i in range(600))


class ByteTokenizer:
    """Deterministic and dependency-free.

    A checkout with no network access still runs this, and the ids do not
    depend on a downloaded vocabulary, so a baseline saved on one machine is
    comparable on another.
    """

    eos_token_id = None
    pad_token_id = 0
    # LaProx compares ids against the vocabulary size to find chat-control
    # tokens to protect. A byte tokenizer has none, so nothing is protected --
    # the honest answer rather than a stub that pretends otherwise.
    vocab_size = 256

    def __call__(self, text, return_tensors=None, truncation=False,
                 add_special_tokens=True):
        import torch
        ids = [b % 256 for b in text.encode("utf-8")][:1024]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def apply_chat_template(self, msgs, **kw):
        return msgs[0]["content"]


def tiny_model():
    """Two Llama layers with GQA, small enough to run on a CPU in seconds."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(SEED)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


def walk_tensors(obj, depth: int = 0):
    """Yield every tensor reachable from a cache object, in a stable order."""
    import torch
    if depth > 4:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            yield from walk_tensors(obj[k], depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from walk_tensors(item, depth + 1)
        return
    for name in ("key_cache", "value_cache", "keys", "values", "layers"):
        if hasattr(obj, name):
            yield from walk_tensors(getattr(obj, name), depth + 1)


def install_recorder(record) -> list:
    """Wrap every state class so each compressed cache is seen exactly once.

    Policies reach their pruning through different classes -- the per-head path
    goes through DynamicEvictionState, the ragged ones through their own -- so
    hooking one of them would silently skip the others.
    """
    from src.method import eviction as E

    undo = []
    targets = [
        (E.DynamicEvictionState, "prune_after_prefill"),
        (E.RaggedLaProxState, "initialize_after_prefill"),
        (E.RaggedLaVaState, "initialize_after_prefill"),
        (E.RaggedRKVState, "initialize_after_prefill"),
    ]
    for cls, name in targets:
        original = getattr(cls, name)

        def wrapper(self, *a, __orig=original, **kw):
            out = __orig(self, *a, **kw)
            record(out)
            return out

        setattr(cls, name, wrapper)
        undo.append((cls, name, original))
    return undo


def config_for(policy: str):
    """Translate a harness policy name into a BaselineConfig."""
    from src.utils.scoring import BaselineConfig

    base = None
    if policy.startswith("rescue:"):
        base = policy.split(":", 1)[1]
        policy = "rfc"
    common = dict(
        policy=policy, budget_tokens=64, sink_tokens=4, recent_tokens=16,
        snapkv_obs_window=8, snapkv_kernel_size=7, snapkv_pooling="avgpool",
        laprox_allocation="global_layer", rkv_window_size=8,
        rkv_mix_lambda=0.07, rkv_kernel_size=7, rkv_retain_ratio=0.1,
        rkv_retain_direction="last", evict_during_decode=False,
        use_chat_template=False,
    )
    if base is None:
        return BaselineConfig(**common)
    ckpt = CKPT / f"rescue_llama3_8b_{base}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}")
    return BaselineConfig(
        **common,
        learned_checkpoint=str(ckpt),
        rfc_objective="RESCUE",
        rfc_use_impact=False,
        rfc_allocation="global_layer" if base == "laprox" else "per_head",
        rfc_recent_style=base,
        rfc_combine_mode="scale_additive",
        rfc_lambda_select="fidelity",
        rfc_fidelity_lambdas="0,1",
        rfc_fidelity_probe_len=1,
    )


def fingerprint(policy: str) -> dict:
    import torch
    from src.method.eviction import DenseEvictionGenerator

    shapes: list[list[int]] = []
    digest = hashlib.sha256()

    def record(cache) -> None:
        """Hash every tensor in the returned cache.

        The ragged policies keep a per-head variable-length cache that
        legacy_layers() does not enumerate, so the object is walked generically
        rather than assumed to be a list of (key, value) pairs.
        """
        seen = 0
        for t in walk_tensors(cache):
            shapes.append(list(t.shape))
            digest.update(t.detach().to(torch.float32).numpy().tobytes())
            seen += 1
        if not seen:
            digest.update(repr(type(cache)).encode())

    undo = install_recorder(record)
    try:
        torch.manual_seed(SEED)
        gen = DenseEvictionGenerator(tiny_model(), ByteTokenizer(),
                                     config_for(policy))
        with torch.no_grad():
            gen.generate([PROMPT], max_new_tokens=8, max_seq_len=2048)
    finally:
        for cls, name, original in undo:
            setattr(cls, name, original)

    return {"cache_digest": digest.hexdigest()[:16], "shapes": shapes}


def collect() -> dict:
    out = {}
    for p in POLICIES:
        try:
            out[p] = fingerprint(p)
        except Exception as exc:                       # noqa: BLE001
            out[p] = {"error": f"{type(exc).__name__}: {exc}"}
        row = out[p]
        if "error" in row:
            print(f"  {p:16s} {row['error'][:70]}")
        else:
            n = sum(int(x[-2]) for x in row["shapes"] if len(x) >= 2)
            print(f"  {p:16s} {row['cache_digest']}  {len(row['shapes'])} tensors, "
                  f"{n} positions retained in total")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", type=Path, help="write a baseline")
    g.add_argument("--check", type=Path, help="compare against a baseline")
    args = ap.parse_args()

    got = collect()
    if args.save:
        args.save.write_text(json.dumps(got, indent=2), encoding="utf-8")
        print(f"\nbaseline -> {args.save}")
        return

    want = json.loads(args.check.read_text(encoding="utf-8"))
    changed = [p for p in POLICIES if want.get(p) != got.get(p)]
    if changed:
        print(f"\nCHANGED: {', '.join(changed)}")
        for p in changed:
            print(f"  {p}\n    before {want.get(p)}\n    after  {got.get(p)}")
        raise SystemExit(1)
    print(f"\nidentical across {len(POLICIES)} policies")


if __name__ == "__main__":
    main()
