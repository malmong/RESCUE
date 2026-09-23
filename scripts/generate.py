#!/usr/bin/env python
"""Generate from one prompt under a chosen eviction policy.

This is the smallest path through the method: no benchmark harness, no
OpenCompass. It loads a model, compresses the prompt cache once after prefill,
and decodes -- which is what every number in the paper is measured on, one
document at a time.

    python scripts/generate.py --model llama3_8b --method snapkv \
        --prompt-file doc.txt --question "Who signed the treaty?"

    python scripts/generate.py --model llama3_8b --method rescue --base snapkv \
        --checkpoint results/checkpoints/rescue_llama3_8b_snapkv.pt \
        --prompt-file doc.txt --question "Who signed the treaty?" --verbose

``--verbose`` prints what the selector decided per document: the KL of each
candidate cache against the dense one, and which lambda won -- or that it
abstained because no candidate produced measurable evidence.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rescue.models import available, load  # noqa: E402

# LookaheadKV is omitted here on purpose: it patches the model class and runs
# its own decoder, so it does not go through this path. Use
# scripts/evaluate.py for it.
METHODS = {
    "dense": "full", "snapkv": "snapkv", "laprox": "laprox", "h2o": "h2o",
    "lava": "lava", "rkv": "rkv", "streamingllm": "streamingllm",
    "foresightkv": "foresightkv", "rescue": "rfc",
}


def build_config(args, cfg):
    """Translate the YAML config plus the command line into a BaselineConfig."""
    from rescue.policies.scoring import BaselineConfig

    rescue = cfg.rescue
    base = args.base or "snapkv"
    lambdas = args.lambdas or ",".join(f"{x:g}" for x in rescue["fidelity_lambdas"])
    return BaselineConfig(
        policy=METHODS[args.method],
        budget_tokens=args.budget_tokens or cfg.budget["total_tokens"],
        sink_tokens=cfg.budget["sink_tokens"],
        recent_tokens=cfg.budget["recent_tokens"],
        snapkv_obs_window=cfg.snapkv["obs_window"],
        snapkv_kernel_size=cfg.snapkv["kernel_size"],
        snapkv_pooling=cfg.snapkv["pooling"],
        laprox_allocation="global_layer",
        rkv_window_size=cfg.rkv["window_size"],
        rkv_mix_lambda=cfg.rkv["mix_lambda"],
        rkv_kernel_size=cfg.rkv["kernel_size"],
        rkv_retain_ratio=cfg.rkv["retain_ratio"],
        rkv_retain_direction=cfg.rkv["retain_direction"],
        evict_during_decode=False,
        use_chat_template=not args.no_chat_template,
        learned_checkpoint=(str(Path(args.checkpoint).expanduser().resolve())
                            if args.checkpoint else None),
        foresight_recent_window=cfg.future_aware["foresight_recent_window"],
        rfc_objective=rescue["objective"],
        rfc_use_impact=not rescue["no_impact"],
        rfc_allocation=cfg.allocation(base),
        rfc_recent_style=base,
        rfc_combine_mode=rescue["combine_mode"],
        rfc_lambda_select=rescue["lambda_select"],
        rfc_fidelity_lambdas=lambdas,
        rfc_fidelity_probe_len=args.probe_len or rescue["fidelity_probe_len"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=available())
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--base", choices=["snapkv", "laprox", "h2o", "lava", "rkv"],
                   help="--method rescue: the base policy being corrected")
    p.add_argument("--checkpoint", help="--method rescue: residual scorer weights")
    p.add_argument("--prompt", help="prompt text")
    p.add_argument("--prompt-file", type=Path, help="read the prompt from a file")
    p.add_argument("--question", default="",
                   help="appended after the document, as LongBench's own templates do")
    p.add_argument("--budget-tokens", type=int)
    p.add_argument("--lambdas", help="override the selector's candidates, e.g. '1'")
    p.add_argument("--probe-len", type=int)
    p.add_argument("--kl-floor", type=float,
                   help="selector abstention floor; 0 disables abstention")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--verbose", action="store_true", help="print the selector's decision")
    args = p.parse_args()

    if args.method == "rescue" and not (args.base and args.checkpoint):
        raise SystemExit("--method rescue needs --base and --checkpoint")
    if bool(args.prompt) == bool(args.prompt_file):
        raise SystemExit("pass exactly one of --prompt / --prompt-file")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    cfg = load(args.model)
    floor = args.kl_floor if args.kl_floor is not None else cfg.rescue["kl_floor"]
    os.environ["RESCUE_KL_FLOOR"] = f"{floor:g}"
    if args.verbose:
        os.environ["RESCUE_GATE_LOG"] = "1"

    import torch  # noqa: E402  (after CUDA_VISIBLE_DEVICES is pinned)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    from rescue.eviction import DenseEvictionGenerator  # noqa: E402

    text = args.prompt if args.prompt else args.prompt_file.read_text(encoding="utf-8")
    if args.question:
        text = f"{text}\n\n{args.question}"

    print(f"loading {cfg.path} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(cfg.path))
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.path), torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=cfg.attn_implementation,
    ).eval()

    gen = DenseEvictionGenerator(model, tok, build_config(args, cfg))
    out = gen.generate([text], max_new_tokens=args.max_new_tokens,
                       max_seq_len=cfg.max_seq_len)
    print(out[0])


if __name__ == "__main__":
    main()
