#!/usr/bin/env python
"""Run one (model, policy, LongBench task) cell under OpenCompass.

Everything that is a property of the model rather than of the experiment --
context limit, protected windows, budget allocation per base policy, the
selector's settings -- comes from ``configs/<model>.yaml``, so the command line
carries only what an experiment actually varies.

    python scripts/evaluate.py --model llama3_8b --method snapkv --task qasper
    python scripts/evaluate.py --model llama3_8b --method rescue --base snapkv \
        --checkpoint results/checkpoints/rescue_llama3_8b_snapkv.pt --task qasper

``--task all`` runs all sixteen in one OpenCompass job. Scores are written to
``results/summaries/`` in the same CSV layout the table scripts read, so
``analysis/`` and ``analysis/`` work off a completed run without a GPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rescue.models import (  # noqa: E402
    LONGBENCH_TASKS,
    ModelConfig,
    available,
    data_root,
    load,
    opencompass_root,
)

# Policies as the paper names them, mapped to the generator's own identifiers.
# "rescue" is the paper's method; internally it is the "rfc" policy with the
# residual objective, which is why the two names differ.
METHODS = {
    "dense": "full",
    "snapkv": "snapkv",
    "laprox": "laprox",
    "h2o": "h2o",
    "lava": "lava",
    "rkv": "rkv",
    "streamingllm": "streamingllm",
    "lookaheadkv": "lookaheadkv",
    "foresightkv": "foresightkv",
    "rescue": "rfc",
}

DATASET_IMPORT = (
    "from opencompass.configs.datasets.longbench.longbench{t}"
    ".longbench_{t}_gen import LongBench_{t}_datasets"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=available())
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--task", default="all",
                   help="a LongBench task name, or 'all' for the sixteen the paper reports")
    p.add_argument("--base", choices=["snapkv", "laprox", "h2o", "lava", "rkv"],
                   help="--method rescue: which base policy the correction is applied to")
    p.add_argument("--checkpoint", help="--method rescue: residual scorer weights")
    p.add_argument("--budget-tokens", type=int,
                   help="override configs/<model>.yaml budget.total_tokens (the sweep uses 256 and 1024)")
    p.add_argument("--lambdas", default=None,
                   help="override the selector's candidate lambdas, e.g. '1' to apply "
                        "the correction unconditionally (the ablation's ungated arm)")
    p.add_argument("--probe-len", type=int, default=None,
                   help="override the selector's probe length (Appendix: probe length)")
    p.add_argument("--kl-floor", type=float, default=None,
                   help="override the selector's abstention floor; 0 disables abstention")
    p.add_argument("--gpu", default="0", help="a single GPU index")
    p.add_argument("--tag", default="", help="suffix appended to the run name, to keep "
                                             "two runs that differ only by checkpoint apart")
    p.add_argument("--work-dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--dry-run", action="store_true", help="write the config and stop")
    return p.parse_args()


def dataset_block(tasks: list[str]) -> str:
    imports = "\n".join("    " + DATASET_IMPORT.format(t=t) for t in tasks)
    joined = ", ".join(f"LongBench_{t}_datasets" for t in tasks)
    plural = f"sum(({joined},), [])" if len(tasks) > 1 else joined
    return (
        "from mmengine.config import read_base\n"
        "with read_base():\n"
        f"{imports}\n"
        f"datasets = {plural}\n"
        "for _dataset in datasets:\n"
        f"    _dataset['path'] = {str(data_root() / 'LongBench')!r}\n"
    )


def run_name(args: argparse.Namespace, cfg: ModelConfig, budget: int) -> str:
    parts = [cfg.name, args.method, f"b{budget}"]
    if args.method == "rescue":
        parts.append(args.base)
    if args.tag:
        parts.append(args.tag)
    return "-".join(parts)


def build_config(args: argparse.Namespace, cfg: ModelConfig, tasks: list[str],
                 budget: int, name: str) -> str:
    """Emit an OpenCompass config.

    The adapter is imported from this repo rather than from the OpenCompass
    checkout: importing it runs its ``@MODELS.register_module()``, so nothing
    inside OpenCompass has to be modified for the type to resolve.
    """
    rescue = dict(cfg.rescue)
    if args.lambdas is not None:
        rescue["fidelity_lambdas"] = [float(x) for x in args.lambdas.split(",")]
    if args.probe_len is not None:
        rescue["fidelity_probe_len"] = args.probe_len
    if args.kl_floor is not None:
        rescue["kl_floor"] = args.kl_floor
    lam = ",".join(f"{x:g}" for x in rescue["fidelity_lambdas"])

    # A task keeps the chat template only if all of the tasks in this job do;
    # `--task all` therefore runs the five raw-completion tasks the way the
    # official harness does. Single-task jobs are the paper's own unit.
    chat = all(cfg.uses_chat_template(t) for t in tasks)
    ckpt = repr(str(Path(args.checkpoint).expanduser().resolve())) if args.checkpoint else "None"

    return f"""{dataset_block(tasks)}
# The adapter lives in this repo, not inside OpenCompass, so the checkout has
# to be importable. The `sys` binding is deleted again because OpenCompass
# deepcopies this config and a module object left in its namespace is not
# picklable ("cannot pickle 'module' object").
import sys as _sys
_sys.path.insert(0, {str(REPO_ROOT)!r})
from scripts.oc_adapter import KVCacheEvictionHF
del _sys

models = [
    dict(
        type=KVCacheEvictionHF,
        abbr={name!r},
        path={str(cfg.path)!r},
        tokenizer_path={str(cfg.path)!r},
        max_seq_len={cfg.max_seq_len},
        max_out_len={cfg.max_out_len},
        batch_size=1,
        model_kwargs=dict(
            device_map="auto",
            torch_dtype="torch.bfloat16",
            attn_implementation={cfg.attn_implementation!r},
        ),
        generation_kwargs=dict(do_sample=False),
        kv_eviction_policy={METHODS[args.method]!r},
        kv_cache_budget_tokens={budget},
        kv_sink_tokens_budget={cfg.budget['sink_tokens']},
        kv_tail_tokens_budget={cfg.budget['recent_tokens']},
        kv_snapkv_obs_window={cfg.snapkv['obs_window']},
        kv_snapkv_kernel_size={cfg.snapkv['kernel_size']},
        kv_snapkv_pooling={cfg.snapkv['pooling']!r},
        kv_laprox_allocation='global_layer',
        kv_rkv_window_size={cfg.rkv['window_size']},
        kv_rkv_mix_lambda={cfg.rkv['mix_lambda']},
        kv_rkv_kernel_size={cfg.rkv['kernel_size']},
        kv_rkv_retain_ratio={cfg.rkv['retain_ratio']},
        kv_rkv_retain_direction={cfg.rkv['retain_direction']!r},
        kv_evict_during_decode=False,
        kv_use_chat_template={chat},
        learned_checkpoint={ckpt},
        kv_foresight_recent_window={cfg.future_aware['foresight_recent_window']},
        kv_lookahead_size={cfg.future_aware['lookahead_size']},
        kv_lookahead_lora_rank={cfg.future_aware['lookahead_lora_rank']},
        kv_lookahead_reduction={cfg.future_aware['lookahead_reduction']!r},
        kv_rfc_objective={rescue['objective']!r},
        kv_rfc_use_impact={not rescue['no_impact']},
        kv_rfc_allocation={cfg.allocation(args.base or 'snapkv')!r},
        kv_rfc_recent_style={(args.base or 'snapkv')!r},
        kv_rfc_combine_mode={rescue['combine_mode']!r},
        kv_rfc_lambda_select={rescue['lambda_select']!r},
        kv_rfc_fidelity_lambdas={lam!r},
        kv_rfc_fidelity_probe_len={rescue['fidelity_probe_len']},
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
"""


def collect(work_dir: Path, name: str, tasks: list[str]) -> dict[str, float]:
    """Copy this run's scores into results/summaries/<name>.csv."""
    out: dict[str, float] = {}
    for f in sorted(glob.glob(str(work_dir / "*" / "summary" / "summary_*.csv")), reverse=True):
        for row in csv.reader(open(f, encoding="utf-8")):
            if not row or not row[0].startswith("LongBench_"):
                continue
            task = row[0].removeprefix("LongBench_")
            for cell in reversed(row):
                if re.fullmatch(r"\d+\.\d+", (cell or "").strip()):
                    out.setdefault(task, float(cell))
                    break
        if out:
            break
    if out:
        dest = REPO_ROOT / "results" / "summaries" / f"{name}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "score"])
            for t in tasks:
                if t in out:
                    w.writerow([t, f"{out[t]:.2f}"])
        print(f"scores -> {dest}")
    return out


def main() -> None:
    args = parse_args()
    cfg = load(args.model)

    if args.method == "rescue":
        if not args.base:
            raise SystemExit("--method rescue needs --base")
        if not args.checkpoint:
            raise SystemExit("--method rescue needs --checkpoint")
        if not Path(args.checkpoint).expanduser().exists():
            raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    elif args.base:
        raise SystemExit("--base applies only to --method rescue")

    tasks = list(LONGBENCH_TASKS) if args.task == "all" else [args.task]
    unknown = [t for t in tasks if t not in LONGBENCH_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; choose from {', '.join(LONGBENCH_TASKS)}")

    oc = opencompass_root()
    if not (oc / "run.py").exists():
        raise SystemExit(
            f"OpenCompass not found at {oc}.\n"
            "Set OPENCOMPASS_ROOT, or run scripts/download_assets.sh to fetch it."
        )

    budget = args.budget_tokens or cfg.budget["total_tokens"]
    name = run_name(args, cfg, budget)
    suffix = "all" if args.task == "all" else args.task

    cfg_dir = REPO_ROOT / "runs" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{name}-{suffix}.py"
    cfg_path.write_text(build_config(args, cfg, tasks, budget, name), encoding="utf-8")
    work_dir = Path(args.work_dir) / f"{name}-{suffix}"
    print("config:  ", cfg_path)
    print("work_dir:", work_dir)
    if args.dry_run:
        return

    env = os.environ.copy()
    # OpenCompass re-exports CUDA_VISIBLE_DEVICES for its own worker from the
    # config's run_cfg, so the card is selected here and the worker inherits a
    # single visible device.
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(oc), env.get("PYTHONPATH", "")]
    )
    rescue = dict(cfg.rescue)
    floor = args.kl_floor if args.kl_floor is not None else rescue["kl_floor"]
    env["RESCUE_KL_FLOOR"] = f"{floor:g}"

    subprocess.run(
        [sys.executable, str(oc / "run.py"), str(cfg_path), "--work-dir", str(work_dir)],
        cwd=str(oc), env=env, check=True,
    )
    collect(work_dir, name, tasks)


if __name__ == "__main__":
    main()
