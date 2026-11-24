#!/usr/bin/env python
"""Run one LU-KV cell, reusing the shipped evaluate.py end to end.

Same trick as followups/scripts/delayed/evaluate_delayed.py: evaluate.py builds the
OpenCompass config, and this driver rewrites only the two lines that name the
model class, so the dataset block, the budget, the selector flags and the
chat-template split are whatever the paper's own driver would have produced.
Run the base arm with --method lava, and the stacked arm with
--method rescue --base lava --checkpoint <the SnapKV scorer for this budget>:
LU-KV selects within a head with SnapKV, so the correction is the SnapKV one.

    python followups/scripts/lukv/evaluate_lukv.py --model llama3_8b \
        --method lava --budget-tokens 256 --task all --gpu 5
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = Path(os.environ.get("RESCUE_RELEASE_ROOT",
                          "/path/to/kv_cache/rescue-release"))
sys.path.insert(0, str(REL))


def _evaluate_module():
    """evaluate.py as a module. It guards its own main(), so this is safe."""
    spec = importlib.util.spec_from_file_location(
        "rescue_evaluate", REL / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _evaluate_module()


def patch(text: str) -> str:
    """Point the generated config at the LU-KV adapter."""
    subs = [
        ("from scripts.oc_adapter import KVCacheEvictionHF",
         f"_sys.path.insert(0, {str(HERE)!r})\n"
         "from lukv_eviction import LUKVEvictionHF"),
        ("type=KVCacheEvictionHF,", "type=LUKVEvictionHF,"),
    ]
    for old, new in subs:
        if old not in text:
            raise SystemExit(
                f"evaluate.py's config template no longer contains {old!r}; "
                "this driver rewrites it and must be updated with it.")
        text = text.replace(old, new, 1)
    return text


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=EV.available())
    p.add_argument("--method", required=True, choices=sorted(EV.METHODS))
    p.add_argument("--task", default="all")
    p.add_argument("--base", choices=["snapkv", "laprox", "h2o", "lava", "rkv"])
    p.add_argument("--checkpoint")
    p.add_argument("--budget-tokens", type=int)
    p.add_argument("--gpu", default="0")
    p.add_argument("--tag", default="")
    p.add_argument("--work-dir", default=str(HERE / "runs"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    # build_config and run_name read the shipped driver's own argument names,
    # so hand them a namespace shaped like its parse_args output.
    args = Namespace(
        model=a.model, method=a.method, task=a.task, base=a.base,
        checkpoint=a.checkpoint, budget_tokens=a.budget_tokens,
        lambdas=None, probe_len=None, kl_floor=None, gpu=a.gpu,
        tag="-".join(x for x in ("lukv", a.tag) if x),
        work_dir=a.work_dir, dry_run=a.dry_run,
    )
    cfg = EV.load(args.model)
    if args.method == "rescue" and not args.base:
        raise SystemExit("--method rescue needs --base")
    if args.method in ("rescue",) and not args.checkpoint:
        raise SystemExit("--method rescue needs --checkpoint")
    if args.checkpoint and not Path(args.checkpoint).expanduser().exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    tasks = list(EV.LONGBENCH_TASKS) if args.task == "all" else [args.task]
    unknown = [t for t in tasks if t not in EV.LONGBENCH_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}")

    oc = EV.opencompass_root()
    if not (oc / "run.py").exists():
        raise SystemExit(f"OpenCompass not found at {oc}; set OPENCOMPASS_ROOT.")

    budget = args.budget_tokens or cfg.budget["total_tokens"]
    name = EV.run_name(args, cfg, budget)

    groups: dict[bool, list[str]] = {}
    for t in tasks:
        groups.setdefault(cfg.uses_chat_template(t), []).append(t)
    labels = ({True: "all-chat", False: "all-raw"} if args.task == "all"
              else {True: args.task, False: args.task})

    cfg_dir = HERE / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for chat in sorted(groups, reverse=True):
        group = groups[chat]
        cfg_path = cfg_dir / f"{name}-{labels[chat]}.py"
        cfg_path.write_text(patch(EV.build_config(args, cfg, group, budget, name)),
                            encoding="utf-8")
        work_dir = Path(args.work_dir) / f"{name}-{labels[chat]}"
        print("config:  ", cfg_path)
        print("work_dir:", work_dir)
        jobs.append((cfg_path, work_dir))
    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # The OpenCompass worker is a separate process that reads a DUMPED config,
    # in which the model type has become the string "lukv_eviction.LUKVEvictionHF" and the config's own sys.path lines are gone.
    # mmengine resolves that string by importing the module (Registry.get's
    # get_object_from_string fallback), so this directory has to be importable
    # there. The shipped adapter resolves the same way, through REL.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(REL), str(oc), env.get("PYTHONPATH", "")])
    env["RESCUE_KL_FLOOR"] = f"{cfg.rescue['kl_floor']:g}"

    for cfg_path, work_dir in jobs:
        subprocess.run(
            [sys.executable, str(oc / "run.py"), str(cfg_path),
             "--work-dir", str(work_dir)],
            cwd=str(oc), env=env, check=True,
        )


if __name__ == "__main__":
    main()
