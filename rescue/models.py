"""Model registry: YAML in ``configs/`` plus the paths the host provides.

Nothing here is machine-specific. Model weights and the LongBench data are
located through environment variables (or ``--model-root`` / ``--data-root``),
so the same config file works on any host; ``scripts/download_assets.sh``
populates them from the Hugging Face Hub.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# LongBench tasks whose prompts are few-shot demonstrations or raw code. A chat
# template wraps them in a turn the format does not expect and the official
# metric then scores a wrapper, not an answer, so these five never get one --
# independently of what the caller asks for.
NO_CHAT_TEMPLATE_TASKS = frozenset(
    {"trec", "triviaqa", "samsum", "lcc", "repobench"}
)

LONGBENCH_TASKS = (
    "narrativeqa", "qasper", "multifieldqa_en",          # single-document QA
    "hotpotqa", "2wikimqa", "musique",                   # multi-document QA
    "gov_report", "qmsum", "multi_news",                 # summarization
    "trec", "triviaqa", "samsum",                        # few-shot
    "passage_count", "passage_retrieval_en",             # synthetic
    "lcc", "repobench",                                  # code
)


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# The model the paper's main grid runs on; scripts default to it, and the
# feature cache and checkpoint names carry a model suffix only for the others.
DEFAULT_MODEL = "llama3_8b"


def model_root() -> Path:
    """Where ``download_assets.sh`` puts model weights."""
    return _env_path("RESCUE_MODEL_ROOT", str(REPO_ROOT / "assets" / "models"))


def data_root() -> Path:
    """Where ``download_assets.sh`` puts LongBench."""
    return _env_path("RESCUE_DATA_ROOT", str(REPO_ROOT / "assets" / "data"))


def opencompass_root() -> Path:
    """The OpenCompass checkout used as the evaluation harness."""
    return _env_path("OPENCOMPASS_ROOT", str(REPO_ROOT / "third_party" / "opencompass"))


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    max_seq_len: int
    max_out_len: int
    attn_implementation: str
    budget: dict[str, Any] = field(default_factory=dict)
    snapkv: dict[str, Any] = field(default_factory=dict)
    rkv: dict[str, Any] = field(default_factory=dict)
    bases: dict[str, Any] = field(default_factory=dict)
    rescue: dict[str, Any] = field(default_factory=dict)
    future_aware: dict[str, Any] = field(default_factory=dict)

    # Class-level, so the notice below is printed once per interpreter rather
    # than once per document.
    _warned: ClassVar[dict[str, bool]] = {}

    @property
    def path(self) -> Path:
        """Local weights directory. A checkout under ``RESCUE_MODEL_ROOT``
        wins; otherwise the Hub id is handed to transformers as-is.

        The fallback is deliberate but expensive -- transformers will pull
        ~16 GB per model into the Hub cache -- so it says so once rather than
        appearing as an unexplained stall in the middle of a long job.
        """
        local = model_root() / self.hf_id.split("/")[-1]
        if local.exists():
            return local
        if not ModelConfig._warned.get(self.hf_id):
            ModelConfig._warned[self.hf_id] = True
            print(f"[rescue] {local} not found; falling back to the Hub id "
                  f"{self.hf_id!r}, which downloads the weights.\n"
                  f"[rescue] Point RESCUE_MODEL_ROOT at your checkout, or run "
                  f"scripts/download_assets.sh, to use local weights instead.",
                  file=sys.stderr, flush=True)
        return Path(self.hf_id)

    def allocation(self, base: str) -> str:
        return self.bases.get(base, {}).get("allocation", "per_head")

    def uses_chat_template(self, task: str) -> bool:
        return task not in NO_CHAT_TEMPLATE_TASKS


def available() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def load(name: str) -> ModelConfig:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(
            f"unknown model {name!r}; configs/ has: {', '.join(available())}"
        )
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    known = ModelConfig.__dataclass_fields__
    unknown = set(raw) - set(known)
    if unknown:
        raise SystemExit(f"{path.name}: unknown key(s) {sorted(unknown)}")
    return ModelConfig(**raw)
