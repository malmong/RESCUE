import sys
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import torch
import transformers
from safetensors.torch import load_file

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

# This module is imported by the generated OpenCompass config, which runs with
# OpenCompass's cwd, so the repo root is resolved from this file rather than
# from the process. Importing it is what registers the model type -- the config
# does the import itself, so no file inside the OpenCompass checkout is patched.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.scoring import BaselineConfig, supported_policies  # noqa: E402
from src.method.eviction import DenseEvictionGenerator  # noqa: E402

PromptType = Union[PromptList, str]


@MODELS.register_module()
class KVCacheEvictionHF(BaseModel):
    """Dense HuggingFace model wrapper with token-level KV eviction baselines.

    Supported policies:
    - full
    - streamingllm
    - h2o
    - snapkv
    - laprox
    - kvp / foresightkv: learned paper-mode checkpoints.
    - lookaheadkv: official LookaheadKV patch path for Llama/Qwen3.
    """

    is_api: bool = False

    def __init__(
        self,
        path: str,
        max_seq_len: int = 8192,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        model_kwargs: dict = dict(device_map="auto", torch_dtype="torch.bfloat16"),
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        kv_eviction_policy: str = "full",
        kv_cache_budget_ratio: float = 0.3,
        kv_cache_budget_tokens: Optional[int] = None,
        kv_sink_tokens_budget: int = 4,
        kv_tail_tokens_budget: int = 64,
        kv_snapkv_obs_window: int = 32,
        kv_snapkv_kernel_size: int = 7,
        kv_snapkv_pooling: str = "maxpool",
        kv_laprox_allocation: str = "global_layer",
        kv_evict_during_decode: bool = True,
        kv_use_chat_template: bool = False,
        kv_stats_path: Optional[str] = None,
        learned_checkpoint: Optional[str] = None,
        kv_foresight_recent_window: int = 128,
        kv_lookahead_size: int = 32,
        kv_lookahead_lora_rank: int = 8,
        kv_lookahead_reduction: str = "mean",
        kv_rfc_lambda: float = 1.0,
        kv_rfc_recent_weight: float = 1.0,
        kv_rfc_use_impact: bool = True,
        kv_rfc_impact_mode: str = "learned",
        kv_rfc_lambda_per_layer: Optional[str] = None,
        kv_rfc_objective: str = "legacy",
        kv_rfc_impact_checkpoint: Optional[str] = None,
        kv_rfc_allocation: str = "per_head",
        kv_rfc_recent_style: str = "contribution",
        kv_rfc_combine_mode: str = "additive",
        kv_rfc_eval_horizon_idx: Optional[int] = None,
        kv_rfc_eval_blend: bool = False,
        kv_rfc_layer_head_gain_path: Optional[str] = None,
        kv_rfc_adaptive_lambda: bool = False,
        kv_rfc_adaptive_a: float = 18.0,
        kv_rfc_adaptive_b: float = 55.0,
        kv_rfc_adaptive_max: float = 8.0,
        kv_rfc_adaptive_scope: str = 'layer',
        kv_rfc_adaptive_warmup: int = 4,
        kv_protect_special: bool = True,
        kv_rfc_ppl_lambda_slope: float = 0.0,
        kv_rfc_ppl_lambda_max: float = 0.0,
        kv_rfc_ppl_gate_tau: float = 0.0,
        kv_rfc_gate_scope: str = "doc",
        kv_rfc_lambda_gate_tau: float = 0.0,
        kv_rfc_rescue_floor_frac: float = 0.0,
        kv_rfc_lambda_select: str = "",
        kv_rfc_kslot: int = 16,
        kv_rfc_fidelity_lambdas: str = "0,1",
        kv_rfc_fidelity_probe_len: int = 16,
        kv_rfc_fidelity_with_gate: bool = False,
        kv_rfc_fidelity_probe: str = "gen",
        kv_future_fusion: str = "fixed",
        kv_lambda_base: float = 2.0,
        kv_confidence_alpha: float = 1.0,
        kv_agreement_beta: float = 1.0,
        kv_future_confidence_log_path: Optional[str] = None,
        kv_rkv_window_size: int = 8,
        kv_rkv_mix_lambda: float = 0.07,
        kv_rkv_kernel_size: int = 7,
        kv_rkv_retain_ratio: float = 0.1,
        kv_rkv_retain_direction: str = "last",
        tokenizer_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
        )
        self.logger = get_logger()
        self.path = path
        self.generation_kwargs = generation_kwargs or {}
        self.kv_stats_path = kv_stats_path or os.environ.get("KV_EVICTION_STATS_PATH")
        policy = str(kv_eviction_policy or "full").lower()
        if policy not in supported_policies():
            raise ValueError(f"Unsupported kv_eviction_policy={policy}; supported={sorted(supported_policies())}")
        self.eviction_cfg = BaselineConfig(
            policy=policy,
            budget_ratio=float(kv_cache_budget_ratio),
            budget_tokens=kv_cache_budget_tokens,
            sink_tokens=int(kv_sink_tokens_budget),
            recent_tokens=int(kv_tail_tokens_budget),
            foresight_recent_window=int(kv_foresight_recent_window),
            snapkv_obs_window=int(kv_snapkv_obs_window),
            snapkv_kernel_size=int(kv_snapkv_kernel_size),
            snapkv_pooling=str(kv_snapkv_pooling),
            laprox_allocation=str(kv_laprox_allocation),
            evict_during_decode=bool(kv_evict_during_decode),
            use_chat_template=bool(kv_use_chat_template),
            learned_checkpoint=learned_checkpoint,
            rfc_lambda=float(kv_rfc_lambda),
            rfc_recent_weight=float(kv_rfc_recent_weight),
            rfc_use_impact=bool(kv_rfc_use_impact),
            rfc_impact_mode=str(kv_rfc_impact_mode),
            rfc_lambda_per_layer=tuple(float(x) for x in kv_rfc_lambda_per_layer.split(",")) if kv_rfc_lambda_per_layer else None,
            rfc_objective=str(kv_rfc_objective),
            rfc_impact_checkpoint=kv_rfc_impact_checkpoint,
            rfc_allocation=str(kv_rfc_allocation),
            rfc_recent_style=str(kv_rfc_recent_style),
            rfc_combine_mode=str(kv_rfc_combine_mode),
            rfc_eval_horizon_idx=kv_rfc_eval_horizon_idx,
            rfc_eval_blend=bool(kv_rfc_eval_blend),
            rfc_layer_head_gain_path=kv_rfc_layer_head_gain_path,
            rfc_adaptive_lambda=bool(kv_rfc_adaptive_lambda),
            rfc_adaptive_a=float(kv_rfc_adaptive_a),
            rfc_adaptive_b=float(kv_rfc_adaptive_b),
            rfc_adaptive_max=float(kv_rfc_adaptive_max),
            rfc_adaptive_scope=str(kv_rfc_adaptive_scope),
            rfc_adaptive_warmup=int(kv_rfc_adaptive_warmup),
            protect_special=bool(kv_protect_special),
            rfc_ppl_lambda_slope=float(kv_rfc_ppl_lambda_slope),
            rfc_ppl_lambda_max=float(kv_rfc_ppl_lambda_max),
            rfc_ppl_gate_tau=float(kv_rfc_ppl_gate_tau),
            rfc_gate_scope=str(kv_rfc_gate_scope),
            rfc_lambda_gate_tau=float(kv_rfc_lambda_gate_tau),
            rfc_rescue_floor_frac=float(kv_rfc_rescue_floor_frac),
            rfc_lambda_select=str(kv_rfc_lambda_select),
            rfc_kslot=int(kv_rfc_kslot),
            rfc_fidelity_lambdas=str(kv_rfc_fidelity_lambdas),
            rfc_fidelity_probe_len=int(kv_rfc_fidelity_probe_len),
            rfc_fidelity_with_gate=bool(kv_rfc_fidelity_with_gate),
            rfc_fidelity_probe=str(kv_rfc_fidelity_probe),
            future_fusion=str(kv_future_fusion),
            lambda_base=float(kv_lambda_base),
            confidence_alpha=float(kv_confidence_alpha),
            agreement_beta=float(kv_agreement_beta),
            future_confidence_log_path=kv_future_confidence_log_path,
            rkv_window_size=int(kv_rkv_window_size),
            rkv_mix_lambda=float(kv_rkv_mix_lambda),
            rkv_kernel_size=int(kv_rkv_kernel_size),
            rkv_retain_ratio=float(kv_rkv_retain_ratio),
            rkv_retain_direction=str(kv_rkv_retain_direction),
        )
        tokenizer_path = tokenizer_path or path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if not tokenizer_only:
            model_kwargs = dict(model_kwargs or {})
            torch_dtype = model_kwargs.get("torch_dtype", torch.bfloat16)
            if isinstance(torch_dtype, str):
                torch_dtype = {
                    "torch.float16": torch.float16,
                    "torch.bfloat16": torch.bfloat16,
                    "torch.float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "auto": "auto",
                }.get(torch_dtype, torch.bfloat16)
                model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs.setdefault("attn_implementation", "eager")
            # A Hub id must be allowed to download; a local checkout must not go online.
            model_kwargs.setdefault("local_files_only", Path(path).is_dir())
            if policy == "lookaheadkv":
                if not learned_checkpoint:
                    raise ValueError("lookaheadkv requires --learned-checkpoint pointing to official lookahead_modules.safetensors")
                # LookaheadKV is a third-party checkout we cannot redistribute. Point
                # RESCUE_LOOKAHEADKV_ROOT at a clone of the authors' release; only
                # --method lookaheadkv needs it.
                official_root = Path(os.environ.get(
                    "RESCUE_LOOKAHEADKV_ROOT", REPO_ROOT / "third_party" / "LookaheadKV"))
                if not (official_root / "lookaheadkv.py").exists():
                    raise FileNotFoundError(
                        f"LookaheadKV not found at {official_root}. Clone the authors' "
                        "repository there or set RESCUE_LOOKAHEADKV_ROOT.")
                if str(official_root) not in sys.path:
                    sys.path.insert(0, str(official_root))
                from lookaheadkv import patch_model, set_config  # noqa: WPS433

                patch_model()
                config = transformers.AutoConfig.from_pretrained(path, local_files_only=Path(path).is_dir(), trust_remote_code=True)
                args_obj = SimpleNamespace(
                    lookahead_size=int(kv_lookahead_size),
                    lora=["q", "v", "k", "o", "up", "down", "gate"],
                    lora_rank=int(kv_lookahead_lora_rank),
                    max_capacity_prompts=int(kv_cache_budget_tokens or max(1, round(float(kv_cache_budget_ratio) * int(max_seq_len)))),
                    reduction=str(kv_lookahead_reduction),
                    pooling=str(kv_snapkv_pooling),
                    kernel_size=int(kv_snapkv_kernel_size),
                    use_cache=True,
                )
                config = set_config(config, args_obj)
                model_kwargs["config"] = config
                self.model = transformers.AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
                modules = load_file(str(learned_checkpoint))
                self.model.load_state_dict(modules, strict=False)
                self.model.eval()
                self.model.generation_config.do_sample = False
                self.generator = None
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
                self.model.eval()
                self.model.generation_config.do_sample = False
                self.generator = DenseEvictionGenerator(self.model, self.tokenizer, self.eviction_cfg)
        self.logger.info(
            "KVCacheEvictionHF loaded path=%s policy=%s budget_ratio=%s budget_tokens=%s",
            path,
            self.eviction_cfg.policy,
            self.eviction_cfg.budget_ratio,
            self.eviction_cfg.budget_tokens,
        )

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(kwargs)
        if min_out_len is not None:
            generation_kwargs["min_new_tokens"] = min_out_len
        if self.eviction_cfg.policy == "lookaheadkv":
            outputs = self._generate_lookaheadkv(inputs, int(max_out_len))
        else:
            # The custom generator is greedy and currently ignores string stopping
            # criteria. OpenCompass postprocessing still trims most benchmark outputs.
            outputs = self.generator.generate(
                inputs,
                max_new_tokens=int(max_out_len),
                max_seq_len=int(self.max_seq_len),
            )
        if self.kv_stats_path:
            stats_path = Path(self.kv_stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_path.open("a", encoding="utf-8") as f:
                for idx, record in enumerate(getattr(self.generator, "stats_records", [])):
                    payload = {
                        "model_path": self.path,
                        "input_index": idx,
                        "prompt_chars": len(inputs[idx]) if idx < len(inputs) else None,
                        **record,
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if stopping_criteria:
            for stop in stopping_criteria:
                outputs = [text.split(stop)[0] for text in outputs]
        return outputs

    @torch.inference_mode()
    def _generate_lookaheadkv(self, inputs: List[str], max_new_tokens: int) -> List[str]:
        outputs = []
        input_budget = max(1, int(self.max_seq_len) - int(max_new_tokens))
        for prompt in inputs:
            if bool(self.eviction_cfg.use_chat_template) and hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                enc = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
            else:
                enc = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = enc["input_ids"]
            if int(input_ids.shape[1]) > input_budget:
                half = input_budget // 2
                tail = input_budget - half
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -tail:]], dim=1)
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            # The model is loaded with attn_implementation="eager" globally
            # (other policies in this file need raw attention weights), but
            # LookaheadKV's own eviction decision (LookaheadKVCluster.update_kv,
            # lookaheadkv_utils.py) recomputes its own small windowed attention
            # directly via matmul -- it never reads this model's returned
            # attn_weights. Those only get retained when output_attentions=True
            # (never requested here), so eager's full [heads, seq, seq] softmax
            # on the initial full-prompt prefill is pure waste, and OOMs once
            # max_seq_len isn't truncating documents down to a few thousand
            # tokens. Swap to sdpa for the whole generate() call (prefill +
            # every decode step) -- safe since update_kv's own logic doesn't
            # depend on the model-level attn_implementation at all.
            self.model.config._attn_implementation = "sdpa"
            try:
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            finally:
                self.model.config._attn_implementation = "eager"
            outputs.append(self.tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True))
        return outputs

    def get_token_len(self, prompt: str) -> int:
        if bool(self.eviction_cfg.use_chat_template) and hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return len(self.tokenizer.encode(prompt))

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors="pt")

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_cache_stats(self) -> Dict:
        return dict(getattr(self.generator, "last_stats", {}))
