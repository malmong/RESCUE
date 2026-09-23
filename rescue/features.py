"""Full-scale listwise RESCUE training: NQ (all 260 docs, QA) + arXiv (all 260
docs, summarization) combined, full-population softmax+KL (same recipe as
rescue_train_listwise_nq.py), but:
  - ALL documents from both corpora (no scaling down) -- per explicit request,
    this is expected to take hours (full LLM forward pass per doc per epoch,
    single GPU).
  - 18 features (raw9 + rank9), NOT the raw+z(18) combo that won the earlier
    BINARY-classification ablation -- a fresh permutation-importance check on
    the just-trained listwise_nq checkpoint (27 features) found z-features
    strongly NEGATIVE for this objective (group total -1.76) and rank
    features strongly POSITIVE (+2.09), the opposite of the binary-
    classification finding. See listwise_feature_importance_result.json.
  - Output checkpoint is a NEW filename (rescue_scorer_listwise_full.pt) --
    does NOT touch rescue_scorer_listwise_nq.pt or its epoch checkpoints
    (already backed up separately to best_checkpoints_backup/ as well).

Usage: python rescue_train_listwise_full.py <gpu>
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rescue.future_contrib.extract import capture_sequence_chunked  # noqa: E402
from rescue.models import load as load_model_config  # noqa: E402

SCRATCH = Path(os.environ.get("RESCUE_SCRATCH", REPO_ROOT / "runs" / "scratch"))
NQ_JSONL = SCRATCH / "nq_corpus.jsonl"
ARXIV_JSONL = SCRATCH / "arxiv_corpus.jsonl"
NQ_PROMPT_TMPL = (
    'You are given a Wikipedia article and a question. Answer the question as concisely as you can, '
    'using a single phrase or sentence if possible. Do not provide any explanation.\n\nArticle: {context}\n\n '
    'Answer the question based on the above article as concisely as you can, using a single phrase or '
    'sentence if possible. Do not provide any explanation.\n\nQuestion: {input}\nAnswer:'
)
ARXIV_PROMPT_TMPL = (
    'You are given the body text of a scientific paper. Write an abstract that summarizes its main topic, '
    'methods, and findings in a few sentences.\n\nPaper: {context}\n\n{input}\nAbstract:'
)
NQ_MAX_OUT_LEN = 128
ARXIV_MAX_OUT_LEN = 256
SINK_TOKENS = 4
OBS_WINDOW = 32
BUDGET_TOKENS = 128
ORACLE_HORIZON = 256
MIN_HORIZON = 16
RAW_ORDER = ["current_qk", "mean_qk", "max_qk", "var_qk", "slope_qk",
             "k_norm", "v_norm", "vwo_norm", "age"]
ALL_FEATURES = RAW_ORDER + [f"rk_{k}" for k in RAW_ORDER]  # raw9 + rank9 = 18 (NOT raw+z -- see docstring)
N_FEAT = 18
EPOCHS = 15
LR = 1e-3
N_VAL_DOCS = 40  # split roughly evenly across both corpora by the shuffle
CKPT_EVERY = 1
MAX_SEQ_LEN = 40000
OUT_CKPT = SCRATCH / "rescue_scorer_listwise_full.pt"
LOG_PATH = SCRATCH / "rescue_train_listwise_full.log.jsonl"


class RescueMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def oracle_head_distribution(q_full, k_full, cand, horizon_start, horizon_end, head_group, kv_repeat, scaling):
    device = q_full.device
    cand = cand.to(device)
    abs_q_pos = torch.arange(horizon_start, horizon_end, device=device).unsqueeze(1)
    key_pos_full = torch.arange(horizon_end, device=device).unsqueeze(0)
    causal_mask = key_pos_full > abs_q_pos
    k_g = k_full[head_group, :horizon_end, :].float()
    acc = torch.zeros(horizon_end - horizon_start, cand.numel(), device=device)
    for h_local in range(kv_repeat):
        h = head_group * kv_repeat + h_local
        q_h = q_full[h, horizon_start:horizon_end, :].float()
        scores = torch.matmul(q_h, k_g.transpose(0, 1)) * scaling
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        acc += attn.index_select(1, cand)
    dist = (acc / kv_repeat).mean(dim=0)
    return dist / dist.sum().clamp_min(1e-12)


@torch.no_grad()
def recent_qk_logits(q_full, k_full, cand, cur_t, window, head_group, kv_repeat, scaling):
    device = q_full.device
    cand = cand.to(device)
    start = max(0, cur_t - window + 1)
    k_g = k_full[head_group, cand, :].float()
    acc = torch.zeros(cur_t - start + 1, cand.numel(), device=device)
    for h_local in range(kv_repeat):
        h = head_group * kv_repeat + h_local
        q_h = q_full[h, start:cur_t + 1, :].float()
        acc += torch.matmul(q_h, k_g.transpose(0, 1)) * scaling
    return acc / kv_repeat


@torch.no_grad()
def recent_laprox_score(q_full, k_full, v_full, cand, cur_t, window, head_group, kv_repeat, scaling, o_proj_block):
    logits = recent_qk_logits(q_full, k_full, cand, cur_t, window, head_group, kv_repeat, scaling)
    attn = torch.softmax(logits, dim=-1)
    attn_norm = attn.norm(p=2, dim=0)
    v_g = v_full[head_group, cand.to(q_full.device), :].float()
    projected = v_g @ o_proj_block
    vwo = projected.norm(p=2, dim=-1)
    return attn_norm * vwo


@torch.no_grad()
def rank_full_pop(x: torch.Tensor):
    n = x.shape[0]
    if n <= 1:
        return torch.zeros_like(x)
    order = x.argsort()
    rank = torch.empty_like(x)
    rank[order] = torch.arange(n, device=x.device, dtype=x.dtype)
    return rank / (n - 1)


def log(record: dict) -> None:
    record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    import os
    gpu = sys.argv[1] if len(sys.argv) > 1 else "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import transformers

    device = torch.device("cuda:0")
    model_path = MODELS["llama31_8b_instruct"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    kv_repeat = num_heads // num_kv_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // num_heads)
    scaling = head_dim ** -0.5

    o_proj_blocks_cache: dict[int, list[torch.Tensor]] = {}

    def get_o_proj_block(layer_idx: int, kv_h: int) -> torch.Tensor:
        if layer_idx not in o_proj_blocks_cache:
            layer = model.model.layers[layer_idx]
            weight = layer.self_attn.o_proj.weight.detach().float()
            blocks = [weight[:, h * head_dim:(h + 1) * head_dim].t().contiguous() for h in range(num_heads)]
            o_proj_blocks_cache[layer_idx] = blocks
        blocks = o_proj_blocks_cache[layer_idx]
        group = blocks[kv_h * kv_repeat:(kv_h + 1) * kv_repeat]
        return torch.stack(group, dim=0).mean(dim=0)

    # --- load ALL docs from both corpora ---
    raw_docs = []  # (context, input, prompt_tmpl, max_out_len)
    with NQ_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                raw_docs.append((r["context"], r["input"], NQ_PROMPT_TMPL, NQ_MAX_OUT_LEN, "nq"))
    with ARXIV_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                raw_docs.append((r["context"], r["input"], ARXIV_PROMPT_TMPL, ARXIV_MAX_OUT_LEN, "arxiv"))
    log({"event": "docs_loaded", "n": len(raw_docs)})

    samples = []  # (ids: list[int], boundary: int, source: str)
    for doc_idx, (context, inp, tmpl, max_out_len, source) in enumerate(raw_docs):
        prompt = tmpl.format(context=context, input=inp)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
        input_ids = enc["input_ids"]
        input_budget = max(1, MAX_SEQ_LEN - max_out_len)
        if int(input_ids.shape[1]) > input_budget:
            half = input_budget // 2
            tail = input_budget - half
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -tail:]], dim=1)
        input_ids = input_ids.to(device)
        prompt_len = int(input_ids.shape[1])

        prev_attn_impl = model.config._attn_implementation
        try:
            try:
                model.config._attn_implementation = "sdpa"
                ref_ids = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_out_len, min_new_tokens=1, do_sample=False, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                )
            finally:
                model.config._attn_implementation = prev_attn_impl
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log({"event": "doc_skip_oom", "doc_idx": doc_idx, "source": source})
            continue

        ids_list = ref_ids[0].tolist()
        samples.append((ids_list, prompt_len, source))
        if (doc_idx + 1) % 20 == 0:
            log({"event": "gen_progress", "n_done": doc_idx + 1, "n_total": len(raw_docs)})

    random.Random(0).shuffle(samples)
    val_samples = samples[:N_VAL_DOCS]
    train_samples = samples[N_VAL_DOCS:]
    n_val_nq = sum(1 for _, _, s in val_samples if s == "nq")
    n_val_arxiv = sum(1 for _, _, s in val_samples if s == "arxiv")
    log({"event": "loaded", "n_train_docs": len(train_samples), "n_val_docs": len(val_samples),
         "n_val_nq": n_val_nq, "n_val_arxiv": n_val_arxiv})

    def compute_doc_layer(boundary, layer_idx, q_full, k_full, v_full):
        cur_t = boundary - 1
        full_len = q_full.shape[1]
        horizon_end = min(cur_t + 1 + ORACLE_HORIZON, full_len)
        if horizon_end - (cur_t + 1) < MIN_HORIZON:
            return None
        protected = set(range(min(SINK_TOKENS, cur_t + 1)))
        recent_start = max(0, cur_t - OBS_WINDOW + 1)
        protected.update(range(recent_start, cur_t + 1))
        all_pos = torch.arange(cur_t + 1, device=device)
        cand_mask = torch.ones(cur_t + 1, dtype=torch.bool, device=device)
        for p in protected:
            cand_mask[p] = False
        cand_idx = all_pos[cand_mask]
        n_cand = int(cand_idx.numel())
        if n_cand < 32:
            return None
        b_remaining = max(1, min(BUDGET_TOKENS - len(protected), n_cand))

        per_head_F, per_head_R, per_head_feat = [], [], []
        for kv_h in range(num_kv_heads):
            F_h = oracle_head_distribution(q_full, k_full, cand_idx, cur_t + 1, horizon_end, kv_h, kv_repeat, scaling)
            o_block = get_o_proj_block(layer_idx, kv_h)
            R_h = recent_laprox_score(q_full, k_full, v_full, cand_idx, cur_t, OBS_WINDOW, kv_h, kv_repeat, scaling, o_block)
            logits_window = recent_qk_logits(q_full, k_full, cand_idx, cur_t, OBS_WINDOW, kv_h, kv_repeat, scaling)
            current_qk = logits_window[-1]
            mean_qk = logits_window.mean(dim=0)
            max_qk = logits_window.max(dim=0).values
            var_qk = logits_window.var(dim=0, unbiased=False)
            w = logits_window.shape[0]
            xw = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0
            xw_ss = (xw * xw).sum().clamp_min(1e-6)
            slope_qk = (logits_window * xw.unsqueeze(1)).sum(dim=0) / xw_ss
            k_norm = k_full[kv_h, cand_idx, :].float().norm(p=2, dim=-1)
            v_norm = v_full[kv_h, cand_idx, :].float().norm(p=2, dim=-1)
            vwo_norm = (v_full[kv_h, cand_idx, :].float() @ o_block).norm(p=2, dim=-1)
            age = (cur_t - cand_idx.float())
            per_head_F.append(F_h)
            per_head_R.append(R_h)
            per_head_feat.append({
                "current_qk": current_qk, "mean_qk": mean_qk, "max_qk": max_qk, "var_qk": var_qk,
                "slope_qk": slope_qk, "k_norm": k_norm, "v_norm": v_norm, "vwo_norm": vwo_norm, "age": age,
            })
        F_all = torch.stack(per_head_F, dim=0).mean(dim=0)
        R_all = torch.stack(per_head_R, dim=0).mean(dim=0)
        topB_R = set(R_all.topk(b_remaining).indices.tolist())
        topB_F = set(F_all.topk(b_remaining).indices.tolist())
        rescue = topB_F - topB_R
        if not rescue:
            return None
        feat_mean = {k: torch.stack([d[k] for d in per_head_feat], dim=0).mean(dim=0) for k in per_head_feat[0]}
        cols = [feat_mean[k] for k in RAW_ORDER]
        for k in RAW_ORDER:
            cols.append(rank_full_pop(feat_mean[k]))
        X = torch.stack(cols, dim=-1)
        rescue_mask = torch.zeros(n_cand, dtype=torch.bool, device=device)
        rescue_mask[list(rescue)] = True
        return X, rescue_mask

    log({"event": "estimating_norm_stats"})
    norm_samples = []
    for ids_list, boundary, source in train_samples[:12]:
        full_ids = torch.tensor(ids_list, dtype=torch.long, device=device).unsqueeze(0)
        try:
            cap = capture_sequence_chunked(model, full_ids, "llama", chunk_size=4096)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        for layer_idx in [0, 8, 16, 24, 31]:
            if layer_idx not in cap.qkv:
                continue
            q_full, k_full, v_full = cap.qkv[layer_idx]
            q_full, k_full, v_full = q_full.to(device), k_full.to(device), v_full.to(device)
            out = compute_doc_layer(boundary, layer_idx, q_full, k_full, v_full)
            if out is None:
                continue
            X, _ = out
            norm_samples.append(X)
        del cap
        torch.cuda.empty_cache()
    all_X = torch.cat(norm_samples, dim=0)
    feat_mean = all_X.mean(dim=0)
    feat_std = all_X.std(dim=0).clamp_min(1e-3)
    log({"event": "norm_stats_ready", "n_rows": int(all_X.shape[0])})

    mlp = RescueMLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)

    def run_doc(ids_list, boundary, train: bool):
        full_ids = torch.tensor(ids_list, dtype=torch.long, device=device).unsqueeze(0)
        try:
            cap = capture_sequence_chunked(model, full_ids, "llama", chunk_size=4096)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return []
        losses = []
        for layer_idx in range(num_layers):
            if layer_idx not in cap.qkv:
                continue
            q_full, k_full, v_full = cap.qkv[layer_idx]
            q_full, k_full, v_full = q_full.to(device), k_full.to(device), v_full.to(device)
            out = compute_doc_layer(boundary, layer_idx, q_full, k_full, v_full)
            if out is None:
                continue
            X, rescue_mask = out
            Xn = (X - feat_mean) / feat_std
            if train:
                logits = mlp(Xn)
            else:
                with torch.no_grad():
                    logits = mlp(Xn)
            log_probs = F.log_softmax(logits, dim=-1)
            n_rescue = int(rescue_mask.sum().item())
            target = rescue_mask.float() / n_rescue
            loss = -(target * log_probs).sum()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
        del cap
        torch.cuda.empty_cache()
        return losses

    def save_ckpt(tag: str):
        payload = {
            "kind": "rescue_scorer_v1",
            "raw_features": RAW_ORDER,
            "all_features": ALL_FEATURES,
            "recent_window": 32,
            "scaler_mean": feat_mean.detach().cpu(),
            "scaler_scale": feat_std.detach().cpu(),
            "mlp_weights": [mlp.net[0].weight.detach().cpu(), mlp.net[2].weight.detach().cpu(), mlp.net[4].weight.detach().cpu()],
            "mlp_biases": [mlp.net[0].bias.detach().cpu(), mlp.net[2].bias.detach().cpu(), mlp.net[4].bias.detach().cpu()],
            "score_mode": "softmax",
            "softmax_temperature": 1.0,
        }
        ckpt_path = SCRATCH / f"rescue_scorer_listwise_full_{tag}.pt"
        torch.save(payload, ckpt_path)
        torch.save(payload, OUT_CKPT)
        return ckpt_path

    for epoch in range(EPOCHS):
        order = list(range(len(train_samples)))
        random.Random(1000 + epoch).shuffle(order)
        all_losses = []
        for i, doc_i in enumerate(order):
            ids_list, boundary, source = train_samples[doc_i]
            all_losses.extend(run_doc(ids_list, boundary, train=True))
            if (i + 1) % 100 == 0:
                log({"event": "train_progress", "epoch": epoch, "n_docs_done": i + 1, "n_docs_total": len(order)})
        train_loss = sum(all_losses) / max(1, len(all_losses))

        val_losses = []
        for ids_list, boundary, source in val_samples:
            val_losses.extend(run_doc(ids_list, boundary, train=False))
        val_loss = sum(val_losses) / max(1, len(val_losses))

        log({"event": "epoch_done", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "n_train_steps": len(all_losses), "n_val_steps": len(val_losses)})

        if epoch % CKPT_EVERY == 0 or epoch == EPOCHS - 1:
            ckpt_path = save_ckpt(f"ep{epoch}")
            log({"event": "checkpoint_saved", "epoch": epoch, "path": str(ckpt_path)})

    log({"event": "training_done"})


if __name__ == "__main__":
    main()
