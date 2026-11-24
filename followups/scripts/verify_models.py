"""Load each backbone through kvbench's own MODELS dict (so the /data1 symlink
is exercised) and run a real forward + greedy decode."""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbench.configs.models import MODELS

PROMPT = "The capital city of France is"
for key, path in MODELS.items():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to("cuda:4").eval()
    ids = tok(PROMPT, return_tensors="pt").to("cuda:4")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.eos_token_id)
        logits = m(**ids).logits
    txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    nan = bool(torch.isnan(logits).any() or torch.isinf(logits).any())
    print(f"{key:26s} ok  {time.time()-t0:5.1f}s  params={sum(p.numel() for p in m.parameters())/1e9:.2f}B  "
          f"vocab={logits.shape[-1]}  nan/inf={nan}")
    print(f"{'':26s}     -> {PROMPT!r} + {txt!r}")
    del m; torch.cuda.empty_cache()
print("\nall three load and decode.")
