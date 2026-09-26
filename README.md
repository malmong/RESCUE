# RESCUE

Reference implementation for **RESCUE**: a policy-conditioned residual scorer
that recovers what a KV cache eviction policy discards, and a fidelity-guided
selector that decides, per input, whether to apply the correction at all.

An eviction policy scores cache entries from evidence already visible at
prefill. That evidence is not the whole story: at a matched budget, **24.6%**
of the oracle-critical entries are recovered by a future-importance signal alone
and not by the observed one. RESCUE trains a
small scorer on exactly that residual — the entries *this* base policy misses —
and combines it with the base score so that a declined correction reproduces the
base policy exactly.

---

## What is here

```
rescue/       the method: eviction and the selector, the 18 features, the
              scorer, the base policies, the model registry
configs/      one YAML per backbone: context limit, budgets, selector settings
scripts/      everything you run: generate, evaluate, train, export, measure
analysis/     table_<n>.py and figure_<n>.py, numbered as the paper numbers
              them, over one shared loader (data.py)
results/      trained scorers, every score the paper reports, and prediction
              samples for checking the metrics
tests/        a bit-exact regression check for edits to the eviction path
```

## Reproducing the paper without a GPU

Every table and figure reads `results/scores.csv` and `results/per_document.csv`,
which are committed. Nothing below needs model weights, benchmark data, or a
GPU:

```bash
pip install -r requirements.txt

# tables that have a script, numbered as the paper numbers them
python analysis/table_1.py  --format text     # LongBench, all three backbones
python analysis/table_2.py  --format text     # correction vs. selector ablation
python analysis/table_5.py                    # budget sweep, per task
python analysis/table_7.py  --format text     # accuracy against cost
python analysis/table_18.py --format text     # cache-selection control
for n in 1 2 4 5 7 9 10 12 13 14 15 18 19 20 21 22 23 25; do
    python analysis/table_$n.py --out analysis/out/table_$n.tex
done

python analysis/figure_1.py    # where the oracle's top-B entries come from
python analysis/figure_3.py    # base -> oracle span, with RESCUE on it
python analysis/figure_4.py    # per-cell and per-document verification
```

Figure 2 is the architecture diagram and has no script. One further table,
`analysis/appendix_lambda_candidates.py`, renders a comparison the paper
reports as prose rather than as a numbered table.

Eighteen of the paper's twenty-five tables have a script here. The seven that
do not --- the per-task LongBench grid (3), the RULER tables (6, 11), throughput
(8), the two abstention-floor tables (16, 17) and the seed/target table (24) ---
are read from `results/scores.csv` and `scripts/measurements/` directly;
`tests/test_numbering.py` lists them so the gap is visible rather than implied.

Table numbers shift whenever a table is added to the body, so
`python tests/test_numbering.py --tex paper.tex` checks that every
`analysis/table_<n>.py` still generates the paper's table `<n>`.


`--format latex` emits the table as it appears in the paper; `--out PATH`
writes to a file instead of stdout. Figures land in `analysis/out/`.

`results/scores.csv` is a set of numbers, and numbers in a file can be
anything. `results/samples/` carries the text underneath a few of them -- the
model's actual generations, their references, and the score each document
received, one sample per LongBench metric family. `python
scripts/verify_samples.py` recomputes those scores with the official metric and
reports whether they come back. That step needs OpenCompass; everything above
it does not.

## Running the method

### 1. Assets

Model weights and LongBench are downloaded, not vendored:

```bash
bash scripts/download_assets.sh        # ~50 GB of weights, ~1 GB of data
export RESCUE_MODEL_ROOT=$PWD/assets/models
export RESCUE_DATA_ROOT=$PWD/assets/data
```

Llama-3.1 and Mistral are gated on the Hub; run `huggingface-cli login` first.

Evaluation uses [OpenCompass](https://github.com/open-compass/opencompass) for
the official LongBench metrics. Point `OPENCOMPASS_ROOT` at a checkout:

```bash
git clone https://github.com/open-compass/opencompass third_party/opencompass
export OPENCOMPASS_ROOT=$PWD/third_party/opencompass
```

Nothing inside that checkout is patched — the generated config imports this
repo's adapter directly.

### 2. One prompt

```bash
python scripts/generate.py --model llama3_8b --method rescue --base snapkv \
    --checkpoint results/checkpoints/rescue_llama3_8b_snapkv.pt \
    --prompt-file document.txt --question "Who signed the treaty?" --verbose
```

`--verbose` prints what the selector decided for the document: the KL of each
candidate cache against the dense one, and which candidate won.

### 3. A benchmark cell

```bash
python scripts/evaluate.py --model llama3_8b --method snapkv --task qasper
python scripts/evaluate.py --model llama3_8b --method rescue --base snapkv \
    --checkpoint results/checkpoints/rescue_llama3_8b_snapkv.pt --task qasper
```

`--task all` runs all sixteen. `scripts/run_longbench.sh` runs the full grid
behind Table 1.

### 4. Training a scorer

The scorers the paper reports are in `results/checkpoints/` (26 files, 316 KB),
so this is only needed to reproduce training itself:

```bash
bash scripts/build_corpora.sh                       # NQ + arXiv, from the Hub
python scripts/train.py --model llama3_8b --base snapkv
```

A scorer is specific to a (model, base policy, budget) triple, because its
target is that policy's own misses at that budget.

### 5. The training-free probe control

Appendix N.1 reports what the probe token's attention row alone buys, with no
scorer and no second cache. Two environment variables turn it on: the number of
probe rows fed into the base policy's vote, and the weight they carry against a
32-row observation window.

```bash
RESCUE_PROBE_SNAPKV=1 RESCUE_PROBE_WEIGHT=1 \
python scripts/evaluate.py --model llama3_8b --method snapkv --task all --gpu 0
```

Unit weight is what the paper reports (+1.95 on the 16-task mean, against
RESCUE's +1.71). Weights 8 and 32 reach +3.14 and +3.54, but they are chosen on
the evaluation set itself; all three are in `results/scores.csv` under the arms
`probe_attn_w1`, `probe_attn_w8` and `probe_attn_w32`, and
`python analysis/table_18.py` prints them beside the cache-selection control.

---

## The method in one page

**Residual target.** For a document, let `K_GT` be the entries with the highest
future attention mass and `K_base` the entries the base policy keeps. The
scorer is trained on `K_GT \ K_base` — what this policy missed — rather than on
`K_GT`. Training on the full set spends capacity re-deriving selections the base
already makes correctly.

**Features.** 18 per entry: nine raw statistics and their nine within-layer
ranks. The raw nine are `current_qk, mean_qk, max_qk, var_qk, slope_qk, k_norm,
v_norm, vwo_norm, age` (`rescue/features.py`). None reference the base
policy, so the expensive part of feature caching is shared across policies.

**Combination.** `S = s_recent + λ · (Σs_recent / Σs_future) · s_future`. The
single global scalar rescales the correction onto the base score's own range
without touching `s_recent`, so `λ=0` is bit-identical to the base policy. That
identity is what makes a declined correction safe.

**Selector.** After eviction, decode one token against the unpruned cache, replay
it against each candidate cache, and keep the λ whose next-token distribution
stays closest to the dense one:

```
λ* = argmin_λ KL(p_dense ‖ p_pruned^(λ)),   λ ∈ {0, 1}
```

Applied unconditionally the correction gains `+0.64` over 80 cells with an
interval containing zero, and 18 cells lose at least a point. Gated, it gains
`+1.28` and three cells lose a point.

**Selector abstention.** The KL is a sum over the vocabulary of
`p·(log p − log q)`. In float32 that reduction carries an absolute error around
`1e-7`, and the true KL falls under it on a large share of documents for some
models — 61.5% of PassageRetrieval-en and 42.7% of MultiFieldQA-en on Qwen3-8B,
against 0.0% of PassageRetrieval-en on Llama, where 0.25% of all probes fall
below the floor. Below that floor `argmin` is reading rounding noise, and the
correction gets accepted about half the time for no reason. When every
candidate's KL is under `rescue.kl_floor` the selector abstains and keeps the
base cache: no evidence is read as no correction rather than as a coin flip.
This is on by default, and it is a no-op wherever the KL is informative.

---

## Costs

The selector is a test-time verification budget, and it is not free. At an
average 5K-token prompt, eviction- and selection-related latency is 5.6 ms for
SnapKV and 134.7 ms for ForesightKV against **277.2 ms** for the complete
one-token pipeline — about twice the most expensive predicted-future method
compared against, not less than it. Almost none of that is the scorer: the dense
probe costs 21 ms and building and replaying the two candidate caches costs
254 ms. This is accuracy bought with latency, not latency saved. Peak memory is
23.0 GB against the base policy's 17.1 GB, because the dense cache and both
candidates are live at once, and the selector therefore assumes an evict-once
setting rather than streaming or chunked prefill.

## Notes

- **Context limits are per model.** Llama-3.1 is 131072, Mistral-v0.3 is 32768,
  Qwen3-8B is 40960, and each config states its own. Feeding a model prompts
  past its trained position range leaves its eviction scores computed from
  positions it has never seen. Correcting that on Mistral moved NarrativeQA and
  nothing else: `+3.48` on SnapKV, `+2.26` averaged over the five base policies.
- **Five LongBench tasks never get a chat template** — `trec`, `triviaqa`,
  `samsum`, `lcc`, `repobench` — because their prompts are few-shot
  demonstrations or raw code, and the official harness does not wrap them. This
  is enforced in the registry rather than left to the caller.
- **LookaheadKV** patches the model class and runs its own decoder, so it is
  reachable from `scripts/evaluate.py` but not from `scripts/generate.py`. It needs the authors'
  own checkout; set `RESCUE_LOOKAHEADKV_ROOT`.

## Third-party components

Nothing third-party is vendored here; each is installed or cloned by the user,
so this repository carries no code under another project's licence.

| | role | licence |
|---|---|---|
| [OpenCompass](https://github.com/open-compass/opencompass) | the evaluation harness and its implementations of the official LongBench metrics | Apache-2.0 |
| [LongBench](https://github.com/THUDM/LongBench) | the benchmark | MIT |
| Llama-3.1, Mistral-v0.3, Qwen3 | backbones, downloaded from the Hub under their own model licences | see each model card |
| [ForesightKV], [LookaheadKV] | comparators, imported only when their own method is selected | see each repository |

The base policies -- SnapKV, LaProx, H2O, LAVa, R-KV -- are reimplementations
written against each method's released code and defaults rather than copies of
it. Where a method's code and its paper disagree, the code's behaviour is
followed and the divergence is recorded at the call site.
