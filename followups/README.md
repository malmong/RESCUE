# Follow-up analyses

Experiments and offline analyses run after the main results in `results/scores.csv`
were complete. Each entry states the question, how it was measured, and what came
back. Everything here either reads committed CSVs (no GPU) or is the driver for a
run whose outcome is recorded below.

```
scripts/      the analysis drivers
results/      what they produced
checkpoints/  scorers trained for the extra budgets and for the probe-feature arm
```

Unless stated otherwise: Llama-3.1-8B-Instruct, 16 LongBench tasks, B=128 per
layer, evict-once. Reference points — dense 49.95, SnapKV base 43.03,
H2O base 43.04, RESCUE over SnapKV 44.74, over H2O 45.55.

---

## 1. Deferred eviction — giving the base policy the step the probe costs

**Question.** Is the gain anything more than having seen a little of the future?
Defer the base policy's eviction by the same number of decode steps the probe
costs, and see how much of the correction's gain survives.

**Method.** `scripts/delayed/delayed_eviction.py`. Run the model on the
uncompressed cache for *w* decode steps, then evict with those *w* steps'
attention included in the evidence, then carry on. The eviction machinery already
does the right thing when the *w* tokens are part of the prompt, so a delay of *w*
is a prompt of `prompt + the w tokens the dense model would have emitted`,
evaluated for `max_new_tokens - w` further steps.

**w=8 first, and why it was dropped.** The scored answer finishes inside an
8-step window for **53% of all documents**. There the delayed arm never evicts
anything that reaches the metric, so its generation is byte-identical to dense
and the control stops being a compression method. Counting EOS inside the window
is a bad proxy for this (TREC and LCC rarely emit EOS at all, yet TREC's delayed
score equals dense to the decimal); the direct test compares the post-processed
generations themselves — `scripts/delayed/identical_to_dense.py`.
`scripts/delayed/subset_evicting.py` recomputes every arm on the complement,
scoring with OpenCompass's own evaluator classes and first re-scoring each arm on
the full set to check it reproduces what the run reported.
Results in `results/delayed_scores.csv`, `results/subset_evicting.out`,
`results/subset_h2o.out`.

**w=1 is the reported control.** The probe is one token long, so w=1 is the
parity, and **0 of 3,750 documents per arm** finish inside a one-step window — no
subset needed, full-benchmark averages apply.

Share of the base-to-dense gap closed, `100 x (score - base) / (dense - base)`:

| base policy | +RESCUE | delay w=1 | delay + RESCUE |
|---|---|---|---|
| SnapKV | 24.8 | 31.9 | 44.2 |
| H2O | 36.4 | 25.5 | 47.9 |

16-task means behind those: SnapKV 43.03 / 44.74 / 45.24 / 46.09, H2O 43.04 /
45.55 / 44.80 / 46.35, dense 49.95.

**Reading.** The deferral is a real effect, and on SnapKV it closes more of the
gap than the correction does alone. The two compose: the correction on top of the
deferral closes a further 12.3 points for SnapKV and 22.4 for H2O — about half of
its standalone gain on SnapKV and three fifths on H2O survive the head start. It
is not a cheaper operating point, though: holding the cache uncompressed even for
one step allocates the dense peak.

---

## 2. Abstention-floor sensitivity, 3 models x 5 policies x 7 values of tau

**Question.** Does the KL gate need per-model or per-task calibration?

**Method.** `scripts/tau_sweep_fix.py`, replayed offline from the stored gate logs
— no retraining, no GPU. Every gated run logged its own KL pair per document, so
raising the floor is replayable: a document abstains at tau when
`max(KL(lam=0), KL(lam=1)) < tau`, and an abstaining document is served the base
cache. Lowering the floor below a run's own value is *not* replayable — an
abstained document has no correction output — so the sweep only goes up from 1e-6.
1,680 rows in `results/tau_sweep_fixed.csv`.

Each cell is the mean gain over the base policy across all 80 (base policy, task)
cells for that backbone. `abstain` is the share where the base cache is kept for
any reason; `floor` is the part of it the floor itself accounts for.

| tau | Llama-3.1 | Mistral-7B | Qwen3-8B | abstain | (floor) |
|---|---|---|---|---|---|
| 1e-6 (shipped) | +1.28 | +1.08 | +0.43 | 55.9% | 0.1% |
| 1e-5 | +1.28 | +1.16 | +0.38 | 58.2% | 2.3% |
| 1e-4 | +1.20 | +1.14 | +0.33 | 61.3% | 5.4% |
| 1e-3 | +1.11 | +1.07 | +0.31 | 66.0% | 10.1% |
| 1e-2 | +0.96 | +0.97 | +0.29 | 72.5% | 16.6% |
| 1e-1 | +0.90 | +0.88 | +0.28 | 81.8% | 26.0% |
| 1 | +0.61 | +0.67 | +0.25 | 94.0% | 38.1% |

**Reading.** One shared tau serves all three backbones: the best setting is 1e-5
on Llama and Mistral and 1e-6 on Qwen, and above 1e-5 the gain falls with tau in
every model. No setting makes the gate harmful; it degrades towards the base
policy. Per task the shared value is not always the best one, but the largest gain
a task-specific tau would add is 1.01 points, on the two code tasks, where
abstaining more often helps. Of the 48 (model, task) cells, 17 already prefer the
shipped value and 11 prefer a *smaller* tau (by at most +0.29) — so "tasks only
ever want to abstain more" is the wrong framing.

### A bug in the first version of this replay

`scripts/tau_sweep.py` (kept here unmodified, for the record) accumulates
`tot += y if live else x`, where `x` is a **separate standalone base run's**
per-document score. For a document the selector itself rejected, the selector
run's own dumped output *is* the base cache's output, so substituting another
job's generation is wrong wherever the two jobs diverge. A handful of documents
per task differ, and the replay drifts.

On Qwen3/H2O this moved the 16-task mean by **+0.1476** (= +0.0295 over 80 cells),
which is exactly the 0.4250 - 0.3952 gap that made the shipped row disagree with
`results/scores.csv`. Llama/H2O (-0.0177) and Mistral/H2O (+0.0027) nearly
cancelled; Qwen/SnapKV was 0.0000. Note how few documents carry it — on
Qwen3/H2O narrativeqa, 5 of 131 rejected documents:

```
model/base      task                  rej  rej&x!=y   sweep    true   delta
qwen3_8b/h2o    narrativeqa           131         5   18.36   17.83   -0.53
qwen3_8b/h2o    passage_retrieval_en  131         5   77.33   79.83   +2.50
qwen3_8b/h2o    2wikimqa              133         2   38.13   37.36   -0.77
qwen3_8b/h2o    musique               127         2   27.53   28.07   +0.54
...
```

The fix in `tau_sweep_fix.py`: substitute the base run's output **only** for
documents the replay *newly* floors (`acc` true but `max|KL| < tau`); documents
the run itself rejected keep the run's own score. The corrected sweep reproduces
`results/scores.csv` on all 80 Qwen cells exactly.

---

## 3. The compression curve at five budgets for all five policies

**Question.** What does the gain look like as a function of how much cache is
actually kept?

**Method.** `scripts/collect_curve.py` plus a scorer trained per (model, base,
budget) — the target is that policy's own misses at that budget, so each budget
needs its own. B=192 and B=512 for all five policies, B=768 for SnapKV;
checkpoints in `checkpoints/`, scores in `results/curve_scores.csv`.
B=128/256/1024 come from `results/scores.csv`.

`scripts/compression_ratio.py` converts the budget axis to a compression ratio
without assuming anything: every evaluated prompt is stored verbatim in the
OpenCompass predictions, so `retained/original = B / N_prompt` is measurable per
document with the tokenizer the runs used.

| Llama-3.1, B per layer | 128 | 192 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| compression | 69x | 46x | 35x | 17x | 8.6x |
| prompt retained (median doc) | 1.45% | 2.17% | 2.89% | 5.78% | 11.57% |
| H2O | +2.51 | +2.22 | +2.19 | +2.21 | +1.48 |
| SnapKV | +1.71 | +1.40 | +0.98 | +0.12 | -0.02 |
| LAVa | +1.09 | +0.92 | +0.73 | +0.10 | -0.05 |
| LaProx | +0.81 | +0.40 | +0.32 | +0.13 | +0.06 |
| R-KV | +0.27 | +0.28 | +0.18 | -0.17 | -0.10 |

Gap to dense at 17x: H2O 3.57, SnapKV 2.11, LaProx 1.88, LAVa 1.68, R-KV 1.62.

**Reading.** The correction only recovers what the base policy missed, so the less
a policy misses the less there is to recover. A wide budget removes the need from
both sides — at 8.6x the median document still keeps 11.57% of its prompt, and
eviction has little point there either.

---

## 4. Scorer-quality ladder

**Question.** With a perfect future predictor the observed signal would be
redundant; the combination helps only because the predictor is an approximation.
So how good is this one, against alternatives?

**Method.** `scripts/scorer_ladder.py`, reads `results/scores.csv` only. Every
rung is compared on SnapKV alone so the ladder is a ladder and not a change of
base. The two training targets: *full-future* predicts the whole top-B under true
future attention, *residual* only what this base got wrong. *Gated* means applied
only where the probe accepts it.

| rung | gain over SnapKV | abs |
|---|---|---|
| true future attention (oracle) | +6.04 | 49.07 |
| residual target, gated (shipped) | +1.71 | 44.74 |
| full-future target, gated | +1.40 | 44.43 |
| residual target, always on | +1.11 | 44.14 |
| full-future target, always on | +0.86 | 43.89 |

Over all five base policies: shipped +1.28, full-future gated +1.06, residual
always-on +0.64, full-future always-on +0.47.

**Reading.** The narrower target wins at both settings — predicting less is what
makes it learnable. Two published future-attention predictors land at or below the
policy they replace: LookaheadKV 43.38, ForesightKV 40.43, against a five-policy
base of 43.77. Of the entries the base drops, only 35% are visible to a future
signal at all, and the shipped scorer recovers 25.7% of them.

---

## 5. Stacking on a head-budget allocator (LU-KV)

**Question.** LU-KV solves head-level budget *allocation* as a global
combinatorial problem and leaves an existing rule to fill each head. This method
changes that second decision and leaves the first alone — do they compose?

**Method.** `scripts/lukv/lukv_eviction.py` — one patch at import,
`rescue.eviction.RaggedLaVaState` -> `LUKVState`, so `--base lava` runs LU-KV and
`--method rescue --base lava` runs LU-KV + the correction. Their released profile,
their allocation and their within-head rule, unchanged. Every run name is tagged
`lukv` so no output can be mistaken for a real LAVa cell.

| | LU-KV | + correction | |
|---|---|---|---|
| B=128 | 44.62 | 45.56 | +0.94 |
| B=256 | 47.41 | 47.99 | +0.58 |

**Where their allocation has room to work** (`scripts/lukv/floor_analysis.py`):
LU-KV ships its profile as a (99, 32, 8) array of per-(layer, head) prune ratios
indexed by global compression ratio, and its allocator forces every head to keep
at least `min_keep_limit = max(round(0.01 * N), sink + window)` tokens. Below
about 1% of the document per head there is nothing left to reallocate and the
curve collapses to uniform. B=128 is roughly 1.45% of the median prompt — just
above that boundary. The script measures, per evaluated document, which row of
their table the budget lands on and how much head differentiation survives there.

---

## 6. Does the scorer transfer across base policies?

**Method.** Swap the scorers: run SnapKV with the scorer trained for H2O, and H2O
with the scorer trained for SnapKV. `results/*-xfer_from_*.csv`.

| base | own scorer | other policy's scorer | gain kept |
|---|---|---|---|
| SnapKV | 44.74 (+1.71) | 44.67 (+1.64) | 95.9% |
| H2O | 45.55 (+2.51) | 45.29 (+2.24) | 89.4% |

**Reading.** The scorer is conditioned on its base policy, but it is not
overfitted to it — most of the gain survives the swap.

---

## 7. The probe row as a feature

**Question.** The probe token's attention over the prompt, folded straight into
the base policy's vote, is a training-free control that scores well on the mean.
Why is that row not simply a feature of the scorer?

**Method.** `scripts/probe_feature/`. The inference side already decodes a probe
token off the dense prompt cache and keeps its attention row, so the overlay does
not add a capture — it redirects the one that exists (three patches at import,
`rescue.scorer.RescueScorer` -> `ProbeRescueScorer` and friends). The row becomes
a tenth raw feature, so 20 columns instead of 18. A **matched control** trained
with the same recipe and the original 18 features separates the feature's effect
from the effect of retraining at all.

| arm | 16-task mean | vs SnapKV |
|---|---|---|
| SnapKV base | 43.03 | — |
| matched 18-feature control | 44.70 | +1.67 |
| shipped (18 features) | 44.74 | +1.71 |
| with the probe row as a feature | 46.31 | +3.28 |

**Reading.** The probe row is worth **+1.61** over the matched control. The
control landing on the shipped number (44.70 vs 44.74) is what makes that
attributable to the feature rather than to the retraining.

### The training-free version of the same row

`results/scores.csv` carries it as `probe_attn_w1 / w8 / w32`, where *w* is the
weight the probe row carries against a 32-row observation window — a different
axis from the *w* in section 1, which is a number of decode steps.

| arm | mean | vs SnapKV |
|---|---|---|
| probe_attn_w1 | 44.98 | +1.95 |
| probe_attn_w8 | 46.17 | +3.14 |
| probe_attn_w32 | 46.57 | +3.54 |

Unit weight is the honest number; 8 and 32 are chosen on the evaluation set
itself. That +1.95 is one task: on TREC the row alone gains +20.50 against the
shipped method's +5.50, and over the other fifteen tasks it is +0.71 against
+1.46, behind on nine. TREC is structural — its answers are 50-way class labels of
two words, so the probe token is nearly the scored output there. The control also
carries the deeper tail (worst task -0.86 against -0.22) and improves eleven of
the sixteen tasks against thirteen.

---

## 8. Where the prefill memory actually goes

**Question.** Do the per-token KV features dominate the prefill memory, as their
O(L) growth suggests?

**Method.** `scripts/memory_accounting.py` and
`scripts/perlayer_feature_pass.py` price every O(L) term against
`results/measurements/memscale.csv`, which measured the same configuration on
RULER niah_single_2. The second script guards itself: it checks three source lines
and voids the claim if they have moved.

The features are not per-head — `rescue/scorer.py` reduces every raw feature over
KV heads before the 18 columns are assembled.

At N=64K, per layer: retained KV (the B=128 budget) 0.500 MiB (kept), the 18
feature columns 4.500 MiB (transient), the 9 raw features before the mean 18.000
MiB (transient), the prompt hidden state **512.000 MiB** (held for the whole pass).

Measured excess over SnapKV's 43.93 GiB peak at 64K is 24.34 GiB:

| term | | share | status |
|---|---|---|---|
| hidden states | 16.50 GiB | 67.8% | **dead** — one flag; the scorer never reads them |
| prompt K/V | 7.84 GiB | 32.2% | live — the selector holds it across lambda candidates |
| features | 0.14 GiB | 0.6% | live |

Projected peak at 64K: SnapKV 43.93 GiB, shipped 68.27 GiB, 51.77 GiB with the
dead hidden-state tuple dropped (-68% of the excess), 44.18 GiB with the pass also
streamed layer by layer (-99%).

---

## 9. Long outputs

**Method.** `scripts/output_length.py` reads the stored predictions only. Per
task: the median length of the model's own output and of the gold answer, against
the mean gain over the five base policies.

| task | output | gap to dense | SnapKV | LaProx | LAVa | R-KV | H2O |
|---|---|---|---|---|---|---|---|
| QMSum | 73 w | 2.47 | +0.07 | +0.11 | +0.04 | -0.12 | +0.15 |
| GovReport | 328 w | 16.02 | +1.68 | +1.26 | +0.06 | -0.26 | -2.17 |
| MultiNews | 400 w | 8.02 | +0.75 | +0.57 | +0.29 | -0.31 | -0.95 |

**Reading.** Length is not what bounds the method. Within a task it does not
predict the gain (rho = -0.015 over 3,550 documents); the gap column explains the
rows. On GovReport the correction returns its 16-task average on SnapKV (+1.68
against +1.71) and loses on H2O (-2.17). An oracle reaches 63.5% of the gap here
against 94.8% elsewhere.

---

## Notes for anyone re-running this

- `python` may not be on PATH; use `python3` for the pure-CSV scripts and the
  project's conda environment for anything importing OpenCompass evaluators.
- Bind OpenCompass GPUs with its own `--gpu N`; a `CUDA_VISIBLE_DEVICES` prefix is
  overwritten and stacks every job on GPU 0.
- Do not score `predictions/` directly — the official post-processing is skipped
  that way and the numbers come out wrong. Go through the evaluator.
- `trec`, `triviaqa`, `samsum`, `lcc` and `repobench` never get a chat template.
- "Same config, different score" is a stale checkpoint until proven otherwise;
  check mtime before reading code.
