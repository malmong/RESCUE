# Results

Everything the tables and figures read. All of it is committed, so the paper's
numbers can be checked without a GPU, model weights, or benchmark data.

| file | what it is |
|---|---|
| `scores.csv` | every LongBench cell: `model, arm, base, budget, task, score`. Built by `scripts/export_results.py` from evaluation output. |
| `per_document.csv` | each evaluated document scored under three arms, plus the KL margin the selector measured. Built by `scripts/export_per_document.py`. |
| `checkpoints/` | the trained residual scorers, one per (model, base policy) and per budget for the sweep. 26 files, 316 KB. |
| `measurements/` | quantities that are measured rather than derived from scores: signal coverage by budget, the latency breakdown. |
| `samples/` | raw model predictions for a few cells, one per LongBench metric family, with the references and the score each document received. `scripts/verify_samples.py` recomputes them. |
| `summaries/` | per-run scores written by `scripts/evaluate.py` as new runs finish. |

## `arm` in `scores.csv`

| arm | meaning |
|---|---|
| `dense` | no eviction — the upper bound |
| `base` | the base eviction policy alone |
| `rescue` | base + residual correction + fidelity selector (the method) |
| `rescue_ungated` | the same correction applied to every document |
| `fullfuture` | a policy-agnostic full-future target, applied to every document |
| `fullfuture_gated` | the same target under the selector |
| `oracle_selector` | per document, whichever cache scored better in hindsight |
| `lookaheadkv`, `foresightkv` | future-aware comparators, which replace the base policy rather than correct it |
| `cache_select` | the selector choosing between SnapKV's and ForesightKV's caches |
| `probe2`, `probe4`, `probe8` | the selector with a longer probe |

A cell is keyed by `(model, arm, base, budget, task)`. `base` is empty for arms
that do not correct a base policy, and `budget` is 0 for `dense`.

## Checking the metrics

`scores.csv` is a set of numbers, and numbers in a file can be anything.
`samples/` is the layer underneath: the text the model actually generated, what
it was scored against, and the score it got. Recomputing those scores with the
official metric is what ties the aggregate to something inspectable:

```bash
python scripts/verify_samples.py
```

```
gov_report             25 documents  LongBenchRougeEvaluator            OK
lcc                    25 documents  LongBenchCodeSimEvaluator          OK
passage_count          25 documents  LongBenchCountEvaluator            OK
passage_retrieval_en   25 documents  LongBenchRetrievalEvaluator        OK
qasper                 25 documents  LongBenchF1Evaluator               OK
trec                   25 documents  LongBenchClassificationEvaluator   OK

150/150 documents reproduce the stored score
```

It needs OpenCompass, since the metrics are its implementations of the official
LongBench ones -- the same code path the evaluation used.

## Rebuilding

```bash
python scripts/export_results.py      --runs /path/to/opencompass_outputs
python scripts/export_per_document.py --runs /path/to/opencompass_outputs
python scripts/export_samples.py      --runs /path/to/opencompass_outputs
```

The first is seconds. The second re-scores every document with the official
LongBench metrics and takes a few minutes.
