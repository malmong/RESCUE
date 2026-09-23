# Results

Everything the tables and figures read. All of it is committed, so the paper's
numbers can be checked without a GPU, model weights, or benchmark data.

| file | what it is |
|---|---|
| `scores.csv` | every LongBench cell: `model, arm, base, budget, task, score`. Built by `tools/export_results.py` from evaluation output. |
| `per_document.csv` | each evaluated document scored under three arms, plus the KL margin the selector measured. Built by `tools/export_per_document.py`. |
| `checkpoints/` | the trained residual scorers, one per (model, base policy) and per budget for the sweep. 26 files, 316 KB. |
| `measurements/` | quantities that are measured rather than derived from scores: signal coverage by budget, the latency breakdown. |
| `summaries/` | per-run scores written by `evaluation/eval_longbench.py` as new runs finish. |

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

## Rebuilding

```bash
python tools/export_results.py      --runs /path/to/opencompass_outputs
python tools/export_per_document.py --runs /path/to/opencompass_outputs
```

The first is seconds. The second re-scores every document with the official
LongBench metrics and takes a few minutes.
