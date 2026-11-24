#!/usr/bin/env python
"""Where the RESCUE prefill memory actually goes (prefill memory).

The objection: "the per-layer, per-head, per-token KV cache features
take O(L) memory", more than the 128-token budget itself. The budget part is
right by three orders of magnitude -- but the features are not the term that
makes it so, and they are not per-head: rescue/scorer.py reduces every raw
feature over KV heads (`.mean(dim=0)`) before the 18 columns are assembled.

This prices every O(L) term against results/measurements/memscale.csv, which
measured the same configuration on RULER niah_single_2.
"""
import csv
from pathlib import Path

GiB = 1024 ** 3
MiB = 1024 ** 2
# Llama-3.1-8B-Instruct
L, H_KV, D, HID = 32, 8, 128, 4096
BF16, FP32 = 2, 4
B = 128  # retained tokens per layer

M = Path("/path/to/kv_cache/rescue-release/results/measurements/memscale.csv")
meas = {(r["arm"], int(r["length_k"])): r
        for r in csv.DictReader(l for l in open(M) if not l.startswith("#"))}

print(f"{'N':>8s} {'retainedKV':>11s} {'hidden':>10s} {'full K/V':>10s} "
      f"{'feat/layer':>11s} {'predicted':>10s} {'measured':>10s}")
for nk in (16, 32, 64):
    N = nk * 1024
    retained = L * H_KV * B * D * 2 * BF16              # the 128-token budget
    hidden = (L + 1) * N * HID * BF16                   # 33 prompt hidden states
    fullkv = L * H_KV * N * D * 2 * BF16                # prompt K/V held for scoring
    feat_head = 9 * H_KV * N * FP32                     # raw features, before the mean
    feat_cols = 18 * N * FP32                           # the 18 columns the MLP reads
    pred = (hidden + fullkv) / GiB
    base = float(meas[("snapkv", nk)]["peak_gib"])
    resc = float(meas[("snapkv+rescue", nk)]["peak_gib"])
    print(f"{N:>8d} {retained/MiB:>8.1f} MiB {hidden/GiB:>7.2f} GiB "
          f"{fullkv/GiB:>6.2f} GiB {feat_cols/MiB:>7.2f} MiB "
          f"{pred:>7.2f} GiB {resc-base:>6.2f} GiB")

print()
N = 64 * 1024
print(f"At N=64K, per layer:")
print(f"  retained KV (the B=128 budget) : {H_KV*B*D*2*BF16/MiB:7.3f} MiB   (kept)")
print(f"  18 feature columns             : {18*N*FP32/MiB:7.3f} MiB   (transient)")
print(f"  9 raw features before the mean : {9*H_KV*N*FP32/MiB:7.3f} MiB   (transient)")
print(f"  prompt hidden state            : {N*HID*BF16/MiB:7.3f} MiB   (held for the whole pass)")
print(f"\nFeature columns as a share of the measured RESCUE excess at 64K: "
      f"{100*L*18*N*FP32/((68.27-43.93)*GiB):.2f}%")
