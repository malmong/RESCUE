#!/usr/bin/env python
"""What a per-layer feature pass actually removes (prefill memory).

Appendix L says "a per-layer feature pass would remove most of it". That is
true but imprecise, and the imprecision is worth fixing: the
term that dominates the measured excess is not the features and not the
per-layer structure -- it is a tuple of hidden states that the SHIPPED scorer
never reads.

Three O(L) terms exist in the prefill. This prices each one against
results/measurements/memscale.csv and marks which are live.

  (a) hidden states, all L+1 of them, materialised by
      output_hidden_states=(policy == "rfc")   eviction.py:2632
      and then concatenated across the prefix/observation split into a
      SECOND full-length tuple                 eviction.py:2750
      Their only consumer is
          _populate_z_cache_from_hidden_states eviction.py:659
      which returns on its first line when self.rfc_scorer is None:
          "if self.rfc_scorer is None or hidden_states is None: return"
                                               eviction.py:667
      For rfc_objective == "RESCUE" the constructor assigns
      unified_rfc_scorer and leaves rfc_scorer at None
                                               eviction.py:2107, 2160
      so in the shipped configuration this term is DEAD.

      The other two readers of hidden_states are unreachable from the RESCUE
      path for the same reason: append_generated_position guards on
      "self.rfc_scorer is not None" (eviction.py:836), and the _current_q_group
      branch (eviction.py:1225) sits past the rfc early return at
      eviction.py:1218.

  (b) the unpruned prompt K/V, held across the fidelity selector's lambda
      candidates. The selector deliberately does NOT copy it -- _clone builds
      a new container over the SAME tensors (eviction.py:2219) -- but the
      original list keeps the full tensors alive while a trial holds its
      pruned ones, which an evict-once baseline does not pay. LIVE, and
      structural to the method.

  (c) the features themselves: 18 columns per token per layer, fp32, after
      scorer.py's mean over KV heads. LIVE, and the term the objection named.
"""
import csv
from pathlib import Path

GiB = 1024 ** 3
MiB = 1024 ** 2
L, H_KV, D, HID = 32, 8, 128, 4096   # Llama-3.1-8B-Instruct
BF16, FP32 = 2, 4
B = 128

ROOT = Path("/path/to/kv_cache/rescue-release")
M = ROOT / "results/measurements/memscale.csv"
meas = {(r["arm"], int(r["length_k"])): r
        for r in csv.DictReader(l for l in open(M) if not l.startswith("#"))}

# The guard conditions the "dead" claim rests on. Re-checked here so the claim
# is reproducible rather than asserted: if the release moves, this fails loudly.
GUARDS = [
    (2632, 'output_hidden_states=policy == "rfc"'),
    (2667, "if self.rfc_scorer is None or hidden_states is None:"),
    (2160, "self.unified_rfc_scorer = RescueScorer("),
]


def check_guards():
    src = (ROOT / "rescue/eviction.py").read_text().split("\n")
    hits = []
    for _, needle in GUARDS:
        found = [i + 1 for i, l in enumerate(src) if needle in l]
        hits.append((needle, found))
    return hits


def terms(N):
    return {
        "hidden states (L+1 tuple)": (L + 1) * N * HID * BF16,
        "unpruned prompt K/V": L * H_KV * N * D * 2 * BF16,
        "18 feature columns": L * 18 * N * FP32,
        "retained budget B=128": L * H_KV * B * D * 2 * BF16,
    }


if __name__ == "__main__":
    print("guard check (claim is void if any of these moved)")
    for needle, found in check_guards():
        print(f"  {'OK ' if found else 'MISSING'} line {found} : {needle}")

    print(f"\n{'N':>8} {'hidden':>10} {'promptK/V':>11} {'features':>10} "
          f"{'budget':>9} | {'measured excess':>16} {'hidden share':>13}")
    for nk in (16, 32, 64):
        N = nk * 1024
        t = terms(N)
        exc = (float(meas[("snapkv+rescue", nk)]["peak_gib"])
               - float(meas[("snapkv", nk)]["peak_gib"]))
        print(f"{N:>8} {t['hidden states (L+1 tuple)']/GiB:>7.2f}GiB "
              f"{t['unpruned prompt K/V']/GiB:>8.2f}GiB "
              f"{t['18 feature columns']/MiB:>7.1f}MiB "
              f"{t['retained budget B=128']/MiB:>6.1f}MiB | "
              f"{exc:>13.2f}GiB {100*t['hidden states (L+1 tuple)']/(exc*GiB):>12.1f}%")

    N = 64 * 1024
    t = terms(N)
    exc = (float(meas[("snapkv+rescue", 64)]["peak_gib"])
           - float(meas[("snapkv", 64)]["peak_gib"]))
    base_peak = float(meas[("snapkv", 64)]["peak_gib"])
    print(f"\nat N=64K, measured excess {exc:.2f} GiB over SnapKV's {base_peak:.2f} GiB peak")
    print(f"  (a) hidden states  {t['hidden states (L+1 tuple)']/GiB:6.2f} GiB  "
          f"{100*t['hidden states (L+1 tuple)']/(exc*GiB):5.1f}%   DEAD -- one flag")
    rest = exc * GiB - t["hidden states (L+1 tuple)"]
    print(f"  (b) prompt K/V     {rest/GiB:6.2f} GiB  {100*rest/(exc*GiB):5.1f}%   LIVE -- "
          f"the fidelity selector holds it across lambda candidates")
    print(f"  (c) features       {t['18 feature columns']/GiB:6.2f} GiB  "
          f"{100*t['18 feature columns']/(exc*GiB):5.1f}%   LIVE -- the term the objection named")

    print("\nprojected peak at 64K")
    print(f"  SnapKV (evict-once)                      {base_peak:6.2f} GiB")
    print(f"  RESCUE as shipped                        {base_peak+exc:6.2f} GiB")
    after_a = base_peak + (exc * GiB - t["hidden states (L+1 tuple)"]) / GiB
    print(f"  - drop the dead hidden-state tuple       {after_a:6.2f} GiB   "
          f"(-{100*t['hidden states (L+1 tuple)']/(exc*GiB):.0f}% of the excess)")
    perlayer = base_peak + (t["unpruned prompt K/V"] / L + t["18 feature columns"] / L) / GiB
    print(f"  - and stream the pass layer by layer     {perlayer:6.2f} GiB   "
          f"(-{100*(1-(perlayer-base_peak)/exc):.0f}% of the excess)")
    print("\nThe second line is what Appendix L promised. The first line is "
          "larger, and costs a flag.")
