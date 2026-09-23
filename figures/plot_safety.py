#!/usr/bin/env python
"""Figure 3: verification at two levels -- per cell, and per document.

Left: each (base, task) cell with the correction applied to every document
against the same correction gated by the selector. Points above the diagonal
are cells the selector improves; the shaded quadrants are losses. The average
and the tail are different claims, and this panel is about the tail.

Right: why an accuracy barely above a coin flip is worth something. Only
accepted corrections move the score -- rejecting returns the base cache and
contributes exactly zero -- and among the accepts, the right decisions sit on
larger differences than the wrong ones.

    python figures/plot_safety.py
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import REPO_ROOT, save  # noqa: E402
from common import BASES, BASE_LABEL, Scores  # noqa: E402
from src.models.registry import LONGBENCH_TASKS  # noqa: E402

PER_DOC = REPO_ROOT / "results" / "per_document.csv"
TOL = 1e-9
ORDER = [BASE_LABEL[b] for b in ("snapkv", "laprox", "lava", "rkv", "h2o")]
KEY = {v: k for k, v in BASE_LABEL.items()}


def load_cells() -> list:
    """(base, task, {base, corr, fid}) for every cell with all three arms."""
    sc = Scores()
    out = []
    for label in ORDER:
        b = KEY[label]
        for t in LONGBENCH_TASKS:
            v = {"base": sc.get("llama3_8b", "base", b, 128, t),
                 "corr": sc.get("llama3_8b", "rescue_ungated", b, 128, t),
                 "fid": sc.get("llama3_8b", "rescue", b, 128, t)}
            if all(x is not None for x in v.values()):
                out.append((label, t, v))
    return out


def load_quadrants() -> dict:
    """Selector decision outcomes, recomputed from the per-document scores.

    A document counts only where the two caches score differently and the
    selector's own score matches one of them; the rest carry no decision.
    """
    if not PER_DOC.exists():
        raise SystemExit(f"{PER_DOC} not found; run tools/export_per_document.py")
    quad = {k: {"mag": [], "contrib": []} for k in ("11", "10", "00", "01")}
    per_base = defaultdict(lambda: {"dec": 0, "acc": 0, "net": []})
    with open(PER_DOC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            b, c, s = (float(r["score_base"]), float(r["score_corr"]),
                       float(r["score_selector"]))
            if abs(c - b) <= TOL:
                continue
            near_c, near_b = abs(s - c) <= TOL, abs(s - b) <= TOL
            if not (near_c or near_b):
                continue
            right = (c > b) == near_c
            quad[f"{int(near_c)}{int(right)}"]["mag"].append(abs(c - b))
            quad[f"{int(near_c)}{int(right)}"]["contrib"].append((c - b) if near_c else 0.0)
            pb = per_base[BASE_LABEL[r["base"]]]
            pb["dec"] += 1
            pb["acc"] += int(near_c)
            pb["net"].append((c - b) if near_c else 0.0)
    n = sum(len(v["mag"]) for v in quad.values())
    return {
        "n_decisive": n,
        "accuracy": (len(quad["11"]["mag"]) + len(quad["01"]["mag"])) / n,
        "net_per_doc": sum(sum(v["contrib"]) for v in quad.values()) / n,
        "quadrants": {k: {"n": len(v["mag"]),
                          "share": len(v["mag"]) / n,
                          "mean_abs": st.mean(v["mag"]) if v["mag"] else 0.0,
                          "contrib": sum(v["contrib"]) / n}
                      for k, v in quad.items()},
        "per_base": {b: {"accept_rate": v["acc"] / v["dec"], "net": st.mean(v["net"])}
                     for b, v in per_base.items()},
    }


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", type=Path)
args = ap.parse_args()

cells = load_cells()
Q = load_quadrants()
NET_FALL = {b: Q["per_base"][b]["net"] for b in ORDER}

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "DejaVu Serif"],
                     "font.size": 15, "axes.linewidth": 1.0,
                     "mathtext.fontset": "cm"})

COL={"SnapKV":"#9ecae1","LaProx":"#4292c6","LAVa":"#2171b5","R-KV":"#08306b","H2O":"#e08214"}

fig,(ax,axR)=plt.subplots(1,2,figsize=(11.6,5.54),gridspec_kw={"width_ratios":[1,0.96]})
lim_lo,lim_hi=-10.4,13.2
ax.axhspan(lim_lo,0,color="#f7d9d0",alpha=.35,zorder=0)
ax.axvspan(lim_lo,0,color="#f7d9d0",alpha=.20,zorder=0)
ax.plot([lim_lo,lim_hi],[lim_lo,lim_hi],ls=(0,(5,4)),c="#888",lw=1.2,zorder=1)
ax.axhline(0,c="#444",lw=.9,zorder=1); ax.axvline(0,c="#444",lw=.9,zorder=1)

for b in ORDER:
    xs=[v["corr"]-v["base"] for bb,_,v in cells if bb==b]
    ys=[v["fid"] -v["base"] for bb,_,v in cells if bb==b]
    ax.scatter(xs,ys,s=52,c=COL[b],edgecolor="white",linewidth=.8,label=b,zorder=3,alpha=.95)

cx=min(cells,key=lambda c:c[2]["corr"]-c[2]["base"])      # worst under always-on
cy=min(cells,key=lambda c:c[2]["fid"] -c[2]["base"])      # worst under the selector
worst_x=cx[2]["corr"]-cx[2]["base"]; wx_y=cx[2]["fid"]-cx[2]["base"]
worst_y=cy[2]["fid"] -cy[2]["base"]; wy_x=cy[2]["corr"]-cy[2]["base"]
# dotted drops from each extreme point to the ZERO line of the axis it is
# extreme on, so the reader can see which dot each "worst" refers to
ax.plot([worst_x,worst_x],[0,wx_y],ls=(0,(2,3)),c=COL[cx[0]],lw=1.2,zorder=2)
ax.plot([wy_x,0],[worst_y,worst_y],ls=(0,(2,3)),c=COL[cy[0]],lw=1.2,zorder=2)
ax.annotate(f"worst {worst_x:.2f}", xy=(worst_x,0), xytext=(worst_x+0.45,-3.6),
            fontsize=15, color=COL[cx[0]], ha="left", va="center",
            arrowprops=dict(arrowstyle="->",color=COL[cx[0]],lw=1.1,shrinkA=0,shrinkB=3))
ax.annotate(f"worst {worst_y:.2f}", xy=(0,worst_y), xytext=(-9.9,-6.5),
            fontsize=15, color=COL[cy[0]], ha="left", va="center",
            arrowprops=dict(arrowstyle="->",color=COL[cy[0]],lw=1.1,shrinkA=0,shrinkB=3))
n_harm_x=sum(1 for _,_,v in cells if v["corr"]-v["base"]<=-1.0)
n_harm_y=sum(1 for _,_,v in cells if v["fid"] -v["base"]<=-1.0)
ax.text(.035,.965,f"harmful cells ($\\leq\\!-1$)\nalways-on  {n_harm_x}/{len(cells)}\n"
                  f"selector   {n_harm_y}/{len(cells)}",
        transform=ax.transAxes,va="top",ha="left",fontsize=14.5,
        bbox=dict(boxstyle="round,pad=0.42",fc="white",ec="#bbb",lw=.8,alpha=.93))
ax.set_xlim(lim_lo,lim_hi); ax.set_ylim(lim_lo,lim_hi); ax.set_aspect("equal")
ax.set_xlabel("$\\Delta$ with correction always applied")
ax.set_ylabel("$\\Delta$ with fidelity selector (RESCUE)")
ax.set_title("Per-cell effect of verifying the correction",pad=10,fontsize=17,
             fontweight="bold",color="#1b2836")
ax.legend(loc="lower right",frameon=True,framealpha=.93,fontsize=14,ncol=2,
          handletextpad=.35,borderpad=.45,columnspacing=.9,labelspacing=.35)
ax.grid(alpha=.22,lw=.7)
for sp in ("top","right"): ax.spines[sp].set_visible(False)

# ---------------- right: per-document decision outcomes ----------------
# Only the two ACCEPT outcomes move the score: rejecting returns the base cache,
# so it contributes exactly zero and needs no bar. Three bars, no legend.
C_GAIN, C_LOSS = "#e08214", "#a4442e"
q_hit, q_miss = Q["quadrants"]["11"], Q["quadrants"]["10"]
net = Q["net_per_doc"]
BARS = [("accepted and right", q_hit["contrib"], C_GAIN, q_hit),
        ("accepted and wrong", q_miss["contrib"], C_LOSS, q_miss),
        ("net", net, "#5b6b7c", None)]
for yy, (lab, val, col, q) in zip([2, 1, 0], BARS):
    axR.barh(yy, val, height=0.50, color=col, zorder=3)
    axR.text(val + (0.18 if val > 0 else -0.22), yy, f"{val:+.2f}",
             va="center", ha="left" if val > 0 else "right", fontsize=15,
             fontweight="bold", color="#33404d")

axR.axhline(0.5, color="#c9d2da", lw=1.0, zorder=1)
axR.axvline(0, c="#5b6b7c", lw=1.1, zorder=4)
axR.set_yticks([2, 1, 0])
axR.set_yticklabels([f"{q_hit['n']/Q['n_decisive']*100:.0f}% of documents\naccepted and right",
                     f"{q_miss['n']/Q['n_decisive']*100:.0f}% of documents\naccepted and wrong",
                     "net"], fontsize=15)
axR.tick_params(axis="y", length=0, pad=6)
# the other 49% are rejections: they return the base cache, so they contribute
# nothing and are left off the chart entirely (stated in the caption).
axR.text(2.33, 2.0, f"{q_hit['mean_abs']:.1f} each", fontsize=13.5, color="white",
         va="center", ha="center", fontweight="bold")
axR.text(-1.31, 1.0, f"{q_miss['mean_abs']:.1f} each", fontsize=13.5, color="white",
         va="center", ha="center", fontweight="bold")
# Per base policy: how often the selector accepts, and what that is worth. The
# acceptance rate tracks how much the base was missing, which is the residual
# claim arriving by a second route.
PB=Q.get("per_base") or {}
nets={b:(PB.get(b,{}).get("net") if PB.get(b,{}).get("net") is not None else NET_FALL[b])
      for b in ORDER}
rates={b:PB.get(b,{}).get("accept_rate",0.0) for b in ORDER}
order=sorted(ORDER,key=lambda b:-nets[b])
axR.axhline(-0.55,color="#c9d2da",lw=1.0,zorder=1)
axR.text(-3.3,-1.05,"per base policy",fontsize=15,fontweight="bold",color="#1b2836",
         va="center",ha="left")
for r,b in enumerate(order):
    yy=-1.85-0.42*r
    axR.barh(yy,nets[b],height=0.30,color=COL[b],zorder=3)
    axR.text(nets[b]+0.12,yy,f"{nets[b]:+.2f}",va="center",ha="left",fontsize=13.5,
             fontweight="bold",color="#1b2836")
    axR.text(-0.16,yy,f"{b}  accepts {100*rates[b]:.0f}%",va="center",ha="right",
             fontsize=13.5,color="#33404d")
axR.set_xlim(-3.9, 6.2); axR.set_ylim(-3.95, 2.75)
axR.set_xticks([-2, 0, 2, 4])
axR.set_xlabel("Points gained over the base policy, per document", fontsize=13.5)
axR.set_title("Why $55.7\\%$ accuracy is enough", pad=10, fontsize=17,
              fontweight="bold", color="#1b2836")
for sp in ("top", "right", "left"): axR.spines[sp].set_visible(False)
axR.grid(axis="x", alpha=.22, lw=.7); axR.set_axisbelow(True)

fig.tight_layout()
save(fig, "fig3_safety", args.out)
print(f"  {len(cells)} cells, harmful {n_harm_x} -> {n_harm_y}, "
      f"{Q['n_decisive']} deciding documents")
