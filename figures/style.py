"""Shared figure style.

One place for the things every figure in the paper agrees on, so a regenerated
figure matches the ones in the PDF: font sizes, the colour pair used for
"before" and "after", and where output lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tables"))

OUT_DIR = REPO_ROOT / "figures" / "out"

# Blue is the arm without the selector, orange the arm with it; grey is a
# reference line or an inactive element. Kept consistent across figures so the
# same colour means the same thing in every panel of the paper.
BLUE, ORANGE, GREY = "#3b6ea5", "#d9822b", "#9aa0a6"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,   # what the paper's own figures were written at
    "savefig.bbox": "tight",
})


def save(fig, name: str, out: Path | None = None) -> None:
    """Write both the PNG the paper includes and a PDF for print.

    300 dpi, which is what the figures in the paper were written at, so a
    regenerated file is comparable to the one in the PDF rather than merely
    similar.
    """
    dest = out or OUT_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = dest.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"-> {p}")
    plt.close(fig)
