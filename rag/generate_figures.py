"""
Generates the figures for the methods/results write-up from the real measured
numbers of the technical test phase. Output: docs/figures/*.png
"""

import sys, io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = Path(__file__).parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Shared style — clean, publication-like
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})
CRIMSON = "#b5342b"; INDIGO = "#5246b8"; TEAL = "#0c6b53"; SAND = "#b9a071"; CORAL = "#c2562f"


# ── Figure 1: corpus composition by ASH presentation tier ──────────────────────
def fig_tiers():
    tiers = ["Plenary", "Oral", "Poster", "Publication-only"]
    counts = [6, 1092, 5330, 1827]
    colors = [INDIGO, TEAL, SAND, CORAL]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(tiers, counts, color=colors, width=0.62)
    ax.set_ylabel("Number of abstracts")
    ax.set_title("Corpus composition by ASH 2025 presentation tier (n = 8,255)", fontsize=12, weight="bold")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 60, f"{c:,}", ha="center", fontsize=10, weight="medium")
    ax.set_ylim(0, max(counts) * 1.12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_corpus_tiers.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: hallucination stress test (grounding under adversarial queries) ───
def fig_stress():
    cats = ["Absent\ntrials/drugs", "False\npremise", "Ambiguous\nquery", "Answerable\n(control)"]
    passed = [3, 2, 2, 1]
    total  = [3, 2, 2, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(cats))
    ax.bar(x, total, color="#e7e0d6", width=0.6, label="Probes")
    ax.bar(x, passed, color=CRIMSON, width=0.6, label="Correct behaviour")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats)
    ax.set_ylabel("Number of probes")
    ax.set_title("Hallucination stress test — grounding behaviour (8/8 correct)", fontsize=12, weight="bold")
    ax.set_ylim(0, 3.6)
    for i, (p, t) in enumerate(zip(passed, total)):
        ax.text(i, p + 0.07, f"{p}/{t}", ha="center", fontsize=10, weight="medium")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_stress_test.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: groundedness (citation faithfulness) on exploratory queries ───────
def fig_faithfulness():
    queries = ["Teclistamab\nefficacy", "Cilta-cel\nneurotoxicity",
               "Luspatercept\nthalassemia", "CAR-T\n(broad)", "AML\n(broad)"]
    rates = [92, 100, 100, 95, 100]   # % cited claims supported by the cited fragment (LLM judge)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(queries, rates, color=INDIGO, width=0.6)
    ax.axhline(90, color=CRIMSON, ls="--", lw=1.3, label="Pass threshold (90%)")
    ax.set_ylabel("Cited claims supported (%)")
    ax.set_ylim(60, 104)
    ax.set_title("Citation faithfulness on exploratory queries (LLM-judge)", fontsize=12, weight="bold")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width()/2, r + 0.6, f"{r}%", ha="center", fontsize=10, weight="medium")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_faithfulness.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_tiers(); fig_stress(); fig_faithfulness()
    print(f"Figures written to {OUT}:")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name, f"({p.stat().st_size//1024} KB)")
