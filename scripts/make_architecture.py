"""Architecture (overview) figure of Disagreement-Reserved Triage -> figures/fig_architecture.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
FILL_B, FILL_O, FILL_N = "#e8f1fb", "#fdeee7", "#f3f2ef"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 6.6, "mathtext.fontset": "stix"})

fig = plt.figure(figsize=(3.45, 3.25 * 0.85))  # cropped to the boxes (y 13.2-98.2)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(13.2, 98.2); ax.axis("off")


def box(x, y, w, h, text, edge=INK2, fill=FILL_N, bold_first=True, lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", lw=lw, ls=ls,
                                edgecolor=edge, facecolor=fill))
    lines = text.split("\n")
    n = len(lines); step = 3.55
    y0 = y + h / 2 + (n - 1) * step / 2
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y0 - i * step, ln, ha="center", va="center", color=INK,
                fontweight="bold" if (i == 0 and bold_first) else "normal",
                fontsize=6.9 if i == 0 else 6.3)


def arrow(p, q, color=INK2, ls="-", rad=0.0, lw=0.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# 1. input + features
box(18, 89, 64, 8, "Transaction stream, review window $w$\n$n_w$ transactions $X_w$")
arrow((50, 88.4), (50, 84.6))
box(18, 76.5, 64, 7.5, "Leakage-safe features\nstrictly earlier history only; no labels used")
arrow((36, 75.9), (23.5, 70.6)); arrow((64, 75.9), (71.5, 70.6))
# 2. two signals
box(3, 59.5, 41, 10.5, "Supervised score $s(x)$\nLightGBM trained on\nlabelled history", edge=BLUE, fill=FILL_B)
box(51, 59.5, 41, 10.5, "Anomaly score $a(x)$\nIsolation Forest,\nfitted without labels", edge=ORANGE, fill=FILL_O)
# 3. budget
box(29, 47.5, 40, 7.5, "Review budget $k_w = c\\,q\\,n_w$\n$k_w^A=(1-\\rho)k_w$,  $k_w^B=\\rho k_w$")
# 4. channels
box(3, 30, 41, 12, "Exploit channel $S^A_w$\ntop-$k^A_w$ of $X_w$ by $s$\n(= SUP's own top picks)", edge=BLUE, fill=FILL_B)
box(51, 30, 41, 12, "Disagreement channel $S^B_w$\ntop-$k^B_w$ of $X_w\\setminus S^A_w$\nby $a$", edge=ORANGE, fill=FILL_O)
arrow((23.5, 58.9), (23.5, 42.6), color=BLUE)
arrow((71.5, 58.9), (71.5, 42.6), color=ORANGE)
arrow((38, 46.9), (32, 42.6)); arrow((60, 46.9), (66, 42.6))
arrow((44.6, 36), (50.4, 36), color=INK2)
ax.text(47.5, 38.6, "rest", ha="center", va="bottom", fontsize=5.8, color=INK2, style="italic")
# 5. review + outcomes
arrow((23.5, 29.4), (40, 22.6), color=BLUE); arrow((71.5, 29.4), (60, 22.6), color=ORANGE)
box(22, 14.5, 56, 7.5, "Analyst review queue $S_w = S^A_w \\cup S^B_w$\n$|S_w| = k_w$; frauds retrieved, outcomes recorded")
# 6. feedback (adaptive variant)
ax.plot([78.6, 96.5, 96.5], [18.2, 18.2, 51.2], color=INK2, lw=0.7, ls=(0, (3, 2)))
ax.add_patch(FancyArrowPatch((96.5, 51.2), (69.9, 51.2), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=INK2,
                             ls=(0, (3, 2)), shrinkA=0, shrinkB=0))
ax.text(98.6, 34.5, "DRT-ADAPT: outcomes update $\\rho$", rotation=90, ha="center", va="center", fontsize=5.7,
        color=INK2, style="italic")
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(OUT, f"fig_architecture.{ext}"), dpi=600)
print("wrote fig_architecture")
