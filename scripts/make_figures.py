"""Paper figures from results/analysis (palette validated with the dataviz validator:
slots 1-4 pass CVD/normal-vision checks; contrast relief = legend + direct labels + markers)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
A = os.environ.get("ANALYSIS_DIR", os.path.join(ROOT, "results", "analysis"))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SLOT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"ulb": "ULB (real)", "banksim": "BankSim", "tabformer": "TabFormer"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def fig_budget():
    m = pd.read_csv(os.path.join(A, "main_base.csv"))
    series = [("SUP", "Supervised (SUP)", SLOT["blue"], "o", "-"),
              ("DRT", r"DRT ($\rho$=0.10)", SLOT["orange"], "s", "-"),
              ("FUSE60", "Fusion 60/40", SLOT["aqua"], "^", "--"),
              ("ANO", "Anomaly only", SLOT["yellow"], "D", ":")]
    ds_list = [d for d in ["ulb", "banksim", "tabformer"] if d in set(m.dataset)]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.45, 1.75), sharey=True)
    for ax, ds in zip(axes, ds_list):
        for key, lab, col, mk, ls in series:
            d = m[(m.dataset == ds) & (m.policy == key)].sort_values("c")
            ax.plot(d.c, d.recall, color=col, lw=1.2, ls=ls, marker=mk, ms=3.2,
                    markeredgecolor="white", markeredgewidth=0.4, label=lab)
            ax.fill_between(d.c, d.lo, d.hi, color=col, alpha=0.10, lw=0)
        ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["½", "1", "2", "4"])
        ax.set_title(NAMES[ds], color=INK, pad=3); ax.set_ylim(0, 1.02); style(ax)
        ax.set_xlabel("Budget multiplier $c$")
    axes[0].set_ylabel("Recall@budget")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.4)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"fig_budget.{ext}"), dpi=600)
    plt.close(fig)


def fig_frontier():
    f = pd.read_csv(os.path.join(A, "frontier_points.csv"))
    ds_list = [d for d in ["ulb", "banksim", "tabformer"] if d in set(f.dataset)]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.45, 1.8))
    for ax, ds in zip(axes, ds_list):
        for fam, col, mk, ls, lab in [("DRT", SLOT["orange"], "s", "-", "DRT ($\\rho$ sweep)"),
                                      ("FUSE", SLOT["aqua"], "^", "--", "Fusion ($w$ sweep)")]:
            d = f[(f.dataset == ds) & (f.family == fam)].sort_values("overall", ascending=False)
            ax.plot(d.overall, d.novel, color=col, ls=ls, lw=1.2, marker=mk, ms=3.0,
                    markeredgecolor="white", markeredgewidth=0.4, label=lab)
        s = f[(f.dataset == ds) & (f.policy == "SUP")].iloc[0]
        ax.plot([s.overall], [s.novel], marker="*", ms=7, color=SLOT["blue"], ls="none",
                markeredgecolor="white", markeredgewidth=0.4, label="SUP", zorder=5)
        ax.set_title(NAMES[ds], color=INK, pad=3); style(ax)
        ax.set_xlabel("Overall recall")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3)); ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4)); ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if f[f.dataset == ds].novel.max() < 0.05 else "%.2f"))
        ax.invert_xaxis()
    axes[0].set_ylabel("Novel-fraud recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"fig_frontier.{ext}"), dpi=600)
    plt.close(fig)


def fig_groups():
    g = pd.read_csv(os.path.join(A, "group_signal.csv"))
    short = {"es_health": "health", "es_sportsandtoys": "sports & toys", "es_travel": "travel",
             "es_wellnessandbeauty": "wellness", "cluster0": "cluster 0", "cluster1": "cluster 1",
             "Chip": "chip", "Swipe": "swipe", "Online": "online"}
    sty = {"ulb": (SLOT["blue"], "o"), "banksim": (SLOT["orange"], "s"), "tabformer": (SLOT["aqua"], "^")}
    place = {"wellness": (-0.30, 0.20), "health": (-0.30, 0.135), "sports & toys": (0.15, 0.17),
             "chip": (0.15, 0.08), "swipe": (-0.30, 0.07)}
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6, ls=":")
    for ds in [d for d in ["ulb", "banksim", "tabformer"] if d in set(g.dataset)]:
        d = g[g.dataset == ds]; col, mk = sty[ds]
        ax.scatter(d.ano_adv_ap, d.drt_gain, s=26, color=col, marker=mk, edgecolor="white", linewidth=0.6,
                   label=NAMES[ds], zorder=3)
        for _, r in d.iterrows():
            lab = short.get(r.group, r.group)
            if lab in place:   # crowded points near the origin: labels placed with thin leader lines
                ax.annotate(lab, (r.ano_adv_ap, r.drt_gain), xytext=place[lab], textcoords="data", fontsize=6,
                            color=INK2, ha="left" if place[lab][0] > r.ano_adv_ap else "right", va="center",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=2))
            else:
                ax.annotate(lab, (r.ano_adv_ap, r.drt_gain), xytext=(4, 3), textcoords="offset points",
                            fontsize=6, color=INK2)
    ax.set_xlabel("Novel-fraud AP: anomaly detector − supervised model")
    ax.set_ylabel("DRT gain in\nnovel recall")
    style(ax); ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"fig_groups.{ext}"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    fig_budget()
    fig_frontier()
    fig_groups()
    print("figures written to", OUT)
