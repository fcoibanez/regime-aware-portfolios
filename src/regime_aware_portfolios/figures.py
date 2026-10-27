"""Figures of the paper.

Every function takes results already computed by the stages and returns a
matplotlib figure. The notebooks in ``notebooks/`` call them and save the
figures to ``output/figures``. All but the recovery figure are in shades of
gray, so that they print in black and white.

Text is set by LaTeX, in the paper's typeface, when a LaTeX installation is
found; otherwise matplotlib's own Computer Modern fonts are used, which look
nearly the same and need nothing installed.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfFile

__all__ = [
    "STYLES",
    "cost_figure",
    "ecdf_figure",
    "recession_spans",
    "recovery_figure",
    "rolling_window_figure",
    "save",
    "selection_figure",
    "strategy_name",
    "style",
    "timeline_figure",
]

#: Line style of each strategy; rolling windows share one style.
STYLES = {
    # The proposed framework and the benchmark it is measured against, in black.
    "framework": dict(ls="-", lw=1.8, color="0.0"),
    "agnostic": dict(ls="--", lw=1.3, color="0.0"),
    # Costa and Kwon solid, so it reads as a single continuous line rather than
    # one more dash pattern among several.
    "costa_kwon": dict(ls="-", lw=1.3, color="0.50"),
    "rolling": dict(ls="-.", lw=1.2, color="0.50"),
    "equal_weight": dict(ls=(0, (1, 1.4)), lw=1.2, color="0.68"),
    "minimum_variance": dict(ls=(0, (5, 2)), lw=1.1, color="0.68"),
}


def style() -> None:
    """Settings shared by every figure: the paper's serif typeface."""
    if shutil.which("latex"):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.sans-serif": ["CMU Serif"],
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["cmr10"],
                "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True,
            }
        )


def save(fig: plt.Figure, path, **kwargs) -> None:
    """Save a figure so that a rerun gives the same file.

    The file carries no creation date, and the tag of each embedded font subset
    is derived from the characters in it. matplotlib derives the tag from
    Python's ``hash`` of an object that differs from run to run.
    """
    options = {"dpi": 300, "transparent": True, "bbox_inches": "tight"} | kwargs
    original = PdfFile._get_subset_prefix
    PdfFile._get_subset_prefix = staticmethod(_subset_prefix)
    try:
        fig.savefig(path, metadata={"CreationDate": None}, **options)
    finally:
        PdfFile._get_subset_prefix = staticmethod(original)


def _subset_prefix(charset) -> str:
    """Six capital letters and a plus sign, determined by the characters."""
    digest = hashlib.sha256(repr(sorted(charset)).encode()).digest()
    return "".join(chr(ord("A") + b % 26) for b in digest[:6]) + "+"


def _text(text: str) -> str:
    """Escape LaTeX's special characters when LaTeX sets the text."""
    if plt.rcParams["text.usetex"]:
        return text.replace("%", r"\%").replace("&", r"\&")
    return text


def _dash(text: str) -> str:
    """A range written with "--", as LaTeX sets an en dash."""
    return text if plt.rcParams["text.usetex"] else text.replace("--", "\N{EN DASH}")


def strategy_name(strategy: str) -> str:
    """Name of a strategy in legends."""
    if strategy.startswith("rolling_"):
        return f"Rolling OLS ({strategy.removeprefix('rolling_')}m)"
    names = {
        "framework": "RWLS",
        "agnostic": "Regime-agnostic",
        "equal_weight": "$1/N$",
        "minimum_variance": "Min. variance",
        "costa_kwon": _text("Costa & Kwon"),
    }
    return names[strategy]


def _kind(strategy: str) -> str:
    return "rolling" if strategy.startswith("rolling_") else strategy


def _legend_below(fig, axes) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
        fontsize=7,
    )


# --------------------------------------------------------------- performance


def ecdf_figure(
    stats: pd.DataFrame, strategies: Sequence[str], phis: Sequence[float]
) -> plt.Figure:
    """Figure 4: distribution of Sharpe ratios across paths, one panel per phi.

    Each curve is the cumulative distribution of a kernel density estimate.
    """
    fig, axes = plt.subplots(1, len(phis), figsize=(7, 3), sharex=True, sharey=True)
    for ax, phi in zip(axes, phis, strict=True):
        sel = stats[stats["phi"] == float(phi)]
        for s in strategies:
            sns.kdeplot(
                data=sel.loc[sel["strategy"] == s, "Sharpe Ratio"].to_numpy(),
                cumulative=True,
                ax=ax,
                zorder=25,
                label=strategy_name(s) if phi == phis[0] else "",
                **STYLES[_kind(s)],
            )
        ax.set_title(rf"$\varphi={phi:g}$")
        ax.set_xlabel("Sharpe Ratio")
        ax.grid(ls="--", alpha=0.5, zorder=-25)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1)
        ax.tick_params(bottom=False, left=False)
    # The panels show a cumulative distribution, not a density; seaborn labels
    # the axis "Density" regardless.
    axes[0].set_ylabel("Cumulative share of paths")
    _legend_below(fig, axes)
    fig.tight_layout()
    return fig


def cost_figure(
    gross: pd.DataFrame,
    turnover: pd.DataFrame,
    strategies: Sequence[str],
    phis: Sequence[float],
    costs: np.ndarray,
    estimate: float,
    lower_estimate: float,
) -> plt.Figure:
    """Mean net Sharpe ratio across paths as a function of the one-way cost.

    ``gross`` and ``turnover`` are indexed by (strategy, phi, path), one column
    per month. The opening purchase is not charged, as in the paper's net
    returns. The vertical line marks ``estimate`` and the band reaches down to
    ``lower_estimate``.
    """
    fig, axes = plt.subplots(1, len(phis), figsize=(7, 3), sharey=True)
    for ax, phi in zip(axes, phis, strict=True):
        for s in strategies:
            r = gross.loc[(s, float(phi))].to_numpy()
            u = turnover.loc[(s, float(phi))].to_numpy().copy()
            u[:, 0] = 0.0
            curve = []
            for c in costs:
                net = r - c * u
                sr = net.mean(axis=1) / net.std(axis=1, ddof=1) * np.sqrt(12)
                curve.append(sr.mean())
            ax.plot(
                100 * costs,
                curve,
                label=strategy_name(s) if phi == phis[0] else "",
                **STYLES[_kind(s)],
            )
        ax.axvline(100 * estimate, color="0.35", lw=1)
        ax.axvspan(100 * lower_estimate, 100 * estimate, color="0.85", zorder=-30)
        ax.set_title(rf"$\varphi={phi:g}$")
        ax.grid(ls="--", alpha=0.5, zorder=-25)
        ax.tick_params(bottom=False, left=False)
    axes[0].set_ylabel("Ann. Sharpe Ratio (net)")
    fig.supxlabel(_text("One-way cost per unit turnover (%)"), fontsize=9)
    _legend_below(fig, axes)
    fig.tight_layout()
    return fig


def rolling_window_figure(differences: pd.DataFrame) -> plt.Figure:
    """Paired difference in Sharpe ratio against rolling OLS, by window.

    ``differences`` has columns phi, window, Delta and Std. Err.
    """
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    phis = sorted(differences["phi"].unique())
    for phi, marker in zip(phis, ["s", "o", "^"], strict=False):
        sub = differences[differences["phi"] == phi].sort_values("window")
        ax.errorbar(
            sub["window"],
            sub["Delta"],
            yerr=1.96 * sub["Std. Err."],
            marker=marker,
            c="k",
            markerfacecolor="white",
            markeredgewidth=1,
            capsize=3,
            lw=1.2,
            label=rf"$\varphi={phi:g}$",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Rolling window (months)")
    ax.set_ylabel(r"$\Delta$ Sharpe vs. rolling OLS")
    ax.set_xticks(sorted(differences["window"].unique()))
    ax.grid(ls="--", alpha=0.5, zorder=-25)
    ax.tick_params(bottom=False, left=False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------- regimes

#: Line style of each criterion in the regime-selection figure.
_CRITERIA = {
    "BIC": dict(ls="-", lw=1.6, marker="s", ms=3.6, mfc="white", mew=1.0),
    "ICL": dict(ls="-", lw=1.2, marker="o", ms=3.2, mfc="white", mew=0.9),
    "HQIC": dict(ls=(0, (1.4, 1.4)), lw=1.2),
    "AIC": dict(ls=(0, (5, 1.6, 1, 1.6)), lw=1.2),
}


def _sample_title(sample: str) -> str:
    """'1963-07 to 1983-12' -> 'Jul 1963--Dec 1983'; other labels unchanged."""
    if " to " not in sample:
        return sample
    first, last = (pd.Period(p, "M").strftime("%b %Y") for p in sample.split(" to "))
    return _dash(f"{first}--{last}")


def selection_figure(criteria: pd.DataFrame) -> plt.Figure:
    """Figure 1: information criteria by number of regimes, one panel per sample.

    ``criteria`` is ``criteria.parquet`` of stage 8, with four samples. A
    triangle marks where each criterion bottoms out.
    """
    samples = list(dict.fromkeys(criteria["sample"]))
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), sharex=True)
    for ax, sample in zip(axes.ravel(), samples, strict=True):
        sub = criteria[criteria["sample"] == sample].set_index("n_regimes")
        for name, spec in _CRITERIA.items():
            shade = "0.15" if name == "BIC" else "0.45"
            y = sub[name].astype(float)
            ax.plot(list(y.index), y.to_numpy(), color=shade, label=name, **spec)
            ax.plot(
                [y.idxmin()],
                [y.min()],
                marker="v",
                ms=5,
                color=shade,
                mec="none",
                zorder=6,
            )
        ax.set_title(_sample_title(sample), fontsize=9.5)
        ax.set_xticks(list(sub.index))
        ax.grid(ls="--", alpha=0.45, lw=0.5)
        ax.tick_params(bottom=False, left=False, labelsize=8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    for ax in axes[:, 0]:
        ax.set_ylabel("Information criterion", fontsize=9)
    fig.supxlabel("Number of market regimes", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def recession_spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The first and last month of each run of True in a monthly series."""
    values = mask.to_numpy(dtype=bool)
    spans, i = [], 0
    while i < len(values):
        if values[i]:
            j = i
            while j + 1 < len(values) and values[j + 1]:
                j += 1
            spans.append((mask.index[i], mask.index[j]))
            i = j + 1
        else:
            i += 1
    return spans


def timeline_figure(
    probability: pd.Series,
    recession: pd.Series,
    label: str,
    split: str,
    smooth: int,
) -> plt.Figure:
    """Figure 2: probability of one regime over time, NBER recessions shaded.

    The probability is plotted as a centered moving average of ``smooth``
    months, and the sample is split across two rows at ``split``.
    """
    shown = probability.rolling(smooth, center=True).mean()
    cut = pd.Timestamp(split)
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.2))
    for ax, (lo, hi) in zip(
        axes, [(shown.index[0], cut), (cut, shown.index[-1])], strict=True
    ):
        for start, end in recession_spans(recession.astype(bool)):
            if end >= lo and start <= hi:
                ax.axvspan(max(start, lo), min(end, hi), color="0.86", lw=0, zorder=0)
        window = (shown.index >= lo) & (shown.index <= hi)
        ax.plot(shown.index[window], shown[window], color="0.05", lw=0.9, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(rf"$P(\mathrm{{{label}}})$", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(bottom=False, left=False, labelsize=8.5)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def recovery_figure(true: np.ndarray, estimated: np.ndarray) -> plt.Figure:
    """Figure 3, one panel: estimated against true loadings on one factor.

    ``true`` and ``estimated`` have shape (stocks, regimes); regimes are
    numbered from 1 in the legend. Each regime has its own color, with kernel
    densities of its loadings in the margins.
    """
    n, m = true.shape
    data = pd.DataFrame(
        {
            "True": true.T.ravel(),
            "Estimated": estimated.T.ravel(),
            "state": np.repeat(np.arange(1, m + 1), n),
        }
    )
    g = sns.jointplot(
        data,
        x="True",
        y="Estimated",
        hue="state",
        space=0,
        palette="Set2",
        zorder=25,
        marginal_kws=dict(bw_adjust=1),
        height=2.25,
        ratio=10,
        alpha=0.25,
        joint_kws={"edgecolor": "gray", "linewidth": 0.5, "s": 10},
    )
    g.ax_joint.grid(ls="--", zorder=-25)
    g.ax_joint.legend(loc="lower right", fontsize=8, markerscale=1, title="Regime")
    g.ax_joint.set_xlabel("True")
    g.ax_joint.set_ylabel("Estimated")
    return g.figure
