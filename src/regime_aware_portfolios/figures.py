"""Figures of the paper.

Every function takes results already computed by the stages and returns a
matplotlib figure. The notebooks in ``notebooks/`` call them and save the
figures to ``output/figures``. The figures are in shades of gray, so that they
print in black and white.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

__all__ = [
    "STRATEGY_NAMES",
    "STYLES",
    "cost_figure",
    "ecdf_figure",
    "recession_spans",
    "recovery_figure",
    "rolling_window_figure",
    "selection_figure",
    "style",
    "timeline_figure",
]

#: Names of the strategies in legends.
STRATEGY_NAMES = {
    "framework": "RWLS",
    "agnostic": "Agnostic",
    "rolling": "Roll. OLS",
    "equal_weight": "1/N",
    "minimum_variance": "Min. Var.",
    "costa_kwon": "C&K",
}

#: Line style of each strategy; rolling windows share one style.
STYLES = {
    "framework": dict(ls="-", lw=1.8, color="0.0"),
    "agnostic": dict(ls="--", lw=1.3, color="0.0"),
    "costa_kwon": dict(ls="-", lw=1.3, color="0.50"),
    "rolling": dict(ls="-.", lw=1.2, color="0.50"),
    "equal_weight": dict(ls=(0, (1, 1.4)), lw=1.2, color="0.68"),
    "minimum_variance": dict(ls=(0, (5, 2)), lw=1.1, color="0.68"),
}


def style() -> None:
    """Settings shared by every figure."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def _kind(strategy: str) -> str:
    return "rolling" if strategy.startswith("rolling_") else strategy


def _finish(ax) -> None:
    ax.grid(ls="--", alpha=0.5, zorder=-25)
    ax.tick_params(bottom=False, left=False)


def _legend_below(fig, axes, columns: int) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=columns,
        frameon=False,
    )


# --------------------------------------------------------------- performance


def ecdf_figure(
    stats: pd.DataFrame, strategies: Sequence[str], phis: Sequence[float]
) -> plt.Figure:
    """Figure 4: distribution of Sharpe ratios across paths, one panel per phi."""
    fig, axes = plt.subplots(1, len(phis), figsize=(7, 3), sharex=True, sharey=True)
    for ax, phi in zip(axes, phis, strict=True):
        sel = stats[stats["phi"] == float(phi)]
        for s in strategies:
            x = np.sort(sel.loc[sel["strategy"] == s, "Sharpe Ratio"].to_numpy())
            ax.step(
                x,
                np.arange(1, len(x) + 1) / len(x),
                where="post",
                label=STRATEGY_NAMES[_kind(s)] if phi == phis[0] else None,
                zorder=25,
                **STYLES[_kind(s)],
            )
        ax.set_title(rf"$\varphi={phi:g}$")
        ax.set_xlabel("Sharpe Ratio")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1)
        _finish(ax)
    axes[0].set_ylabel("Cumulative share of paths")
    fig.tight_layout()
    _legend_below(fig, axes, len(strategies))
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
                label=STRATEGY_NAMES[_kind(s)] if phi == phis[0] else None,
                **STYLES[_kind(s)],
            )
        ax.axvline(100 * estimate, color="0.35", lw=1)
        ax.axvspan(100 * lower_estimate, 100 * estimate, color="0.85", zorder=-30)
        ax.set_title(rf"$\varphi={phi:g}$")
        _finish(ax)
    axes[0].set_ylabel("Ann. Sharpe Ratio (net)")
    fig.supxlabel("One-way cost per unit turnover (%)", fontsize=9)
    fig.tight_layout()
    _legend_below(fig, axes, len(strategies))
    return fig


def rolling_window_figure(differences: pd.DataFrame) -> plt.Figure:
    """Paired difference in Sharpe ratio against rolling OLS, by window.

    ``differences`` has columns phi, window, Delta and Std. Err.
    """
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    markers = ("o", "s", "^", "D", "v")
    for marker, (phi, sub) in zip(markers, differences.groupby("phi"), strict=False):
        sub = sub.sort_values("window")
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
    _finish(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------- regimes


def selection_figure(criteria: pd.DataFrame) -> plt.Figure:
    """Figure 1: information criteria by number of regimes, one panel per sample.

    ``criteria`` is ``criteria.parquet`` of stage 8. A filled marker shows the
    number of regimes each criterion selects.
    """
    names = ("BIC", "ICL", "HQIC", "AIC")
    lines = {
        "BIC": dict(ls="-", color="0.0", marker="o"),
        "ICL": dict(ls="--", color="0.0", marker="s"),
        "HQIC": dict(ls="-.", color="0.45", marker="^"),
        "AIC": dict(ls=":", color="0.45", marker="D"),
    }
    samples = list(dict.fromkeys(criteria["sample"]))
    fig, axes = plt.subplots(1, len(samples), figsize=(7, 2.6))
    for ax, sample in zip(np.atleast_1d(axes), samples, strict=True):
        sub = criteria[criteria["sample"] == sample].set_index("n_regimes")
        for name in names:
            spec = lines[name]
            ax.plot(
                sub.index,
                sub[name],
                ls=spec["ls"],
                color=spec["color"],
                lw=1.1,
                label=name if sample == samples[0] else None,
            )
            best = sub[name].idxmin()
            ax.plot(
                best,
                sub.loc[best, name],
                marker=spec["marker"],
                color=spec["color"],
                ms=5,
                ls="none",
            )
        ax.set_title(sample)
        ax.set_xticks(list(sub.index))
        ax.set_xlabel("Number of regimes")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(_thousands))
        _finish(ax)
    np.atleast_1d(axes)[0].set_ylabel("Criterion (lower is better)")
    fig.tight_layout()
    _legend_below(fig, np.atleast_1d(axes), len(names))
    return fig


def _thousands(value: float, _) -> str:
    """Axis label in thousands, with a typographic minus sign."""
    return f"{value / 1e3:g}k".replace("-", "\N{MINUS SIGN}")


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
        ax.set_ylabel(rf"$P(\mathrm{{{label}}})$")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(bottom=False, left=False)
    fig.tight_layout()
    return fig


def recovery_figure(
    true: np.ndarray,
    estimated: np.ndarray,
    labels: Sequence[str],
    max_points: int = 4000,
    seed: int = 0,
) -> plt.Figure:
    """Figure 3, one panel: estimated against true loadings on one factor.

    ``true`` and ``estimated`` have shape (observations, regimes). Each regime is
    drawn in its own shade, with the kernel densities of the true (solid) and
    estimated (dashed) loadings in the margins. At most ``max_points`` points of
    each regime are drawn, chosen at random; the densities use them all.
    """
    shades = np.linspace(0.0, 0.6, len(labels))
    markers = ("o", "s", "^", "D", "v")
    fig = plt.figure(figsize=(2.6, 2.6))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    side = fig.add_subplot(grid[1, 1], sharey=ax)
    rng = np.random.default_rng(seed)
    lo = np.percentile(np.r_[true.ravel(), estimated.ravel()], 0.5)
    hi = np.percentile(np.r_[true.ravel(), estimated.ravel()], 99.5)
    grid_x = np.linspace(lo, hi, 200)
    for s, label in enumerate(labels):
        color = str(shades[s])
        t, e = true[:, s], estimated[:, s]
        keep = rng.choice(len(t), min(max_points, len(t)), replace=False)
        ax.scatter(
            t[keep],
            e[keep],
            s=3,
            marker=markers[s % len(markers)],
            color=color,
            alpha=0.25,
            lw=0,
            label=label,
            rasterized=True,
        )
        top.plot(grid_x, gaussian_kde(t)(grid_x), color=color, lw=1)
        top.plot(grid_x, gaussian_kde(e)(grid_x), color=color, lw=1, ls="--")
        side.plot(gaussian_kde(e)(grid_x), grid_x, color=color, lw=1, ls="--")
        side.plot(gaussian_kde(t)(grid_x), grid_x, color=color, lw=1)
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True loading")
    ax.set_ylabel("Estimated loading")
    legend = ax.legend(loc="upper left", markerscale=3, frameon=False, fontsize=7)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    for a in (top, side):
        a.axis("off")
    ax.tick_params(bottom=False, left=False)
    return fig
