"""Tables and numbers of the paper, written as LaTeX.

Every function here takes results already computed by the stages and returns the
text of one table, in the layout the paper includes with ``\\input``. Nothing is
estimated here. The notebooks in ``notebooks/`` call these functions and write
their output to ``output/``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

__all__ = [
    "APPENDIX_PREFIX",
    "DECOMPOSITION_STEPS",
    "FACTOR_LABELS",
    "OBJECTIVE_PAIRS",
    "PERFORMANCE_ROWS",
    "decomposition_table",
    "detection_table",
    "forecast_quality_table",
    "macros",
    "momentum_table",
    "number",
    "objective_table",
    "paired_measures_table",
    "percent",
    "performance_table",
    "regime_conditional_table",
    "regime_panels",
    "regime_parameter_table",
    "significance_table",
    "starred",
    "state_selection_table",
    "table_columns",
    "write",
]

#: Names of the factors in the paper, where they differ from French's files.
FACTOR_LABELS = {"Mkt-RF": "Mkt-Rf", "Mom": "UMD"}

#: Prefix of the Online Appendix files of each appendix study.
APPENDIX_PREFIX = {"alt_calibration": "alt", "four_regimes": "m4"}

#: Table 4, Panel A: one change per row, from Costa and Kwon to the framework
#: under the mean--variance objective.
DECOMPOSITION_STEPS = (
    ("State count", "costa_kwon", "path_regimes"),
    ("Emission dimension", "path_regimes", "path_factors"),
    ("Probabilistic forecast", "path_factors", "path_forecast"),
    ("Diagonal factor covariance", "path_forecast", "path_covariance"),
    ("RWLS estimator", "path_covariance", "framework_mvo"),
)

#: Table 4, Panel B: (row suffix, mean--variance, expected utility).
OBJECTIVE_PAIRS = (
    ("", "framework_mvo", "framework"),
    (" (known)", "oracle_mvo", "oracle"),
)


def table_columns(rolling_window: int) -> dict[str, str]:
    """Columns of Tables 2 and 3: the framework, then the five benchmarks."""
    return {
        "framework": "RWLS",
        "agnostic": "Agnostic",
        f"rolling_{rolling_window}": "Roll. OLS",
        "equal_weight": "$1/N$",
        "minimum_variance": "Min. Var.",
        "costa_kwon": r"C\&K",
    }


# ------------------------------------------------------------------ formatting


def percent(x: float, digits: int = 2) -> str:
    """A fraction as a percentage: 0.1234 -> 12.34\\%."""
    return number(100 * x, digits) + r"\%"


def number(x: float, digits: int = 2) -> str:
    """A number to ``digits`` decimals, without the sign of a rounded zero."""
    text = f"{x:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


# Space as wide as three stars. A starred number ends with it, its stars printed
# over it; the other cells of a right-aligned column holding starred numbers end
# with it too, so that every number in the column lines up.
_STAR_SPACE = r"\phantom{$^{***}$}"


def starred(x: float, p: float, digits: int = 3) -> str:
    """A number with the stars of its p-value, aligned on the stars' width."""
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    mark = rf"\rlap{{$^{{{stars}}}$}}" if stars else ""
    return f"{number(x, digits)}{mark}{_STAR_SPACE}"


def _phi_label(phi: float) -> str:
    return rf"$\varphi = {phi:g}$"


def _table(
    header: Sequence[str],
    blocks: Sequence[tuple[str | None, Sequence[tuple[str, Sequence[str]]]]],
    *,
    title: str | None = None,
    top: bool = True,
    last: str = r"\bottomrule",
    notes: Sequence[str] = (),
) -> str:
    """A full-width table: a header row, then blocks of rows under a title each."""
    n = len(header)
    spec = r"l@{\extracolsep{\fill}}" + "r" * (n - 1)
    lines = [rf"\begin{{tabular*}}{{\textwidth}}{{{spec}}}"]
    if top:
        lines.append(r"\toprule")
    if title:
        lines += [rf"\multicolumn{{{n}}}{{c}}{{\textit{{{title}}}}} \\", r"\midrule"]
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    for i, (name, rows) in enumerate(blocks):
        if name:
            lines += [rf"\multicolumn{{{n}}}{{l}}{{\textit{{{name}}}}} \\", r"\midrule"]
        lines += [" & ".join([label, *cells]) + r" \\" for label, cells in rows]
        if i < len(blocks) - 1:
            lines.append(r"\midrule")
    lines += [last, r"\end{tabular*}"]
    if notes:
        lines += [""] + [f"% {note}" for note in notes]
    return "\n".join(lines) + "\n"


def write(text: str, path: str | Path) -> Path:
    """Write a table or macro file, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def macros(values: dict[str, str]) -> str:
    """LaTeX commands, one per number quoted in the text: {name: text}."""
    lines = ["% Numbers quoted in the text, written by notebooks/quoted_numbers.py."]
    for name, value in values.items():
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"a LaTeX command name has letters only: {name!r}")
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- performance

#: Rows of Table 2: label in the paper, column of the path statistics, format.
PERFORMANCE_ROWS: tuple[tuple[str, str, Callable[[float], str]], ...] = (
    ("Ann. Excess Return", "Ann. Excess Return", percent),
    ("Ann. Std. Deviation", "Ann. Std. Deviation", percent),
    ("Skewness", "Skewness", number),
    ("Excess Kurtosis", "Excess Kurtosis", number),
    ("Sharpe Ratio", "Sharpe Ratio", number),
    ("Certainty Equivalent", "Certainty Equivalent", percent),
    ("Max. Drawdown", "Max. Drawdown", percent),
    (r"Value-at-Risk (95\%)", "Value-at-Risk (95%)", percent),
    ("Expected Shortfall", "Expected Shortfall", percent),
    ("Portfolio Turnover", "Portfolio Turnover", number),
    ("Avg. Num. Constituents", "Avg. Num. Holdings", lambda x: number(x, 1)),
    ("Effective Num. Bets", "Effective Num. Holdings", lambda x: number(x, 1)),
)


def performance_table(
    stats: pd.DataFrame, columns: dict[str, str], phis: Sequence[float]
) -> str:
    """Table 2: mean of each statistic across paths, strategies in columns.

    ``stats`` is ``path_statistics.parquet`` of stage 6 and ``columns`` maps each
    strategy to its column heading.
    """
    means = stats.groupby(["strategy", "phi"]).mean(numeric_only=True)
    blocks = []
    for phi in phis:
        rows = [
            (label, [fmt(means.loc[(s, float(phi)), col]) for s in columns])
            for label, col, fmt in PERFORMANCE_ROWS
        ]
        blocks.append((_phi_label(phi), rows))
    return _table(["", *columns.values()], blocks)


def significance_table(
    stats: pd.DataFrame,
    significance: pd.DataFrame,
    framework: str,
    columns: dict[str, str],
    phis: Sequence[float],
    cost: float,
) -> str:
    """Table 3: Sharpe ratios gross and net of costs, and tests on the net ones.

    ``significance`` is ``significance.parquet`` of stage 7, one row per
    benchmark and risk aversion. The first entry of ``columns`` is the framework.
    """
    means = stats.groupby(["strategy", "phi"]).mean(numeric_only=True)
    sig = significance.set_index(["benchmark", "phi"])
    blocks = []
    for phi in phis:
        phi = float(phi)
        gross = [means.loc[(s, phi), "Sharpe Ratio"] for s in columns]
        net = [means.loc[(s, phi), "Sharpe Ratio (net)"] for s in columns]

        def test(s, fmt, phi=phi):
            return "--" if s == framework else fmt(sig.loc[(s, phi)])

        rows = [
            ("Sharpe (gross)", [number(x) for x in gross]),
            ("Sharpe (net)", [number(x) for x in net]),
            ("Cost drag", [number(g - n) for g, n in zip(gross, net, strict=True)]),
            (
                r"$\Delta$ Sharpe (net)",
                [
                    test(s, lambda r: starred(r["Delta"], r["p (paired t)"]))
                    for s in columns
                ],
            ),
            ("t-stat", [test(s, lambda r: number(r["t-stat"])) for s in columns]),
            (
                r"LW reject 5\%",
                [
                    test(
                        s,
                        lambda r: percent(
                            r["Reject (favorable)"] + r["Reject (adverse)"], 1
                        ),
                    )
                    for s in columns
                ],
            ),
        ]
        # Every cell takes the stars' space, so the starred row lines up with the rest.
        rows = [
            (label, [c if c.endswith(_STAR_SPACE) else c + _STAR_SPACE for c in cells])
            for label, cells in rows
        ]
        blocks.append((_phi_label(phi), rows))
    return _table(
        ["", *columns.values()],
        blocks,
        notes=(
            f"Costs charged at {100 * cost:.4f}% per unit of two-sided turnover.",
            "Tests compare RWLS with the benchmark in each column, on net returns.",
            "***, **, * denote significance of the paired t-test at the 1%, 5% and"
            " 10% levels.",
        ),
    )


def paired_measures_table(
    stats: pd.DataFrame, treatment: str, control: str, phis: Sequence[float]
) -> str:
    """Paired tests of every measure of Table 2, ``treatment`` minus ``control``.

    Measures shown as percentages in Table 2 are differenced in percentage points.
    """
    from regime_aware_portfolios.inference import paired_comparison

    blocks = []
    for phi in phis:
        a = stats[(stats["strategy"] == treatment) & (stats["phi"] == float(phi))]
        b = stats[(stats["strategy"] == control) & (stats["phi"] == float(phi))]
        a, b = a.set_index("path"), b.set_index("path")
        rows = []
        for label, col, fmt in PERFORMANCE_ROWS:
            pp = fmt is percent
            scale = 100.0 if pp else 1.0
            r = paired_comparison(a[col] * scale, b[col] * scale)
            p = r["p (paired t)"]
            rows.append(
                (
                    label + (" (pp)" if pp else ""),
                    [
                        number(r["Delta"]),
                        number(r["Std. Err."]),
                        number(r["t-stat"]),
                        "$<$0.001" if p < 0.001 else number(p, 3),
                        percent(r["Frac. > 0"], 1),
                    ],
                )
            )
        blocks.append((_phi_label(phi), rows))
    header = [
        "",
        r"Mean $\Delta$",
        "Std. Err.",
        "t-stat",
        "p (paired t)",
        r"Frac. $>$ 0",
    ]
    return _table(header, blocks)


# -------------------------------------------------------------- decomposition


def decomposition_table(
    stats: pd.DataFrame,
    steps: Sequence[tuple[str, str, str]],
    phis: Sequence[float],
) -> str:
    """Table 4, Panel A: the paired change in Sharpe ratio at each step.

    ``steps`` lists (label, from strategy, to strategy); each step starts where
    the previous one ended, so the rows add up to the total.
    """
    from regime_aware_portfolios.inference import paired_comparison

    for (_, _, end), (_, start, _) in zip(steps, steps[1:], strict=False):
        if end != start:
            raise ValueError("each step must start where the previous one ended")
    blocks = []
    for phi in phis:
        sel = stats[stats["phi"] == float(phi)].set_index(["strategy", "path"])
        rows, turnover = [], []
        for label, a, b in steps:
            r = paired_comparison(
                sel.loc[b, "Sharpe Ratio"], sel.loc[a, "Sharpe Ratio"]
            )
            du = (
                sel.loc[b, "Portfolio Turnover"] - sel.loc[a, "Portfolio Turnover"]
            ).mean()
            turnover.append(du)
            rows.append(
                (
                    label,
                    [
                        starred(r["Delta"], r["p (paired t)"]),
                        number(r["Std. Err."], 3),
                        number(r["t-stat"]),
                        percent(r["Frac. > 0"], 1),
                        number(du),
                    ],
                )
            )
        total = (
            sel.loc[steps[-1][2], "Sharpe Ratio"] - sel.loc[steps[0][1], "Sharpe Ratio"]
        ).mean()
        rows.append(
            (
                "Total",
                [
                    f"{number(total, 3)}{_STAR_SPACE}",
                    "--",
                    "--",
                    "--",
                    number(sum(turnover)),
                ],
            )
        )
        blocks.append((_phi_label(phi), rows))
    header = [
        "",
        r"$\Delta$ Sharpe",
        "Std. Err.",
        "t-stat",
        r"Frac. $>$ 0",
        r"$\Delta$ Turnover",
    ]
    return _table(
        header,
        blocks,
        title="Panel A: Estimation choices, mean--variance objective",
        last=r"\midrule[\heavyrulewidth]",
        notes=(
            "Each row is a paired difference across the same simulated paths.",
            "Rows within a panel sum to the total by construction.",
            "***, **, * denote significance at the 1%, 5% and 10% levels"
            " (paired t-test).",
        ),
    )


def objective_table(
    stats: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
    phis: Sequence[float],
) -> str:
    """Table 4, Panel B: mean--variance against expected utility of the mixture.

    ``pairs`` lists (row suffix, mean--variance strategy, expected-utility
    strategy), for example ("", "framework_mvo", "framework").
    """
    from regime_aware_portfolios.inference import paired_comparison

    blocks = []
    for phi in phis:
        sel = stats[stats["phi"] == float(phi)].set_index(["strategy", "path"])
        rows = []
        for suffix, mvo, eu in pairs:
            for label, col, fmt, dfmt in (
                ("Sharpe Ratio", "Sharpe Ratio", number, lambda x: number(x, 3)),
                ("Certainty Equivalent", "Certainty Equivalent", percent, percent),
            ):
                r = paired_comparison(sel.loc[eu, col], sel.loc[mvo, col])
                rows.append(
                    (
                        label + suffix,
                        [
                            fmt(r["Mean (control)"]),
                            fmt(r["Mean (treatment)"]),
                            dfmt(r["Delta"]),
                            dfmt(r["Median delta"]),
                            percent(r["Frac. > 0"], 1),
                        ],
                    )
                )
        blocks.append((_phi_label(phi), rows))
    header = [
        "",
        "Mean--variance",
        "Expected utility",
        r"Mean $\Delta$",
        r"Median $\Delta$",
        r"Frac. $>$ 0",
    ]
    return _table(
        header,
        blocks,
        title="Panel B: Choice of objective, RWLS estimator",
        top=False,
        notes=(
            "Differences are expected exponential utility minus mean--variance,"
            " paired across paths.",
        ),
    )


# -------------------------------------------------------------------- regimes


def regime_conditional_table(
    conditional: pd.DataFrame, labels: Sequence[str], phis: Sequence[float]
) -> str:
    """Table 5, Panel A: framework against agnostic by the true regime."""
    rows_spec = (
        ("Share of months", "Share of months", lambda x: percent(x, 1)),
        ("Return advantage (ann.)", "Return advantage (ann.)", percent),
        ("Volatility ratio", "Volatility ratio", number),
        (r"VaR (95\%) improvement", "VaR (95%) improvement", percent),
    )
    c = conditional.set_index(["phi", "regime"])
    blocks = []
    for phi in phis:
        rows = [
            (
                label,
                [fmt(c.loc[(float(phi), s), col]) for s in range(1, len(labels) + 1)],
            )
            for label, col, fmt in rows_spec
        ]
        blocks.append((_phi_label(phi), rows))
    return _table(
        ["", *labels],
        blocks,
        title="Panel A: Performance by the regime governing each period",
        last=r"\midrule[\heavyrulewidth]",
        notes=(
            "Advantage of the proposed framework over the regime-agnostic benchmark,"
            " by the regime governing each period.",
        ),
    )


def forecast_quality_table(quality: pd.DataFrame) -> str:
    """Table 5, Panel B: hit rates and Brier scores of the regime probabilities."""
    cols = ("Hit rate", "Constant forecast", "Brier score", "Brier skill")
    rows = [(name, [number(r[c], 3) for c in cols]) for name, r in quality.iterrows()]
    return _table(
        ["", *cols],
        [(None, rows)],
        title="Panel B: Quality of the regime probabilities",
        top=False,
        notes=(
            "Scored against the realized regime over all simulated paths.",
            "The constant forecast always predicts the most frequent state; it is"
            " reported as the hit rate that forecast attains.",
            "The Brier skill score is measured against a constant forecast of each"
            " state's frequency over the scored periods.",
        ),
    )


def detection_table(detection: pd.DataFrame, columns: dict[str, str]) -> str:
    """Rejection rate of the Ledoit and Wolf test as the history lengthens.

    ``detection`` is ``detection.parquet`` of stage 7, and ``columns`` maps each
    benchmark to its column heading.
    """
    rates = detection.pivot(
        index="Months", columns="benchmark", values="Detection rate"
    )
    years = detection.groupby("Months")["Years"].first()
    lines = [
        rf"\begin{{tabular}}{{lr{'r' * len(columns)}}}",
        r"\toprule",
        "Months & Years & " + " & ".join(columns.values()) + r" \\",
        r"\midrule",
    ]
    for months in rates.index:
        cells = [percent(rates.loc[months, b], 1) for b in columns]
        lines.append(
            f"{int(months)} & {years[months]:.1f} & " + " & ".join(cells) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "% Share of simulated histories on which the Ledoit-Wolf test rejects"
        " equality of Sharpe ratios at the 5% level, framework against each"
        " benchmark, returns net of costs.",
    ]
    return "\n".join(lines) + "\n"


def regime_parameter_table(
    means: pd.DataFrame,
    vols: pd.DataFrame,
    transmat: pd.DataFrame,
    labels: Sequence[str],
) -> str:
    """Table 1: monthly means and volatilities by regime, and the transitions.

    ``means`` and ``vols`` have one row per regime and one column per factor;
    ``transmat`` is regime by regime.
    """
    M = len(labels)
    cells = "c" * (2 * M)
    lines = [
        rf"\begin{{tabular}}{{l{cells}}}",
        r"\toprule",
        rf"\multicolumn{{{2 * M + 1}}}{{c}}"
        r"{Panel A: Regime-Conditional Factor Distribution} \\",
        r"\midrule",
        " & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{label}}}" for label in labels)
        + r" \\",
        *[rf" \cmidrule(r){{{2 * s + 2}-{2 * s + 3}}}" for s in range(M)],
        " & " + " & ".join([r"$\lambda$ & $\sigma$"] * M) + r" \\",
        r"\midrule",
    ]
    for f in means.columns:
        values = [x for s in range(M) for x in (means.iloc[s][f], vols.iloc[s][f])]
        lines.append(
            f"{FACTOR_LABELS.get(f, f)} & "
            + " & ".join(percent(x) for x in values)
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\bigskip",
        "",
        rf"\begin{{tabular}}[t]{{l{'r' * M}}}",
        r"\toprule",
        rf"\multicolumn{{{M + 1}}}{{c}}{{Panel B: Transition Matrix}} \\",
        rf"\midrule & \multicolumn{{{M}}}{{c}}{{To}} \\",
        rf"\cmidrule(r){{2-{M + 1}}}",
        "From & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for s, label in enumerate(labels):
        lines.append(
            f"{label} & " + " & ".join(percent(x) for x in transmat.iloc[s]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def regime_panels(
    means: pd.DataFrame,
    vols: pd.DataFrame,
    transmat: pd.DataFrame,
    labels: Sequence[str],
) -> tuple[str, str, str]:
    """The regime model in three stacked panels, as in the Online Appendix."""

    def factor_rows(frame):
        return [
            (
                FACTOR_LABELS.get(f, f),
                [percent(frame.iloc[s][f]) for s in range(len(labels))],
            )
            for f in frame.columns
        ]

    header = ["", *labels]
    heavy = r"\midrule[\heavyrulewidth]"
    return (
        _table(
            header,
            [(None, factor_rows(means))],
            title="Panel A: Monthly means",
            last=heavy,
        ),
        _table(
            header,
            [(None, factor_rows(vols))],
            title="Panel B: Monthly volatilities",
            top=False,
            last=heavy,
        ),
        _table(
            ["From / to", *labels],
            [
                (
                    None,
                    [
                        (label, [percent(x) for x in transmat.iloc[s]])
                        for s, label in enumerate(labels)
                    ],
                )
            ],
            title="Panel C: Transition probabilities",
            top=False,
        ),
    )


def momentum_table(
    factors: pd.DataFrame,
    posteriors: pd.DataFrame,
    labels: Sequence[str],
    quantile: float = 0.01,
    momentum: str = "Mom",
    market: str = "Mkt-RF",
) -> str:
    """The worst months for momentum and the regime probabilities in them.

    Months in the lowest ``quantile`` of the momentum factor, with the momentum
    and market returns and the smoothed probability of each regime; the last row
    is the unconditional mean.
    """
    worst = factors.index[factors[momentum] <= factors[momentum].quantile(quantile)]
    cols = [FACTOR_LABELS.get(momentum, momentum), FACTOR_LABELS.get(market, market)]
    lines = [
        rf"\begin{{tabular}}{{l{'r' * (2 + len(labels))}}}",
        r"\toprule",
        " & "
        + " & ".join([f"${c}$" for c in cols] + [f"P({label})" for label in labels])
        + r" \\",
        r"\midrule",
    ]

    def row(name, mom, mkt, probs):
        cells = [percent(mom), percent(mkt), *[number(p, 3) for p in probs]]
        return f"{name} & " + " & ".join(cells) + r" \\"

    for month in worst:
        lines.append(
            row(
                pd.Period(month, "M").strftime("%Y-%m"),
                factors.loc[month, momentum],
                factors.loc[month, market],
                posteriors.loc[month],
            )
        )
    lines.append(
        row(
            "Unconditional mean",
            factors[momentum].mean(),
            factors[market].mean(),
            posteriors.mean(),
        )
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        f"% Months in the lowest {100 * quantile:g}% of the momentum factor over the"
        " estimation sample.",
    ]
    return "\n".join(lines) + "\n"


def state_selection_table(selected: pd.DataFrame) -> str:
    """The number of regimes each criterion selects, one row per sample.

    ``selected`` has one row per sample and one column per criterion.
    """
    cols = list(selected.columns)
    lines = [
        rf"\begin{{tabular}}{{l{'r' * len(cols)}}}",
        r"\toprule",
        " & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for name, r in selected.iterrows():
        lines.append(f"{name} & " + " & ".join(str(int(v)) for v in r) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "% Criteria are minimized; the number reported is the minimizing M.",
    ]
    return "\n".join(lines) + "\n"
