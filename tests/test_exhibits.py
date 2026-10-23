import numpy as np
import pandas as pd
import pytest

from regime_aware_portfolios import exhibits
from regime_aware_portfolios.metrics import STATISTICS


def test_formatting():
    assert exhibits.percent(0.1234) == r"12.34\%"
    assert exhibits.percent(-0.0005, 1) == r"-0.1\%"
    assert exhibits.starred(0.25, 0.001) == r"0.250\rlap{$^{***}$}\phantom{$^{***}$}"
    assert exhibits.starred(0.25, 0.07) == r"0.250\rlap{$^{*}$}\phantom{$^{***}$}"
    assert exhibits.starred(0.25, 0.5) == r"0.250\phantom{$^{***}$}"


def test_macros():
    text = exhibits.macros({"oneWayCost": "0.12\\%"})
    assert r"\newcommand{\oneWayCost}{0.12\%}" in text
    with pytest.raises(ValueError, match="letters only"):
        exhibits.macros({"cost2": "1"})


def _stats(strategies, phis=(10.0, 25.0), paths=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, s in enumerate(strategies):
        for phi in phis:
            for p in range(paths):
                row = {c: rng.normal(1.0 + 0.1 * k, 0.1) for c in STATISTICS}
                rows.append({"strategy": s, "phi": phi, "path": p, **row})
    return pd.DataFrame(rows)


def test_performance_table_has_a_row_per_statistic_and_block():
    columns = exhibits.table_columns(36)
    stats = _stats(list(columns))
    text = exhibits.performance_table(stats, columns, [10, 25])
    body = [line for line in text.splitlines() if line.endswith(r"\\")]
    # header, two block titles, twelve statistics per block
    assert len(body) == 1 + 2 * (1 + len(exhibits.PERFORMANCE_ROWS))
    assert r"$\varphi = 25$" in text and r"C\&K" in text
    rows = [line for line in body if line.startswith("Sharpe Ratio")]
    assert len(rows) == 2 and all(r.count(" & ") == len(columns) for r in rows)


def test_decomposition_rows_add_up_to_the_total():
    steps = exhibits.DECOMPOSITION_STEPS
    names = [steps[0][1]] + [b for _, _, b in steps]
    stats = _stats(names, phis=(10.0,))
    text = exhibits.decomposition_table(stats, steps, [10])
    deltas = []
    for line in text.splitlines():
        label = line.split(" & ")[0]
        if label in [s[0] for s in steps] or label == "Total":
            deltas.append(float(line.split(" & ")[1].split("\\")[0]))
    assert sum(deltas[:-1]) == pytest.approx(deltas[-1], abs=5e-3)
    with pytest.raises(ValueError, match="must start where"):
        exhibits.decomposition_table(stats, [steps[0], steps[2]], [10])


def test_regime_parameter_table():
    labels = ["Calm", "Volatile"]
    means = pd.DataFrame({"Mkt-RF": [0.01, -0.02], "Mom": [0.005, -0.01]}, index=[1, 2])
    vols = means.abs() * 3
    trans = pd.DataFrame([[0.9, 0.1], [0.3, 0.7]], index=[1, 2], columns=[1, 2])
    text = exhibits.regime_parameter_table(means, vols, trans, labels)
    assert r"Mkt-Rf & 1.00\% & 3.00\% & -2.00\% & 6.00\% \\" in text
    assert r"Volatile & 30.00\% & 70.00\% \\" in text
    panels = exhibits.regime_panels(means, vols, trans, labels)
    assert "Panel C" in panels[2] and r"UMD & 0.50\% & -1.00\%" in panels[0]
