import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest
import rwls

from regime_aware_portfolios import load_config, paths
from regime_aware_portfolios.simulation import DataGeneratingProcess, simulate_path
from regime_aware_portfolios.strategies import (
    Strategy,
    _factor_moments,
    decision_dates,
    decisions,
    portfolio,
    regime_weights,
    study_strategies,
)

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
NAMES = ["Mkt-RF", "SMB"]


def _process():
    """Two factors, two regimes and simple stock parameters."""
    trans = np.array([[0.9, 0.1], [0.2, 0.8]])
    return DataGeneratingProcess(
        factor_names=NAMES,
        initial=np.array([2 / 3, 1 / 3]),
        transmat=trans,
        factor_means=np.array([[0.01, 0.002], [-0.02, 0.004]]),
        factor_vols=np.array([[0.04, 0.02], [0.08, 0.03]]),
        loadings_mean=np.array([1.0, 1.2, 0.3, 0.5]),
        loadings_factor=0.2 * np.eye(4),
        resid_intercept=np.array([-5.5, -5.0]),
        resid_coef=np.zeros((2, 4)),
        resid_factor=0.3 * np.eye(2),
    )


@pytest.fixture(scope="module")
def cfg():
    base = load_config(paths.CONFIG / "main.toml")
    return dataclasses.replace(
        base,
        regimes=dataclasses.replace(base.regimes, n_regimes=2),
        simulation=dataclasses.replace(
            base.simulation, n_in_sample=150, n_out_of_sample=4, n_stocks=15
        ),
    )


@pytest.fixture(scope="module")
def path(cfg):
    sim = cfg.simulation
    return simulate_path(_process(), sim.n_periods, sim.n_stocks, seed=5, path=0)


def test_decision_dates(cfg):
    assert list(decision_dates(cfg)) == [149, 150, 151, 152]


def test_each_costa_kwon_step_changes_one_choice():
    s = study_strategies(load_config(paths.CONFIG / "main.toml"))
    fields = [f.name for f in dataclasses.fields(Strategy) if f.name != "name"]

    def changed(a, b):
        return {f for f in fields if getattr(s[a], f) != getattr(s[b], f)}

    assert changed("costa_kwon", "path_regimes") == {"n_regimes"}
    assert changed("path_regimes", "path_factors") == {"regime_factors"}
    assert changed("path_factors", "path_forecast") == {"forecast"}
    assert changed("path_forecast", "path_covariance") == {"factor_cov"}
    assert changed("path_covariance", "framework_mvo") == {"weighting", "window"}
    assert changed("framework_mvo", "framework") == {"objective"}
    assert changed("framework", "oracle") == {"oracle"}
    ck = s["costa_kwon"]
    assert (ck.n_regimes, ck.regime_factors, ck.window) == (2, ("Mkt-RF",), 24)
    assert (ck.forecast, ck.factor_cov, ck.weighting) == ("current_row", "full", "hard")
    assert {"rolling_24", "rolling_36", "rolling_60"} <= set(s)


@pytest.mark.parametrize("name, window", [("agnostic", None), ("rolling_36", 36)])
def test_agnostic_expected_returns_are_the_sample_means(cfg, path, name, window):
    s = study_strategies(cfg)[name]
    decs = list(decisions(s, path.factors, path.returns, cfg))
    assert [d.date for d in decs] == list(decision_dates(cfg))
    for dec in decs:
        lo = 0 if window is None else dec.date + 1 - window
        Y = path.returns.iloc[lo : dec.date + 1].to_numpy()
        X = path.factors.iloc[lo : dec.date + 1].to_numpy()
        np.testing.assert_allclose(dec.moments.means[0], Y.mean(axis=0), atol=1e-12)
        # Covariance: loadings times the sample factor covariance, plus residuals.
        D = np.column_stack([np.ones(len(X)), X])
        coefs, *_ = np.linalg.lstsq(D, Y, rcond=None)
        resid = Y - D @ coefs
        expected = coefs[1:].T @ np.cov(X, rowvar=False) @ coefs[1:] + np.diag(
            (resid**2).sum(axis=0) / (len(X) - 3)
        )
        np.testing.assert_allclose(dec.moments.covariance(0), expected, atol=1e-12)


@pytest.mark.parametrize("weighting", ["posterior", "hard"])
def test_regime_weights_are_those_of_rwls(path, weighting):
    rng = np.random.default_rng(0)
    T = 150
    post = rng.dirichlet([1.0, 1.0], T)
    post[:140] = [[0.95, 0.05]] * 140  # regime 2 is rare: its fallback is used
    settings = rwls.RWLSSettings(weighting=weighting, hard_window=24)
    X = path.factors.iloc[:T]
    res = rwls.RWLS(
        path.returns.iloc[:T], X, pd.DataFrame(post, index=X.index), settings
    ).fit()
    weights = regime_weights(post, settings, n_coefficients=3)
    means, covs = _factor_moments(X.to_numpy(), weights, full=False)
    np.testing.assert_allclose(means, res.factor_means.to_numpy(), atol=1e-14)
    np.testing.assert_allclose(covs, res.factor_covariances, atol=1e-14)
    np.testing.assert_array_equal(
        [w.sum() > 3 + 1 for w in weights], True
    )  # every regime ends above the threshold


def test_full_factor_covariance_keeps_the_diagonal(path):
    X = path.factors.to_numpy()
    w = [np.ones(len(X))]
    _, diag = _factor_moments(X, w, full=False)
    _, full = _factor_moments(X, w, full=True)
    np.testing.assert_allclose(np.diag(full[0]), np.diag(diag[0]))
    assert abs(full[0][0, 1]) > 0
    np.testing.assert_allclose(full[0], np.cov(X, rowvar=False, ddof=0))


def test_costa_kwon_uses_the_transition_row_of_the_current_regime(cfg, path):
    s = study_strategies(cfg)["costa_kwon"]
    first = next(iter(decisions(s, path.factors, path.returns, cfg)))
    fit = rwls.RegimeModel(n_regimes=2, seed=cfg.seeds.regimes).fit(
        path.factors.iloc[: first.date + 1][["Mkt-RF"]]
    )
    row = fit.transmat.to_numpy()[np.argmax(fit.posteriors.to_numpy()[-1])]
    np.testing.assert_allclose(first.moments.probs, row / row.sum(), atol=1e-10)


def test_framework_follows_rwls_expanding(cfg, path):
    s = study_strategies(cfg)["framework"]
    decs = list(decisions(s, path.factors, path.returns, cfg))
    steps = list(
        rwls.expanding(
            path.returns,
            path.factors,
            start=150,
            end=152,
            regime_settings=rwls.RegimeSettings(n_regimes=2, seed=cfg.seeds.regimes),
        )
    )
    for dec, step in zip(decs, steps, strict=True):
        np.testing.assert_allclose(dec.moments.probs, step.forecast.to_numpy())
        m = step.results.moments()
        np.testing.assert_allclose(
            dec.moments.means, [m.mean(r) for r in m.regime_labels]
        )


def test_oracle_given_the_estimated_parameters_matches_the_framework(cfg, path):
    """The two variants differ only in where the regime model comes from."""
    s = study_strategies(cfg)
    est = next(iter(decisions(s["framework"], path.factors, path.returns, cfg)))
    fit = rwls.RegimeModel(n_regimes=2, seed=cfg.seeds.regimes).fit(
        path.factors.iloc[: est.date + 1]
    )
    fitted = dataclasses.replace(
        _process(),
        initial=fit.startprob.to_numpy(),
        transmat=fit.transmat.to_numpy(),
        factor_means=fit.means.to_numpy(),
        factor_vols=np.sqrt(fit.variances.to_numpy()),
    )
    orc = next(iter(decisions(s["oracle"], path.factors, path.returns, cfg, fitted)))
    np.testing.assert_allclose(orc.moments.probs, est.moments.probs, atol=1e-8)
    # Same regressions; only the factor moments differ, so remove their effect.
    np.testing.assert_allclose(orc.moments.idio_var, est.moments.idio_var, rtol=1e-6)


def test_oracle_uses_the_true_model(cfg, path):
    dgp = _process()
    s = study_strategies(cfg)["oracle"]
    dec = next(iter(decisions(s, path.factors, path.returns, cfg, dgp)))
    from hmmlearn.hmm import GaussianHMM

    hmm = GaussianHMM(n_components=2, covariance_type="diag")
    hmm.startprob_, hmm.transmat_ = dgp.initial, dgp.transmat
    hmm.means_, hmm.covars_ = dgp.factor_means, dgp.factor_vols**2
    filtered = hmm.predict_proba(path.factors.iloc[:150].to_numpy())[-1]
    np.testing.assert_allclose(dec.moments.probs, filtered @ dgp.transmat, atol=1e-12)
    with pytest.raises(ValueError, match="data-generating process"):
        next(iter(decisions(s, path.factors, path.returns, cfg)))


def test_portfolios(cfg, path):
    s = study_strategies(cfg)
    ew = next(iter(decisions(s["equal_weight"], path.factors, path.returns, cfg)))
    np.testing.assert_allclose(portfolio(s["equal_weight"], ew, 10).weights, 1 / 15)
    mv = next(iter(decisions(s["minimum_variance"], path.factors, path.returns, cfg)))
    assert mv.returns.shape == (150, 15)
    w = portfolio(s["minimum_variance"], mv, 10).weights
    assert w.sum() == pytest.approx(1, abs=1e-7) and w.min() >= -1e-9
    fw = next(iter(decisions(s["framework"], path.factors, path.returns, cfg)))
    for phi in (1, 50):
        w = portfolio(s["framework"], fw, phi).weights
        assert w.sum() == pytest.approx(1, abs=1e-7)


def test_invalid_strategies():
    with pytest.raises(ValueError, match="objective"):
        Strategy("x", "regimes", objective="median")
    with pytest.raises(ValueError, match="window"):
        Strategy("x", "regimes", weighting="hard")
