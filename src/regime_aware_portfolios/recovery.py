"""Recovery of known loadings by RWLS on simulated data (Figure 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rwls

__all__ = ["recover_path"]


def recover_path(config: str | Path, path: int) -> tuple[np.ndarray, np.ndarray]:
    """True and estimated loadings on the in-sample months of one simulated path.

    The regime model is fitted to the path's factor returns as at the framework's
    first decision date, and RWLS to its stock returns. Both arrays have shape
    (stocks, factors, regimes); the estimated regimes are put in the order of the
    true ones, matched by their factor means.
    """
    from regime_aware_portfolios.backtest import _study
    from regime_aware_portfolios.calibration import estimate_stocks
    from regime_aware_portfolios.evaluation import match_regimes
    from regime_aware_portfolios.simulation import simulate_path

    cfg, dgp = _study(str(config))
    sim = cfg.simulation
    data = simulate_path(dgp, sim.n_in_sample, sim.n_stocks, cfg.seeds.simulation, path)
    settings = rwls.RegimeSettings(n_regimes=dgp.n_regimes, seed=cfg.seeds.regimes)
    fit = rwls.RegimeModel(settings).fit(data.factors)
    est = estimate_stocks(
        data.returns, data.factors, fit.posteriors, cfg.rwls_settings()
    )
    # Loadings by factor, then regime -> (stocks, factors, regimes), true order.
    K, M = len(dgp.factor_names), dgp.n_regimes
    estimated = est.loadings.to_numpy().reshape(-1, K, M)
    aligned = np.empty_like(estimated)
    aligned[:, :, match_regimes(fit.means.to_numpy(), dgp.factor_means)] = estimated
    return data.loadings, aligned
