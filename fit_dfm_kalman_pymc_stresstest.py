"""
Same DFM + adstock/saturation spec as fit_dfm_kalman.py, but wired into
PyMC via pymc_extras.statespace.core.PyMCStateSpace instead of statsmodels.

Why this is the "best of both" version:
  - Like fit_dfm_kalman.py: the Kalman filter marginalizes f_t analytically.
    NUTS only ever samples the 13 hyperparameters (theta, k, s, phi, gamma,
    q, 3 loadings, 4 variances) - no 300-dimensional latent state to sample,
    unlike dfm_pymc_attempt_slow_reference.py.
  - Like dfm_pymc_attempt_slow_reference.py: real PyMC priors instead of
    MLE point estimates, so identifiability pressure (e.g. keeping theta
    away from a degenerate 0) comes from the prior, not just a boundary
    constraint, and you get a proper posterior instead of an asymptotic CI.

Only the adstock transform needs the closed-form "T x T decay matrix" trick
(same one from the earlier scan-avoidance fix) - the factor's own AR(1)
recursion is handled natively by the Kalman filter itself, since that's
exactly the recursion Kalman filters are built for. No decay-matrix trick
needed for phi at all.
"""
import json
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.statespace.core import PyMCStateSpace

df = pd.read_csv("toy_mmm_data.csv")
with open("toy_mmm_truth.json") as fh:
    truth = json.load(fh)

T = len(df)
spend = df["spend"].values.astype("float64")
endog = df[["branded_search", "direct_traffic", "organic_search", "brand_recall_survey"]].values

# precompute the adstock decay-matrix ingredients (fixed, not parameters)
t_idx = np.arange(T)
exponent = t_idx[:, None] - t_idx[None, :]
mask = (exponent >= 0).astype("float64")
exponent_clipped = np.clip(exponent, 0, None).astype("float64")
spend_const = pt.constant(spend, name="spend_const")
mask_const = pt.constant(mask, name="mask_const")
exponent_const = pt.constant(exponent_clipped, name="exponent_const")
zero_first_const = pt.constant(
    np.concatenate([[0.0], np.ones(T - 1)]).reshape(T, 1), name="zero_first_const"
)


class DFMAdstockSaturationPyMC(PyMCStateSpace):
    """
    Observation: y_t (4,) = Z f_t + eps_t,  eps_t ~ N(0, H)  H diagonal
    State:       f_t = phi f_{t-1} + c_t + eta_t,  eta_t ~ N(0, q)
                 c_t = gamma * saturated(adstock(spend_t; theta); k, s), c_0 = 0
    """

    def __init__(self):
        super().__init__(k_endog=4, k_states=1, k_posdef=1, measurement_error=True)
        # fixed (non-parameterized) matrices
        self.ssm["selection", :, :] = np.ones((1, 1))
        self.ssm["initial_state", :] = np.zeros(1)
        self.ssm["initial_state_cov", :, :] = np.array([[5.0]])  # diffuse-ish

    @property
    def param_names(self):
        return [
            "theta", "k", "s", "phi", "gamma", "q",
            "lam_direct", "lam_organic", "lam_recall",
            "r_branded", "r_direct", "r_organic", "r_recall",
        ]

    def make_symbolic_graph(self):
        theta = self.make_and_register_variable("theta", shape=())
        k = self.make_and_register_variable("k", shape=())
        s = self.make_and_register_variable("s", shape=())
        phi = self.make_and_register_variable("phi", shape=())
        gamma = self.make_and_register_variable("gamma", shape=())
        q = self.make_and_register_variable("q", shape=())
        lam_direct = self.make_and_register_variable("lam_direct", shape=())
        lam_organic = self.make_and_register_variable("lam_organic", shape=())
        lam_recall = self.make_and_register_variable("lam_recall", shape=())
        r_branded = self.make_and_register_variable("r_branded", shape=())
        r_direct = self.make_and_register_variable("r_direct", shape=())
        r_organic = self.make_and_register_variable("r_organic", shape=())
        r_recall = self.make_and_register_variable("r_recall", shape=())

        # --- design (Z): branded_search is the anchor, loading fixed to 1 ---
        self.ssm["design", 0, 0] = 1.0
        self.ssm["design", 1, 0] = lam_direct
        self.ssm["design", 2, 0] = lam_organic
        self.ssm["design", 3, 0] = lam_recall

        # --- obs_cov (H): diagonal idiosyncratic variances ---
        self.ssm["obs_cov", 0, 0] = r_branded
        self.ssm["obs_cov", 1, 1] = r_direct
        self.ssm["obs_cov", 2, 2] = r_organic
        self.ssm["obs_cov", 3, 3] = r_recall

        # --- transition (T) and state_cov (Q): the factor's own AR(1) ---
        # this is the ONLY recursion the Kalman filter needs to handle -
        # no decay-matrix trick required, it's what Kalman filters do natively.
        self.ssm["transition", 0, 0] = phi
        self.ssm["state_cov", 0, 0] = q

        # --- adstock + saturation, closed-form (T x T decay matrix) ---
        # theta^(t-i) for i<=t, else 0 - same trick as the scan-avoidance fix
        ThetaMat = mask_const * (theta ** exponent_const)
        adstocked = pt.dot(ThetaMat, spend_const)
        adstocked_safe = pt.switch(pt.lt(adstocked, 1e-6), 1e-6, adstocked)
        saturated = pm.math.sigmoid(s * (pt.log(adstocked_safe) - pt.log(k)))

        # --- state_intercept (c): gamma * saturated spend drives the factor ---
        # this is the piece that's genuinely time-varying, and the reason we
        # need declare_time_varying here.
        drive = (gamma * saturated).reshape((T, 1)) * zero_first_const  # c_0 = 0
        self.ssm.declare_time_varying("state_intercept")
        self.ssm["state_intercept"] = drive


mod = DFMAdstockSaturationPyMC()

with pm.Model() as pymc_mod:
    # STRESS TEST: priors deliberately tight and centered AT the true values.
    # If the data actually carried information about theta/k/s/gamma, the
    # posterior should shrink noticeably tighter than these priors and/or
    # shift if the prior mean were wrong. If instead the posterior comes
    # back looking almost identical to the prior, that's direct proof the
    # likelihood is nearly flat in these directions - weak identification,
    # not a modeling bug, not bad luck.
    theta = pm.Normal("theta", mu=0.6, sigma=0.03)  # true theta = 0.6
    k = pm.Normal("k", mu=150.0, sigma=5.0)           # true k = 150
    s = pm.Normal("s", mu=2.0, sigma=0.1)             # true s = 2.0
    phi = pm.Beta("phi", alpha=5, beta=2)
    gamma = pm.Normal("gamma", mu=3.0, sigma=0.15)    # true gamma = 3.0
    q = pm.HalfNormal("q", sigma=0.5)
    lam_direct = pm.Normal("lam_direct", mu=1, sigma=0.5)
    lam_organic = pm.Normal("lam_organic", mu=1, sigma=0.5)
    lam_recall = pm.Normal("lam_recall", mu=1, sigma=0.5)
    r_branded = pm.HalfNormal("r_branded", sigma=1)
    r_direct = pm.HalfNormal("r_direct", sigma=1)
    r_organic = pm.HalfNormal("r_organic", sigma=1)
    r_recall = pm.HalfNormal("r_recall", sigma=1)

    mod.build_statespace_graph(data=endog)

    idata = pm.sample(draws=500, tune=500, chains=4, cores=4, target_accept=0.9, random_seed=42)

import arviz as az
summary = az.summary(idata, var_names=mod.param_names)
print(summary)

print("\n--- Ground truth for comparison ---")
print(f"theta={truth['theta']}, k={truth['k']}, s={truth['s']}, "
      f"phi={truth['phi']}, gamma={truth['gamma']}, q={truth['q']}")
print(f"lambda: direct={truth['lambda']['direct_traffic']}, "
      f"organic={truth['lambda']['organic_search']}, "
      f"recall={truth['lambda']['brand_recall_survey']}")

idata.to_netcdf("dfm_kalman_pymc_stresstest_fit.nc")
summary.to_csv("dfm_kalman_pymc_stresstest_summary.csv")
print("\nDone.")