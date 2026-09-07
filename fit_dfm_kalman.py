"""
DFM + adstock/saturation as a proper state-space model, fit via statsmodels'
Kalman filter (MLE) instead of PyMC/NUTS.

Why this is the right move, not just a workaround: f_t is a *linear Gaussian*
latent state given theta/k/s/phi (those are what make it nonlinear - once
they're fixed, the observation and state equations are linear-Gaussian).
That means f_t can be marginalized out ANALYTICALLY via the Kalman filter's
prediction-error decomposition - exactly like your gold model. Sampling all
300 f_t's as free NUTS parameters (what we tried before) is strictly worse:
bigger parameter space, divergence-prone, and throws away a closed-form
marginalization that exists for free.

Only these are estimated: theta, k, s (adstock/saturation),
phi, gamma, q (factor AR(1) + driver), lambda_direct/organic/recall
(loadings, branded_search anchored at 1), and the four idiosyncratic
variances. 13 parameters total, estimated via MLE with asymptotic
standard errors on gamma (the number we actually want).
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

df = pd.read_csv("toy_mmm_data.csv")
with open("toy_mmm_truth.json") as fh:
    truth = json.load(fh)

T = len(df)
spend = df["spend"].values.astype("float64")
endog = df[["branded_search", "direct_traffic", "organic_search", "brand_recall_survey"]].values


def geometric_adstock(x, theta):
    # dtype must follow theta, not x: statsmodels computes standard errors via
    # complex-step differentiation, which perturbs theta with a tiny complex
    # increment. If `out` is forced real (np.empty_like(x) when x is real
    # spend), that imaginary part silently gets discarded on every assignment -
    # which quietly zeroes out theta's derivative and broke its std err above.
    out = np.empty(len(x), dtype=np.result_type(x, theta))
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + theta * out[t - 1]
    return out


def hill_saturation_stable(x, k, s):
    # np.clip doesn't handle complex values well (needed since adstocked can
    # be complex during complex-step differentiation w.r.t. theta) - floor
    # using the real part only, preserving any imaginary part on the rest.
    x_safe = np.where(x.real < 1e-6, 1e-6 + 0j if np.iscomplexobj(x) else 1e-6, x)
    # sigmoid(s * (log x - log k)) -- same stable form as the PyMC version
    return 1.0 / (1.0 + np.exp(-(s * (np.log(x_safe) - np.log(k)))))


class DFMAdstockSaturation(MLEModel):
    """
    Observation: y_t (4,) = Z * f_t + eps_t,  eps_t ~ N(0, H)   H diagonal
    State:       f_t = phi * f_{t-1} + gamma * saturated_spend_t + eta_t, eta_t ~ N(0, q)
    """

    param_names = [
        "theta", "k", "s", "phi", "gamma", "q",
        "lam_direct", "lam_organic", "lam_recall",
        "r_branded", "r_direct", "r_organic", "r_recall",
    ]

    def __init__(self, endog, spend):
        super().__init__(endog, k_states=1, k_posdef=1, initialization="stationary")
        self.spend = spend
        self.ssm["design"] = np.zeros((4, 1, 1))
        self.ssm["obs_cov"] = np.zeros((4, 4, 1))
        self.ssm["transition"] = np.zeros((1, 1, 1))
        self.ssm["selection"] = np.ones((1, 1, 1))
        self.ssm["state_cov"] = np.zeros((1, 1, 1))
        # state intercept is time-varying (driven by adstocked/saturated spend)
        self.ssm["state_intercept"] = np.zeros((1, self.nobs))

    @property
    def start_params(self):
        return np.array([
            0.5,   # theta
            80.0,  # k
            2.0,   # s
            0.7,   # phi
            1.0,   # gamma
            0.1,   # q
            1.0, 1.0, 1.0,      # loadings
            1.0, 1.0, 1.0, 1.0,  # obs variances
        ])

    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        p[0] = 1 / (1 + np.exp(-unconstrained[0]))   # theta in (0,1)
        p[1] = np.exp(unconstrained[1])               # k > 0
        p[2] = np.exp(unconstrained[2])               # s > 0
        p[3] = 1 / (1 + np.exp(-unconstrained[3]))   # phi in (0,1)
        # gamma (index 4) unconstrained, real line
        p[5] = np.exp(unconstrained[5])               # q > 0
        # loadings (6,7,8) unconstrained, real line
        p[9:13] = np.exp(unconstrained[9:13])          # variances > 0
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[0] = np.log(constrained[0] / (1 - constrained[0]))
        p[1] = np.log(constrained[1])
        p[2] = np.log(constrained[2])
        p[3] = np.log(constrained[3] / (1 - constrained[3]))
        p[5] = np.log(constrained[5])
        p[9:13] = np.log(constrained[9:13])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        theta, k, s, phi, gamma, q = params[:6]
        lam_direct, lam_organic, lam_recall = params[6:9]
        r_branded, r_direct, r_organic, r_recall = params[9:13]

        adstocked = geometric_adstock(self.spend, theta)
        saturated = hill_saturation_stable(adstocked, k, s)

        self.ssm["design", :, 0, 0] = [1.0, lam_direct, lam_organic, lam_recall]
        self.ssm["obs_cov", :, :, 0] = np.diag([r_branded, r_direct, r_organic, r_recall])
        self.ssm["transition", 0, 0, 0] = phi
        self.ssm["state_cov", 0, 0, 0] = q
        # drive[0] unused (matches the toy generator: f_0 has no spend contribution)
        drive = gamma * saturated
        drive[0] = 0.0
        self.ssm["state_intercept"] = drive.reshape(1, -1)


mod = DFMAdstockSaturation(endog, spend)
res = mod.fit(disp=False, maxiter=2000)

print(res.summary())

print("\n--- Ground truth for comparison ---")
print(f"theta={truth['theta']}, k={truth['k']}, s={truth['s']}, "
      f"phi={truth['phi']}, gamma={truth['gamma']}, q={truth['q']}")
print(f"lambda: direct={truth['lambda']['direct_traffic']}, "
      f"organic={truth['lambda']['organic_search']}, "
      f"recall={truth['lambda']['brand_recall_survey']}")

# Confidence interval specifically on gamma - the number that answers
# "does spend move brand momentum"
ci = res.conf_int()
gamma_idx = mod.param_names.index("gamma")
print(f"\ngamma point estimate: {res.params[gamma_idx]:.3f}")
print(f"gamma 95% CI: [{ci[gamma_idx][0]:.3f}, {ci[gamma_idx][1]:.3f}]")
print(f"true gamma was: {truth['gamma']}")

res.save("dfm_mle_fit.pickle")
