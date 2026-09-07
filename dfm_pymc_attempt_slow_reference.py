"""
Same model as fit_dfm_model.py, but both recursions (adstock, factor AR(1))
are expressed as T x T "decay matrices" instead of pytensor.scan.

Why this is equivalent: geometric adstock a_t = spend_t + theta*a_{t-1}
unrolls to a_t = sum_{i<=t} theta^(t-i) * spend_i - a lower-triangular
matrix-vector product. Same trick for f_t = phi*f_{t-1} + drive_t.
This avoids backprop through a 300-step scan, which is what was killing
compile time on a single core.
"""
import json
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

df = pd.read_csv("toy_mmm_data.csv")
with open("toy_mmm_truth.json") as fh:
    truth = json.load(fh)

T = len(df)
spend = df["spend"].values.astype("float64")
branded_search = df["branded_search"].values
direct_traffic = df["direct_traffic"].values
organic_search = df["organic_search"].values
brand_recall = df["brand_recall_survey"].values

# precompute the (t - i) exponent grid once, shared by both recursions
t_idx = np.arange(T)
exponent = t_idx[:, None] - t_idx[None, :]          # exponent[t,i] = t - i
mask = (exponent >= 0).astype("float64")            # lower-triangular incl diag
exponent_clipped = np.clip(exponent, 0, None).astype("float64")


def decay_matrix(param, exponent_clipped_, mask_):
    """T x T matrix where entry (t,i) = param^(t-i) for i<=t, else 0."""
    return mask_ * (param ** exponent_clipped_)


with pm.Model() as model:

    spend_data = pm.Data("spend_data", spend)

    # ---- adstock + saturation ----
    theta = pm.Beta("theta", alpha=2, beta=2)
    k = pm.HalfNormal("k", sigma=150)
    s = pm.Gamma("s", alpha=2, beta=1)  # weakly informative, mean 2

    ThetaMat = decay_matrix(theta, exponent_clipped, mask)      # T x T
    adstocked = pt.dot(ThetaMat, spend_data)                    # T

    # Hill saturation in log-space for numerical stability:
    # adstocked^s / (adstocked^s + k^s) == sigmoid(s * (log(adstocked) - log(k)))
    # Computing adstocked**s directly overflows for large s / adstocked - this
    # is exactly the kind of thing that silently wrecks NUTS (looks like
    # divergences / max-treedepth trees, not an obvious crash).
    adstocked_safe = pt.switch(pt.lt(adstocked, 1e-6), 1e-6, adstocked)
    saturated = pm.math.sigmoid(s * (pt.log(adstocked_safe) - pt.log(k)))

    # ---- factor AR(1) driven by saturated spend ----
    phi = pm.Beta("phi", alpha=5, beta=2)
    gamma = pm.Normal("gamma", mu=0, sigma=5)
    q_sd = pm.HalfNormal("q_sd", sigma=0.5)
    f0 = pm.Normal("f0", mu=0, sigma=1)

    innov = pm.Normal("innov", mu=0, sigma=1, shape=T)
    drive = gamma * saturated + q_sd * innov
    zero_first = np.concatenate([[0.0], np.ones(T - 1)])  # drive[0] unused, matches generator
    drive = drive * zero_first

    PhiMat = decay_matrix(phi, exponent_clipped, mask)          # T x T
    phi_powers = phi ** t_idx.astype("float64")                 # phi^t, T

    f = pt.dot(PhiMat, drive) + phi_powers * f0
    f = pm.Deterministic("f", f)

    # ---- observation equation (branded_search is the anchor, loading=1) ----
    lam_direct = pm.Normal("lam_direct_traffic", mu=1, sigma=0.5)
    lam_organic = pm.Normal("lam_organic_search", mu=1, sigma=0.5)
    lam_recall = pm.Normal("lam_brand_recall", mu=1, sigma=0.5)

    r_branded = pm.HalfNormal("r_branded_search", sigma=1)
    r_direct = pm.HalfNormal("r_direct_traffic", sigma=1)
    r_organic = pm.HalfNormal("r_organic_search", sigma=1)
    r_recall = pm.HalfNormal("r_brand_recall", sigma=1)

    pm.Normal("obs_branded", mu=f, sigma=r_branded, observed=branded_search)
    pm.Normal("obs_direct", mu=lam_direct * f, sigma=r_direct, observed=direct_traffic)
    pm.Normal("obs_organic", mu=lam_organic * f, sigma=r_organic, observed=organic_search)
    pm.Normal("obs_recall", mu=lam_recall * f, sigma=r_recall, observed=brand_recall)

    idata = pm.sample(
        draws=300, tune=300, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        progressbar=True,
    )

idata.to_netcdf("dfm_fit.nc")

summary = az.summary(idata, var_names=[
    "theta", "k", "s", "phi", "gamma", "q_sd",
    "lam_direct_traffic", "lam_organic_search", "lam_brand_recall",
    "r_branded_search", "r_direct_traffic", "r_organic_search", "r_brand_recall",
])
print(summary)

print("\n--- Ground truth for comparison ---")
print(f"theta={truth['theta']}, k={truth['k']}, s={truth['s']}, "
      f"phi={truth['phi']}, gamma={truth['gamma']}, q={np.sqrt(truth['q']):.3f}")
print(f"lambda: direct={truth['lambda']['direct_traffic']}, "
      f"organic={truth['lambda']['organic_search']}, "
      f"recall={truth['lambda']['brand_recall_survey']}")

summary.to_csv("dfm_fit_summary.csv")
print("\nDone.")
