"""
Toy data generator for the DFM + adstock/saturation MMM spec:

    Observation:  y_t = Lambda * f_t + eps_t          eps_t ~ N(0, R)  (diagonal)
    State:        f_t = phi * f_{t-1} + gamma * adstock(spend)_t + eta_t   eta_t ~ N(0, q)

We pick TRUE values for everything so you can check whether the model
recovers them. This is the single most useful thing you can do before
trusting any fit on real data.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

T = 300  # days

# ---- 1. Simulate raw ad spend -------------------------------------------------
# Campaigns: bursty, autocorrelated spend (realistic - not iid noise, since
# that autocorrelation is exactly what makes adstock/saturation weakly
# identified in practice)
spend = np.zeros(T)
campaign_level = 0.0
for t in range(T):
    # campaign intensity itself is a slow AR(1) process, plus occasional bursts
    campaign_level = 0.9 * campaign_level + rng.normal(0, 8)
    burst = 60.0 if rng.random() < 0.05 else 0.0  # occasional big push
    spend[t] = max(0.0, 40 + campaign_level + burst)

# ---- 2. Adstock transform (geometric) -----------------------------------------
THETA_TRUE = 0.6  # decay/retention rate

def geometric_adstock(x, theta):
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + theta * out[t - 1]
    return out

adstocked_spend = geometric_adstock(spend, THETA_TRUE)

# ---- 3. Saturation transform (Hill function) ----------------------------------
K_TRUE = 150.0   # half-saturation point
S_TRUE = 2.0     # steepness

def hill_saturation(x, k, s):
    return (x ** s) / (x ** s + k ** s)

saturated_spend = hill_saturation(adstocked_spend, K_TRUE, S_TRUE)  # in [0,1]

# ---- 4. Latent brand-momentum factor -------------------------------------------
PHI_TRUE = 0.85     # factor's own persistence
GAMMA_TRUE = 3.0    # <-- the number you actually want: effect of saturated,
                    #     adstocked spend on brand momentum
Q_TRUE = 0.05       # state innovation variance

f = np.zeros(T)
f[0] = rng.normal(0, 1)
for t in range(1, T):
    f[t] = PHI_TRUE * f[t - 1] + GAMMA_TRUE * saturated_spend[t] + rng.normal(0, np.sqrt(Q_TRUE))

# ---- 5. Observed proxy series (loadings + idiosyncratic noise) ----------------
# Anchor series (branded search) fixed at loading = 1 for identification.
LAMBDA_TRUE = {
    "branded_search":  1.0,   # anchor
    "direct_traffic":  0.7,
    "organic_search":  1.3,
    "brand_recall_survey": 0.5,  # sparser / noisier proxy
}
R_TRUE = {
    "branded_search":  0.3,
    "direct_traffic":  0.5,
    "organic_search":  0.4,
    "brand_recall_survey": 0.8,
}

data = {"date": pd.date_range("2024-01-01", periods=T, freq="D"), "spend": spend}
for name, lam in LAMBDA_TRUE.items():
    noise = rng.normal(0, np.sqrt(R_TRUE[name]), size=T)
    data[name] = lam * f + noise

df = pd.DataFrame(data)

# Stash ground truth alongside for later comparison - not something you'd
# have in real life, but essential for sanity-checking the model.
truth = {
    "theta": THETA_TRUE, "k": K_TRUE, "s": S_TRUE,
    "phi": PHI_TRUE, "gamma": GAMMA_TRUE, "q": Q_TRUE,
    "lambda": LAMBDA_TRUE, "r": R_TRUE,
}

df.to_csv("/home/claude/toy_mmm_data.csv", index=False)
np.save("/home/claude/toy_mmm_latent_factor.npy", f)  # true f_t, for later comparison
import json
with open("/home/claude/toy_mmm_truth.json", "w") as fh:
    json.dump(truth, fh, indent=2)

print(df.head(10))
print("\nTrue params:", json.dumps(truth, indent=2))
