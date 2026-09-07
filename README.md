# MMM / Dynamic Factor Model exploration — summary

## What this was

A toy exercise in adapting a Bayesian TVP (time-varying-parameter) state-space
approach — originally built for a gold price model — to marketing mix
modeling (MMM). Specifically: can a dynamic factor model (DFM) recover a
latent "brand momentum" factor from several noisy proxy metrics (organic
search, direct traffic, branded queries, brand recall survey), driven by
adstocked + saturated ad spend?

All data is **synthetic, with known ground truth** — built specifically so
model output could be checked against the true parameter values.

## The model

    Observation:  y_t = Lambda * f_t + eps_t        eps_t ~ N(0, R)  (diagonal)
    State:        f_t = phi * f_{t-1} + gamma * saturated(adstock(spend_t)) + eta_t

- **Adstock** (geometric decay, param `theta`): carryover effect of ad
  exposure over time.
- **Saturation** (Hill function, params `k`, `s`): diminishing returns on
  spend.
- **gamma**: the number that actually matters — does adstocked, saturated
  spend move the latent brand-momentum factor.
- **Lambda**: how much each observed proxy loads onto the factor
  (branded_search fixed to 1 as the identification anchor).

## Files (in this folder)

| File | What it is |
|---|---|
| `generate_toy_data.py` | Builds the synthetic dataset with known true parameters (`toy_mmm_truth.json`) |
| `toy_mmm_data.csv` | The synthetic dataset itself | - not commited but can be generated. 
| `fit_dfm_kalman.py` | **Fastest, most reliable.** Fits via a real Kalman filter (statsmodels `MLEModel`), which marginalizes the latent factor `f_t` out analytically. Only 13 hyperparameters ever get estimated. MLE point estimates + asymptotic std errors. |
| `dfm_pymc_attempt_slow_reference.py` | Full Bayesian version that samples every `f_t` explicitly as a latent variable (~313-dimensional posterior) instead of marginalizing. Real priors, correct uncertainty, but expensive — kept mainly as a "what not to do for cost reasons" reference, and as the brute-force cross-check for the Kalman version below. |
| `fit_dfm_kalman_pymc.py` | **The "best of both."** Uses `pymc-extras`'s `PyMCStateSpace` to get Kalman-filter marginalization *and* real PyMC priors together — cheap like the statsmodels version, honest uncertainty like the brute-force version. |
| `fit_dfm_kalman_pymc_stresstest.py` | Same model as above, but with priors deliberately centered tightly on the true values — a diagnostic for identifiability (see below). |
| `plot_posterior_ridge.py` | Plots the joint posterior of theta/k/s/gamma pairwise, to visually confirm the identifiability ridge. |

## The central finding: an identifiability ridge

Across **every version of the model** (MLE, brute-force Bayesian,
Kalman-marginalized Bayesian), the loadings (Lambda) recovered almost
exactly, but **theta, k, s, and gamma did not** — they're tangled together.
Multiple different combinations of "adstock memory / saturation shape /
spend's effect size" fit the observed data almost equally well. This shows
up as:

- MLE landing on a confidently wrong answer (theta ≈ 0, p = 0.91 — i.e.
  "no adstock exists," clearly wrong given the true theta = 0.6), with
  gamma's 95% CI *missing* the true value entirely.
- Bayesian versions giving a wider, more honest gamma interval that *does*
  contain the truth, but still centered off from it.
- The pairwise posterior plot showing genuine diagonal "bands" between
  theta/k/s/gamma — not round, independent clouds.
- **The decisive test:** refitting with priors clamped tightly to the true
  values barely moved the posteriors at all (e.g. gamma's posterior sd
  stayed ~0.13 against a prior sd of 0.15). That means the *data itself*
  carries very little information to distinguish these parameters — a
  strong prior (right or wrong) would produce an equally confident,
  equally clean-looking posterior either way.

**Why this happens:** the synthetic spend series is smooth and
autocorrelated (business-as-usual campaign spend), which structurally
cannot distinguish "slow decay + gentle saturation" from "fast decay +
steep saturation" — no amount of *more* data of the same kind fixes this.

## What would actually fix it (not yet built)

A real incrementality/geo-holdout test result, used to calibrate/anchor the
saturation curve at one known point. This is fundamentally different from a
better prior — it's real information the data doesn't otherwise contain.
This is the natural next step if this gets picked back up.

## Other reusable lessons from this exercise

- **Hill saturation numerical stability:** `x**s / (x**s + k**s)` overflows
  for plausible `s`. Rewrite in log-space as a sigmoid:
  `sigmoid(s * (log(x) - log(k)))`.
- **Avoid `pytensor.scan` for linear recursions** (geometric adstock, AR(1)
  factors) if you're going to sample them directly — differentiating
  through hundreds of sequential steps is what choked the brute-force
  version. A closed-form "decay matrix" (T x T, entry (t,i) = param^(t-i))
  does the same math as one matrix multiply.
- **A real Kalman filter (statsmodels `MLEModel` or `pymc-extras`
  `PyMCStateSpace`) sidesteps this entirely** for the *state's own*
  recursion — the filter's internal recursion handles the AR(1) part
  natively, so the decay-matrix trick is only still needed for adstock,
  which the filter has no built-in concept of.
- **Existing MMM libraries (PyMC-Marketing, Robyn, LightweightMMM) don't
  solve this particular problem** — they're single-target regressions
  (one KPI, several channels), not built for a shared-latent-factor
  structure across multiple noisy proxies. Their adstock/saturation
  transform functions are still reusable building blocks, though.
- **A tight, confident-looking posterior is not by itself evidence of a
  well-estimated parameter.** Always stress-test with a deliberately
  tight, truth-centered prior — if the posterior just mirrors the prior
  back, the data isn't doing the work.