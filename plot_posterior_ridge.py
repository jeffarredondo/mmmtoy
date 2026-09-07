"""
Manual version - bypasses arviz.plot_pair entirely since its plotting API
changed between versions and broke on the installed arviz version. This pulls
raw posterior draws out of the InferenceData object directly (just xarray/
numpy) and builds the scatter matrix with plain matplotlib instead - more
verbose, but immune to arviz API churn.
"""
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

idata = az.from_netcdf("dfm_fit.nc")  # adjust path if needed

var_names = ["theta", "k", "s", "gamma"]
truth = {"theta": 0.6, "k": 150.0, "s": 2.0, "gamma": 3.0}

# flatten (chain, draw) into one long vector of samples per parameter
samples = {
    name: idata.posterior[name].values.flatten()
    for name in var_names
}

n = len(var_names)
fig, axes = plt.subplots(n, n, figsize=(11, 11))

for i, row_name in enumerate(var_names):
    for j, col_name in enumerate(var_names):
        ax = axes[i, j]
        if i == j:
            # diagonal: marginal histogram
            ax.hist(samples[row_name], bins=40, color="steelblue", alpha=0.7)
            ax.axvline(truth[row_name], color="red", linestyle="--", linewidth=1.5)
        elif i > j:
            # lower triangle: scatter of posterior draws
            ax.scatter(samples[col_name], samples[row_name],
                       alpha=0.15, s=6, color="steelblue")
            ax.axvline(truth[col_name], color="red", linestyle="--", linewidth=1)
            ax.axhline(truth[row_name], color="red", linestyle="--", linewidth=1)
        else:
            # upper triangle: leave blank (redundant with lower triangle)
            ax.axis("off")

        if i == n - 1:
            ax.set_xlabel(col_name)
        if j == 0:
            ax.set_ylabel(row_name)

fig.suptitle("Joint posterior: theta / k / s / gamma\n(red dashed lines = true values)", y=1.01)
plt.tight_layout()
plt.savefig("posterior_ridge.png", dpi=150, bbox_inches="tight")
print("Saved posterior_ridge.png")
plt.show()