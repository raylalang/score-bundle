"""Phase 1 — held-out imputation: does the score graph beat the baselines?

Mask a fraction of notes and predict their expressive variables from the rest.
On clean (near-exact) targets this is the honest test of the graph prior, since the
question is structured prediction, not denoising.

Run:  python examples/phase1_imputation.py
"""
import numpy as np

from score_bundle import metrics
from score_bundle.baselines import independent_field, ridge_impute, temporal_field
from score_bundle.synthetic import make_synthetic, random_mask


def main() -> None:
    rng = np.random.default_rng(7)
    data = make_synthetic(rng, n=150, lam=0.4, eta=3.0, noise_var=0.02)
    mask = random_mask(150, rng, observed_frac=0.6)
    held = ~mask
    yt = data.y_true[held]

    # proposed: full score-graph prior
    g_mean, g_std = data.field.posterior(data.y_obs, data.noise_var, mask=mask)
    # baselines
    i_mean, i_std = independent_field(150, prior_var=float(np.var(data.y_obs[mask]))).posterior(
        data.y_obs, data.noise_var, mask=mask
    )
    t_mean, t_std = temporal_field(data.score, lam=0.4, eta=2.0).posterior(
        data.y_obs, data.noise_var, mask=mask
    )
    r_pred, r_sigma = ridge_impute(data.score, data.y_obs, mask)

    rows = {
        "score-graph (ours)": (g_mean[held], g_std[held]),
        "independent": (i_mean[held], i_std[held]),
        "temporal AR(1)": (t_mean[held], t_std[held]),
        "ridge features": (r_pred[held], np.full(held.sum(), r_sigma)),
    }

    print(f"Phase 1 — held-out imputation ({held.sum()} of 150 notes masked)\n")
    print(f"{'method':22s} {'RMSE':>8s} {'NLL':>8s} {'cov@.9':>8s}")
    for name, (m, s) in rows.items():
        print(f"{name:22s} {metrics.rmse(yt, m):8.4f} {metrics.gaussian_nll(yt, m, s):8.4f} "
              f"{metrics.coverage(yt, m, s, 0.9):8.3f}")


if __name__ == "__main__":
    main()
