"""Phase 1 — synthetic recovery test.

Draw a performance field from a known graph prior, observe it with noise, and check
that the posterior recovers the latents and that its credible intervals are calibrated.

Run:  python examples/phase1_synthetic_recovery.py
"""
import numpy as np

from score_bundle import metrics
from score_bundle.synthetic import make_synthetic


def main() -> None:
    rng = np.random.default_rng(0)
    data = make_synthetic(rng, n=120, lam=0.4, eta=3.0, noise_var=0.03)

    mean, std = data.field.posterior(data.y_obs, data.noise_var)
    results = metrics.evaluate(data.y_true, mean, std, level=0.9)

    print("Phase 1 — synthetic recovery (n=120)")
    for k, v in results.items():
        print(f"  {k:20s} {v:.4f}")


if __name__ == "__main__":
    main()
