from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ComparisonResult:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff_mean: float
    ci_low: float
    ci_high: float
    p_value: float


def compare_two_compounds(
    df: pd.DataFrame,
    group_a: str = "SOFT",
    group_b: str = "MEDIUM",
    metric_col: str = "lap_time_sec",
    n_boot: int = 3000,
    n_perm: int = 3000,
    seed: int = 42,
) -> ComparisonResult | None:
    """Compare two tyre compounds using bootstrap CI and two-sided permutation test."""
    a = df.loc[df["Compound"] == group_a, metric_col].dropna().to_numpy(dtype=float)
    b = df.loc[df["Compound"] == group_b, metric_col].dropna().to_numpy(dtype=float)

    if len(a) < 5 or len(b) < 5:
        return None

    rng = np.random.default_rng(seed)
    obs_diff = float(np.mean(a) - np.mean(b))

    # Bootstrap 95% güven aralığı
    boot_diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        boot_diffs[i] = float(np.mean(sample_a) - np.mean(sample_b))

    ci_low, ci_high = [float(x) for x in np.percentile(boot_diffs, [2.5, 97.5])]

    # Permütasyon testi (H0: iki hamur arasında ortalama tur süresi farkı yoktur)
    pooled = np.concatenate([a, b])
    exceed_count = 0
    abs_obs = abs(obs_diff)

    for _ in range(n_perm):
        permuted = rng.permutation(pooled)
        perm_a = permuted[: len(a)]
        perm_b = permuted[len(a) :]
        if abs(float(np.mean(perm_a) - np.mean(perm_b))) >= abs_obs:
            exceed_count += 1

    p_value = float((exceed_count + 1) / (n_perm + 1))

    return ComparisonResult(
        group_a=group_a,
        group_b=group_b,
        n_a=len(a),
        n_b=len(b),
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        diff_mean=obs_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )