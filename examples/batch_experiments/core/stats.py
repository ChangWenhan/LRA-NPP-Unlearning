import numpy as np
try:
    from scipy import stats
except Exception:
    stats = None


def cliffs_delta_against_zero(diffs):
    diffs = np.asarray(diffs)
    gt = np.sum(diffs > 0)
    lt = np.sum(diffs < 0)
    n = len(diffs)
    if n == 0:
        return 0.0
    return float((gt - lt) / n)


def cohens_dz(diffs):
    diffs = np.asarray(diffs)
    if len(diffs) < 2:
        return 0.0
    sd = np.std(diffs, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diffs) / sd)


def bootstrap_ci_mean(diffs, n_boot=5000, alpha=0.05, seed=42):
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        samples.append(np.mean(diffs[idx]))
    lo = np.quantile(samples, alpha / 2)
    hi = np.quantile(samples, 1 - alpha / 2)
    return float(lo), float(hi)


def paired_tests(lhs, rhs):
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if len(lhs) != len(rhs):
        raise ValueError("paired tests require equal length arrays")

    diffs = lhs - rhs
    mean_diff = float(np.mean(diffs)) if len(diffs) else 0.0
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    if len(diffs) < 2:
        p_t = 1.0
        p_w = 1.0
    else:
        if stats is None:
            p_t = 1.0
            p_w = 1.0
        else:
            t_res = stats.ttest_rel(lhs, rhs)
            p_t = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0
            if np.allclose(diffs, 0):
                p_w = 1.0
            else:
                try:
                    w_res = stats.wilcoxon(lhs, rhs, zero_method="wilcox", alternative="two-sided")
                    p_w = float(w_res.pvalue)
                except ValueError:
                    p_w = 1.0

    ci_low, ci_high = bootstrap_ci_mean(diffs)
    delta = cliffs_delta_against_zero(diffs)
    dz = cohens_dz(diffs)

    return {
        "n_runs": int(len(lhs)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_paired_t": p_t,
        "p_wilcoxon": p_w,
        "cliffs_delta": delta,
        "cohens_dz": dz,
    }


def format_p_value(p):
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.4g}"
