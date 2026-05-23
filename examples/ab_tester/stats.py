"""Statistical utilities for A/B testing.

Phase 7 | A/B Testing -- Statistical Rigor
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ABStatResult:
    """Statistical comparison of two groups."""

    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int
    t_statistic: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]
    effect_size: float
    recommended_sample_size: int


def _std(values: list[float], mean: float) -> float:
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def _t_cdf_approx(t: float, df: float) -> float:
    """Approximate the CDF of the t-distribution using a normal approximation.

    For large df this is accurate; for small df it is a reasonable estimate.
    Uses the refined approximation: z = t * (1 - 1/(4*df)) / sqrt(1 + t^2/(2*df))
    then falls through to the standard normal CDF.
    """
    if df <= 0:
        return 0.5
    z = t * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t * t / (2.0 * df))
    return _normal_cdf(z)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(
    scores_a: list[float], scores_b: list[float], alpha: float = 0.05
) -> ABStatResult:
    """Perform Welch's t-test (unequal variance) on two score groups."""
    n_a = len(scores_a)
    n_b = len(scores_b)

    if n_a < 2 or n_b < 2:
        return ABStatResult(
            mean_a=sum(scores_a) / max(n_a, 1) if scores_a else 0.0,
            mean_b=sum(scores_b) / max(n_b, 1) if scores_b else 0.0,
            std_a=0.0,
            std_b=0.0,
            n_a=n_a,
            n_b=n_b,
            t_statistic=0.0,
            p_value=1.0,
            significant=False,
            confidence_interval=(0.0, 0.0),
            effect_size=0.0,
            recommended_sample_size=0,
        )

    mean_a = sum(scores_a) / n_a
    mean_b = sum(scores_b) / n_b
    std_a = _std(scores_a, mean_a)
    std_b = _std(scores_b, mean_b)

    var_a = std_a**2 / n_a
    var_b = std_b**2 / n_b

    # Both groups have zero variance — every sample is identical.
    # Any difference in means is real but the sample is too small/uniform to
    # compute a meaningful t-statistic. Return not-significant rather than a
    # spurious t → ∞ caused by the 1e-9 floor.
    if var_a + var_b == 0:
        return ABStatResult(
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            n_a=n_a,
            n_b=n_b,
            t_statistic=0.0,
            p_value=1.0,
            significant=False,
            confidence_interval=(0.0, 0.0),
            effect_size=0.0,
            recommended_sample_size=0,
        )

    se = math.sqrt(var_a + var_b)
    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    numerator = (var_a + var_b) ** 2
    denominator = (
        (var_a**2 / (n_a - 1) + var_b**2 / (n_b - 1)) if (var_a + var_b) > 0 else 1.0
    )
    df = numerator / denominator if denominator > 0 else 1.0

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _t_cdf_approx(abs(t_stat), df))
    p_value = max(0.0, min(1.0, p_value))

    # 95% confidence interval for (mean_a - mean_b)
    # Use z=1.96 as approximation for the t critical value
    z_crit = 1.96
    ci_lower = (mean_a - mean_b) - z_crit * se
    ci_upper = (mean_a - mean_b) + z_crit * se

    # Cohen's d
    pooled_std = math.sqrt((std_a**2 + std_b**2) / 2) if (std_a + std_b) > 0 else 1e-9
    effect_size = abs(mean_a - mean_b) / pooled_std

    rec_n = minimum_sample_size(effect_size, alpha)

    return ABStatResult(
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        n_a=n_a,
        n_b=n_b,
        t_statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        confidence_interval=(ci_lower, ci_upper),
        effect_size=effect_size,
        recommended_sample_size=rec_n,
    )


def minimum_sample_size(
    effect_size: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Estimate minimum sample size per group for a two-sample t-test.

    Uses the approximation n = (z_alpha + z_beta)^2 * 2 / d^2
    where d is Cohen's d.
    """
    if effect_size <= 0:
        return 0
    z_alpha = _z_from_alpha(alpha)
    z_beta = _z_from_power(power)
    n = ((z_alpha + z_beta) ** 2 * 2) / (effect_size**2)
    return max(2, math.ceil(n))


def _z_from_alpha(alpha: float) -> float:
    """Two-tailed z-value for significance level alpha."""
    return _inverse_normal_cdf(1.0 - alpha / 2.0)


def _z_from_power(power: float) -> float:
    """z-value for statistical power."""
    return _inverse_normal_cdf(power)


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse of the standard normal CDF (rational approximation)."""
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if p < 0.5:
        return -_inverse_normal_cdf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
