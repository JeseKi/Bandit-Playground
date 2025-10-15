import math
import hashlib
from typing import Tuple, Sequence, List, TYPE_CHECKING
import math as _math

if TYPE_CHECKING:
    from bandit_lib.agents.base import BaseAgent


def determine_layout(metric_count: int) -> Tuple[int, int]:
    """Determine the layout of the subplots (rows, cols) based on the number of metrics"""
    if metric_count <= 0:
        return 1, 1
    cols = 3 if metric_count > 2 else 3
    rows = math.ceil(metric_count / cols)
    return rows, cols


def get_color_from_name(name: str) -> str:
    """Generate a stable hex color from a string name using hash

    Args:
        name: The string to generate color from (e.g., agent name)

    Returns:
        A hex color string like "#a1c3ef"
    """
    hash_obj = hashlib.md5(name.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()

    color_hex = hash_hex[:6]
    return f"#{color_hex}"


def compute_convergence_rate_series(
    steps: Sequence[float], agents: Sequence["BaseAgent"]
) -> List[float]:
    """Compute the convergence rate series (the proportion of agents that have converged within a given time step)"""
    if not agents:
        raise ValueError(
            "`agents` cannot be empty, otherwise the convergence rate cannot be calculated"
        )

    total = len(agents)
    rates: List[float] = []
    for step in steps:
        converged = 0
        for agent in agents:
            convergence_step = getattr(agent, "convergence_step", None)
            if convergence_step is None or convergence_step <= 0:
                continue
            if convergence_step <= step:
                converged += 1
        rates.append(converged / total)
    return rates


def _z_value_for_confidence(confidence: float) -> float:
    """Return the approximate z-value for the confidence level using the normal distribution

    - 0.90 -> 1.6449
    - 0.95 -> 1.96
    - 0.98 -> 2.3263
    - 0.99 -> 2.5758
    other values are approximated to 0.95 using linear approximation
    """
    if confidence >= 0.989:
        return 2.5758
    if confidence >= 0.979:
        return 2.3263
    if confidence >= 0.949:
        return 1.96
    if confidence >= 0.899:
        return 1.6449
    # fallback to 95%
    return 1.96


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color to an rgba string

    Args:
        hex_color: a color value like "#aabbcc" or "aabbcc"
        alpha: transparency [0,1]

    Returns:
        a string like "rgba(r,g,b,alpha)"
    """
    hc = hex_color.lstrip("#")
    if len(hc) != 6:
        return f"rgba(0,0,0,{max(0.0, min(1.0, alpha))})"
    r = int(hc[0:2], 16)
    g = int(hc[2:4], 16)
    b = int(hc[4:6], 16)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a})"


def compute_mean_and_ci(
    series_list: Sequence[Sequence[float]], confidence: float = 0.95
) -> Tuple[List[float], List[float], List[float]]:
    """Compute the mean and confidence interval for a list of time series with the same length

    Args:
        series_list: a list of time series, each series has the same length
        confidence: the confidence level, e.g. 0.95

    Returns:
        (mean, lower, upper), all are lists with the same length as the time series
    """
    if not series_list:
        return [], [], []

    min_len = min(len(s) for s in series_list if s is not None) if series_list else 0
    if min_len <= 0:
        return [], [], []

    z = _z_value_for_confidence(confidence)
    means: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []

    for i in range(min_len):
        vals = [float(s[i]) for s in series_list if s is not None and len(s) > i]
        vals = [v for v in vals if not _math.isnan(v)]
        n = len(vals)
        if n == 0:
            means.append(float("nan"))
            lowers.append(float("nan"))
            uppers.append(float("nan"))
            continue
        mean_v = sum(vals) / n
        if n == 1:
            se = 0.0
        else:
            # sample standard deviation (n-1)
            var = sum((v - mean_v) ** 2 for v in vals) / (n - 1)
            sd = _math.sqrt(var)
            se = sd / _math.sqrt(n)
        margin = z * se
        means.append(mean_v)
        lowers.append(mean_v - margin)
        uppers.append(mean_v + margin)

    return means, lowers, uppers
