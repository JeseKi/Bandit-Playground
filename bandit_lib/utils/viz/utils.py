import math
import hashlib
from typing import Tuple, Sequence, List, TYPE_CHECKING

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
