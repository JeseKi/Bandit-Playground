from typing import List

_METRIC_LABELS: List[str] = [
    "regret_rate",
    "regret",
    "reward_rate",
    "reward",
    "optimal_arm_rate",
    "convergence_rate",
    "sliding_window_reward_rate",
]


def get_metric_labels() -> List[str]:
    return _METRIC_LABELS.copy()
