from __future__ import annotations

from typing import List, Sequence, Mapping, Tuple, TYPE_CHECKING

import math
import pandas as pd

from bandit_lib.utils.viz.utils import z_value_for_confidence

if TYPE_CHECKING:
    from bandit_lib.agents.schemas import Metrics
    from bandit_lib.agents.base import BaseAgent
    from bandit_lib.utils.schemas import ProcessDataDump


def _format_mean_ci(mean: float, margin: float, digits: int) -> str:
    """Format mean and interval margin to a fixed decimal places string.

    Example: mean=0.301352, margin=0.003467, digits=4 -> "0.3014 ± 0.0035"
    """
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(margin)}"


def build_confidence_table_from_metrics(
    groups: Mapping[str, Sequence[Metrics]],
    metrics: List[str],
    confidence: float = 0.95,
    digits: int = 4,
) -> pd.DataFrame:
    """Calculate mean and confidence interval of specified metrics from final `Metrics` results of multiple experiments and return as DataFrame.

    Args:
    - groups: Mapping from agent name to its final `Metrics` from multiple experiments.
    - metrics: Names of `Metrics` fields to be calculated, such as "regret_rate", "regret", etc.
    - confidence: Confidence level, default is 0.95.
    - digits: Number of decimal places to display, default is 4.

    Returns:
    - pandas.DataFrame: Columns are ["Agent", *metrics], data as formatted strings.

    Raises:
    - ValueError: Raised when samples for an agent are empty.
    - AttributeError: Raised when the metric field does not exist on `Metrics`.
    """

    if digits < 0:
        raise ValueError("digits cannot be negative")

    z = z_value_for_confidence(confidence)

    rows: List[dict] = []
    metric_list: List[str] = list(metrics)

    for agent_name, samples in groups.items():
        values_count = 0 if samples is None else len(samples)
        if values_count == 0:
            raise ValueError(
                f"Agent '{agent_name}' has empty samples, cannot calculate confidence interval"
            )

        row: dict = {"Agent": agent_name}
        for field in metric_list:
            # Collect all sample values for this metric
            xs: List[float] = [float(getattr(m, field)) for m in samples]

            n = len(xs)
            mean_v = sum(xs) / n
            if n <= 1:
                margin = 0.0
            else:
                var = sum((x - mean_v) ** 2 for x in xs) / (n - 1)
                sd = math.sqrt(var)
                se = sd / math.sqrt(n)
                margin = z * se

            row[field] = _format_mean_ci(mean_v, margin, digits)

        rows.append(row)

    # Construct DataFrame with column order Agent + metrics
    df = pd.DataFrame(rows)
    ordered_cols = ["Agent", *metric_list]
    # Only keep required columns to avoid other fields carried in groups
    return df.loc[:, ordered_cols]


def build_confidence_table_from_runs(
    runs_data: Sequence[Tuple[ProcessDataDump, Sequence["BaseAgent"]]],
    metrics: List[str],
    confidence: float = 0.95,
    digits: int = 4,
) -> pd.DataFrame:
    """Generate confidence interval table directly from `RESULTS_LIST` structure in example notebook."""

    grouped: dict[str, List[Metrics]] = {}

    for item in runs_data:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "runs_data elements must be in the form (ProcessDataDump, List[Agent])"
            )

        process_data, agents = item
        if not agents:
            raise ValueError(
                "agents in a run result is empty, cannot calculate confidence interval"
            )

        agent_name = agents[0].name

        samples: List[Metrics] = []
        for agent in agents:
            m_list = agent.metrics
            if m_list and len(m_list) > 0:
                samples.append(m_list[-1].model_copy(deep=True))
                continue

            try:
                current_step = float(agent.steps)
                regret_rate = float(agent.regret_rate())
                regret = float(agent.regret())
                reward_rate = float(agent.reward_rate())
                reward = float(agent.total_reward())
                optimal_arm_rate = float(agent.optimal_arm_rate())
                try:
                    sliding_window_reward_rate = float(
                        agent.sliding_window_reward_rate()
                    )
                except ZeroDivisionError:
                    sliding_window_reward_rate = -1.0
                convergence_step = float(agent.convergence_step)
            except Exception:
                current_step = -1.0
                regret_rate = -1.0
                regret = -1.0
                reward_rate = -1.0
                reward = -1.0
                optimal_arm_rate = -1.0
                sliding_window_reward_rate = -1.0
                convergence_step = -1.0

            samples.append(
                Metrics(
                    current_step=current_step,
                    regret_rate=regret_rate,
                    regret=regret,
                    reward_rate=reward_rate,
                    reward=reward,
                    sliding_window_reward_rate=sliding_window_reward_rate,
                    optimal_arm_rate=optimal_arm_rate,
                    convergence_step=convergence_step,
                )
            )

        if agent_name not in grouped:
            grouped[agent_name] = []
        grouped[agent_name].extend(samples)

    df = build_confidence_table_from_metrics(
        groups=grouped,
        metrics=metrics,
        confidence=confidence,
        digits=digits,
    )

    if "Agent" in df.columns:
        df = df.sort_values(by="Agent").reset_index(drop=True)
    return df
