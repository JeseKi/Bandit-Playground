from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, TYPE_CHECKING

import math

from bandit_lib.agents.schemas import Metrics
from .schemas import ProcessDataDump, MetaDataDump

if TYPE_CHECKING:
    from bandit_lib.agents.base import BaseAgent


def save_meta_data(meta_data: MetaDataDump, path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(meta_data.model_dump_json(indent=4))


def save_process_data(process_data: ProcessDataDump, path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(process_data.model_dump_json(indent=4))


class ProcessDataLogger:
    def __init__(
        self,
        run_id: str,
        total_steps: int,
        agent: "BaseAgent",
    ) -> None:
        # config
        self.run_id: str = run_id
        self.total_steps: int = total_steps
        self.agent: "BaseAgent" = agent

        # state
        self._grid: List[int] = []

        # init
        self._grid = self._build_log_grid()

    @property
    def grid(self) -> List[int]:
        return self._grid

    def record(self, metrics: Metrics) -> None:
        """Record the metrics for the agent if the step is in the grid."""
        self.agent.metrics.append(metrics)

    def should_record(self, step: int) -> bool:
        return step in self.grid

    def save(self, path: Path) -> None:
        dump = self.export()
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(dump.model_dump_json(indent=4))

    def export(self) -> ProcessDataDump:
        metrics_history = self._build_metrics_history()
        metrics_history_avg = self._calculate_metrics_history_avg(metrics_history)
        final_metrics = self._determine_final_metrics(metrics_history_avg)

        return ProcessDataDump(
            run_id=self.run_id,
            create_at=datetime.now(),
            rewards=self.agent.rewards_states,
            metrics=final_metrics,
            metrics_history_avg=metrics_history_avg,
            metrics_history=metrics_history,
        )

    def _build_metrics_history(self) -> List[List[Metrics]]:
        agent_history = [metric.model_copy(deep=True) for metric in self.agent.metrics]
        return [agent_history]

    def _calculate_metrics_history_avg(
        self, metrics_history: List[List[Metrics]]
    ) -> List[Metrics]:
        if not metrics_history:
            return []

        min_length = min(len(history) for history in metrics_history)
        if min_length == 0:
            return []

        averaged: List[Metrics] = []
        for idx in range(min_length):
            step_metrics = [history[idx] for history in metrics_history]
            averaged.append(self._average_metrics(step_metrics))
        return averaged

    def _average_metrics(self, metrics_list: List[Metrics]) -> Metrics:
        if not metrics_list:
            raise ValueError("metrics_list cannot be empty when averaging metrics")

        count = len(metrics_list)
        current_step = float(metrics_list[0].current_step)
        regret_rate = sum(metric.regret_rate for metric in metrics_list) / count
        regret = sum(metric.regret for metric in metrics_list) / count
        reward_rate = sum(metric.reward_rate for metric in metrics_list) / count
        reward = sum(metric.reward for metric in metrics_list) / count
        sliding_window_reward_rate = (
            sum(metric.sliding_window_reward_rate for metric in metrics_list) / count
        )
        optimal_arm_rate = (
            sum(metric.optimal_arm_rate for metric in metrics_list) / count
        )
        convergence_step = (
            sum(metric.convergence_step for metric in metrics_list) / count
        )

        return Metrics(
            current_step=current_step,
            regret_rate=float(regret_rate),
            regret=float(regret),
            reward_rate=float(reward_rate),
            reward=float(reward),
            sliding_window_reward_rate=float(sliding_window_reward_rate),
            optimal_arm_rate=float(optimal_arm_rate),
            convergence_step=float(convergence_step),
        )

    def _determine_final_metrics(self, metrics_history_avg: List[Metrics]) -> Metrics:
        if metrics_history_avg:
            return metrics_history_avg[-1].model_copy(deep=True)
        return self._snapshot_current_metrics()

    def _snapshot_current_metrics(self) -> Metrics:
        current_step = float(self.agent.steps)
        regret = float(self.agent.regret())
        regret_rate = float(self.agent.regret_rate())
        reward = float(self.agent.total_reward())
        reward_rate = float(self.agent.reward_rate())
        optimal_arm_rate = (
            float(self.agent.optimal_arm_rate()) if self.agent.steps > 0 else 0.0
        )
        convergence_step = float(self.agent.convergence_step)

        try:
            sliding_window_reward_rate = float(self.agent.sliding_window_reward_rate())
        except ZeroDivisionError:
            sliding_window_reward_rate = -1.0

        return Metrics(
            current_step=current_step,
            regret_rate=regret_rate,
            regret=regret,
            reward_rate=reward_rate,
            reward=reward,
            sliding_window_reward_rate=sliding_window_reward_rate,
            optimal_arm_rate=optimal_arm_rate,
            convergence_step=convergence_step,
        )

    def _build_log_grid(self) -> List[int]:
        if self.total_steps <= 0:
            raise ValueError("Total steps must be greater than 0")
        if self.agent.metrics_config.metrics_history_size <= 0:
            raise ValueError("Metrics history size must be greater than 0")

        logT = 0.0 if self.total_steps == 1 else math.log10(self.total_steps)
        if self.agent.metrics_config.metrics_history_size == 1:
            raw = [0.0]
        else:
            step = logT / (self.agent.metrics_config.metrics_history_size - 1)
            raw = [
                i * step for i in range(self.agent.metrics_config.metrics_history_size)
            ]

        grid = [int(math.ceil(10**v)) for v in raw]
        grid.append(1)
        grid.append(self.total_steps)

        grid = sorted(set(grid))
        grid = [x for x in grid if 1 <= x <= self.total_steps]
        return grid
