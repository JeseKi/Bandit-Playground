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
        return ProcessDataDump(
            run_id=self.run_id,
            create_at=datetime.now(),
            rewards=self.agent.rewards_states,
            metrics=self.agent.metrics,
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
