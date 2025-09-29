from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import math

from bandit_lib.agents.schemas import Metrics, MetricsConfig
from .schemas import ProcessDataDump

class ProcessDataLogger:
    def __init__(self, run_id: str, total_steps: int, metrics_config: MetricsConfig) -> None:
        # config
        self.run_id: str = run_id
        self.total_steps: int = total_steps
        self.metrics_config: MetricsConfig = metrics_config
        
        # state
        self.metrics: List[Metrics] = []
        self._grid: List[int] = []
        
        # init
        self._grid = self._build_log_grid()
    
    @property
    def grid(self) -> List[int]:
        return self._grid
    
    def record(self, step: int, metrics: Metrics) -> None:
        """Record the metrics for the agent if the step is in the grid."""
        if step in self.grid:
            self.metrics.append(metrics)
            
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
            total_steps=self.total_steps,
            metrics=self.metrics,
        )
    
    def _build_log_grid(self) -> List[int]:
        if self.total_steps <= 0:
            raise ValueError("Total steps must be greater than 0")
        if self.metrics_config.metrics_history_size <= 0:
            raise ValueError("Metrics history size must be greater than 0")
        
        logT = 0.0 if self.total_steps == 1 else math.log10(self.total_steps)
        if self.metrics_config.metrics_history_size == 1:
            raw = [0.0]
        else:
            step = logT / (self.metrics_config.metrics_history_size - 1)
            raw = [i * step for i in range(self.metrics_config.metrics_history_size)]
            
        grid = [int(math.ceil( 10 ** v)) for v in raw]
        grid.append(1)
        grid.append(self.total_steps)
        
        grid = sorted(set(grid))
        grid = [x for x in grid if 1 <= x <= self.total_steps]
        return grid