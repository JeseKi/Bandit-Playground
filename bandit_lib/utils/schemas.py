from __future__ import annotations

from datetime import datetime
from typing import List
import uuid

from pydantic import BaseModel, Field

from bandit_lib.agents.schemas import (
    Metrics,
    MetricsConfig,
    AlgorithmConfig,
    BaseRewardStates,
)
from bandit_lib.env.schemas import EnvConfig


class ProcessDataDump(BaseModel):
    run_id: str
    create_at: datetime = Field(default_factory=datetime.now)
    rewards: BaseRewardStates
    metrics: Metrics
    metrics_history_avg: List[Metrics] = Field(default_factory=list)
    metrics_history: List[List[Metrics]] = Field(default_factory=list)


class MetaDataDump(BaseModel):
    experiment_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    experiment_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The id of the experiment.",
    )
    agent_runs_num: int = Field(
        default=1,
        description="The number of agents repeated running in the experiment.",
    )
    total_steps: int = Field(
        ..., description="The total steps of every agent in the experiment."
    )
    arm_num: int = Field(..., description="The number of arms in the environment.")
    agent_seed: int = Field(
        ..., description="The seed of the agent first run in the experiment."
    )
    algorithm: AlgorithmConfig = Field(..., description="The algorithm of the agent.")
    metrics_config: MetricsConfig = Field(
        ..., description="The metrics config of the agent."
    )
    env_config: EnvConfig = Field(
        ..., description="The environment config of the agent."
    )
