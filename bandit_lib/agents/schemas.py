from __future__ import annotations

from enum import Enum
from collections import deque
from typing import Deque, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BaseRewardStates(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_encoders={np.ndarray: lambda v: v.tolist()}
    )

    rewards: np.ndarray = Field(
        ...,
        description="Rewards of the agent, shape: (num_arms, 2) where stats[i, 0] = arm i's pull count and stats[i, 1] = arm i's reward value.",
    )
    sliding_window_rewards: Deque[Tuple[float, float]] = Field(
        ...,
        description="Rewards of the agent in a sliding window. Tuple of (pull count, reward value).",
    )

    @classmethod
    def create(cls, arm_num: int, sliding_window_size: int) -> "BaseRewardStates":
        return cls(
            rewards=np.zeros((arm_num, 2)),
            sliding_window_rewards=deque(maxlen=sliding_window_size),
        )


class GreedyRewardStates(BaseRewardStates):
    q_values: np.ndarray = Field(
        ...,
        description="Q values of the arms, shape: (num_arms, 1) where stats[i, 0] = arm i's Q value.",
    )

    @classmethod
    def create(cls, arm_num: int, sliding_window_size: int) -> "GreedyRewardStates":
        return cls(
            rewards=np.zeros((arm_num, 2)),
            sliding_window_rewards=deque(maxlen=sliding_window_size),
            q_values=np.zeros((arm_num, 1)),
        )


class UCB1RewardStates(GreedyRewardStates):
    ucb1_values: np.ndarray = Field(
        ...,
        description="Q values with upper confidence bound, shape: (num_arms, 1) where stats[i, 0] = arm i's UCB1 value.",
    )

    @classmethod
    def create(cls, arm_num: int, sliding_window_size: int) -> "UCB1RewardStates":
        return cls(
            rewards=np.zeros((arm_num, 2)),
            sliding_window_rewards=deque(maxlen=sliding_window_size),
            q_values=np.zeros((arm_num, 1)),
            ucb1_values=np.zeros((arm_num, 1)),
        )


class ThompsonSamplingRewardStates(BaseRewardStates):
    alpha: np.ndarray = Field(
        ...,
        description="Alpha values of the arms, shape: (num_arms, 1) where stats[i, 0] = arm i's alpha value.",
    )
    beta: np.ndarray = Field(
        ...,
        description="Beta values of the arms, shape: (num_arms, 1) where stats[i, 0] = arm i's beta value.",
    )

    @classmethod
    def create(
        cls, arm_num: int, sliding_window_size: int
    ) -> "ThompsonSamplingRewardStates":
        return cls(
            rewards=np.zeros((arm_num, 2)),
            sliding_window_rewards=deque(maxlen=sliding_window_size),
            alpha=np.ones((arm_num, 1)),
            beta=np.ones((arm_num, 1)),
        )


class AlgorithmType(Enum):
    GREEDY = "greedy"
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"


class AlgorithmConfig(BaseModel):
    algorithm_type: AlgorithmType
    enable_decay_alpha: bool = Field(default=False)


class GreedyConfig(AlgorithmConfig):
    algorithm_type: AlgorithmType = AlgorithmType.GREEDY
    optimistic_initialization_enabled: bool = Field(default=False)
    optimistic_initialization_value: float = Field(default=1)
    epsilon: float = Field(default=0.1)
    enable_epsilon_decay: bool = Field(default=False)
    epsilon_decay_factor: float = Field(default=0.99)
    epsilon_min_value: float = Field(default=0.01)
    constant_step_decay_alpha: float = Field(default=0.1)


class UCB1Config(AlgorithmConfig):
    algorithm_type: AlgorithmType = AlgorithmType.UCB1
    constant_step_decay_alpha: float = Field(default=0.1)


class ThompsonSamplingConfig(AlgorithmConfig):
    algorithm_type: AlgorithmType = AlgorithmType.THOMPSON_SAMPLING
    discount_factor: float = Field(default=0.9)


class Metrics(BaseModel):
    """Metrics of the agent.
    In order to use the class to collect metrics while not training, some fields are initialized with -1.
    """

    current_step: float = Field(
        default=-1, description="The step number of the metrics when it is collected."
    )
    regret_rate: float
    regret: float
    reward_rate: float
    reward: float
    sliding_window_reward_rate: float = Field(
        default=-1, description="A reward rate that is in any sliding window."
    )
    optimal_arm_rate: float
    convergence_step: float = Field(
        default=-1,
        description="Using in static environment. Step count when the agent converges to the optimal arm.",
    )


class MetricsConfig(BaseModel):
    metrics_history_size: int = Field(default=500)
    sliding_window_size: int = Field(default=1000)
    optimal_arm_rate_threshold: float = Field(default=0.95)
    min_convergence_step: float = Field(
        default=100.0,
        description="The minimum step count when the agent converges to the optimal arm.",
    )
