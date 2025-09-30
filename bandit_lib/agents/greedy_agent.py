from __future__ import annotations

import numpy as np

from .base import BaseAgent, BaseAlgorithm
from .schemas import (
    AlgorithmConfig,
    GreedyRewardStates,
)

from bandit_lib.env import Environment
from bandit_lib.utils import ProcessDataLogger
from bandit_lib.agents.schemas import MetricsConfig


class GreedyAlgorithm(BaseAlgorithm["GreedyAgent"]):
    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__(config, agent_type=GreedyAgent)
        self.optimistic_init_history = [0] * self.agent.env.arm_num
        self.optimistic_inited = (
            False if self.config.optimistic_initialization_enabled else True
        )

    def run(self) -> int:
        arm_index = int(np.argmax(self.agent.rewards_states.q_values))
        if not self.optimistic_inited:
            for i in range(self.agent.env.arm_num):
                if (
                    self.optimistic_init_history[i]
                    >= self.config.optimistic_initialization_value
                ):
                    continue
                self.optimistic_init_history[i] += 1
                arm_index = i
                break
            self.optimistic_inited = True

        return arm_index


class GreedyAgent(BaseAgent[GreedyRewardStates, GreedyAlgorithm]):
    def __init__(
        self,
        name: str,
        env: Environment,
        algorithm: GreedyAlgorithm,
        metrics_config: MetricsConfig = MetricsConfig(),
        process_data_logger: ProcessDataLogger | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            name, env, algorithm, metrics_config, process_data_logger, seed
        )
        self.rewards_states = GreedyRewardStates.create(
            env.arm_num, metrics_config.sliding_window_size
        )

    def act(self) -> int:
        return self.algorithm.run()

    def pull(self, arm_index: int) -> int:
        return self.env.get_arm_result(arm_index, self.steps)

    def update(self, arm_index: int, reward: int) -> None:
        super().update(arm_index, reward)

        old_q = self.rewards_states.q_values[arm_index, 0]
        count = self.rewards_states.rewards[arm_index, 0]
        q = 0
        if self.algorithm.config.enable_decay_alpha:
            q = old_q + self.algorithm.config.constant_step_decay_alpha * (
                reward - old_q
            )
        else:
            q = old_q + (reward - old_q) / count
        self.rewards_states.q_values[arm_index, 0] = q
