from __future__ import annotations

import numpy as np

from .base import BaseAgent, BaseAlgorithm
from .schemas import (
    UCBConfig,
    UCBRewardStates,
)

from bandit_lib.env import Environment
from bandit_lib.agents.schemas import MetricsConfig


class UCBAlgorithm(BaseAlgorithm["UCBAgent", UCBConfig]):
    def __init__(self, config: UCBConfig) -> None:
        super().__init__(config, agent_type=UCBAgent)

    def run(self) -> int:
        return int(np.argmax(self.agent.rewards_states.ucb_values[:, 0]))


class UCBAgent(BaseAgent[UCBRewardStates, UCBAlgorithm]):
    def __init__(
        self,
        name: str,
        env: Environment,
        algorithm: UCBAlgorithm,
        metrics_config: MetricsConfig = MetricsConfig(),
        seed: int = 42,
    ) -> None:
        super().__init__(name, env, algorithm, metrics_config, seed)

        self.rewards_states = UCBRewardStates.create(
            env.arm_num, metrics_config.sliding_window_size
        )

    def act(self) -> int:
        return self.algorithm.run()

    def pull(self, arm_index: int) -> int:
        return self.env.get_arm_result(arm_index, current_step=self.steps)

    def update(self, arm_index: int, reward: int) -> None:
        super().update(arm_index, reward)

        old_q = self.rewards_states.q_values[arm_index, 0]
        old_count = self.rewards_states.discounted_count[arm_index, 0]

        gamma = self.algorithm.config.forgetting_factor
        self.rewards_states.discounted_count *= gamma
        self.rewards_states.discounted_count[arm_index, 0] += 1

        error = reward - old_q
        effective_learning_rate = 1 / (gamma * old_count + 1)
        new_q = old_q + error * effective_learning_rate
        self.rewards_states.q_values[arm_index, 0] = new_q

        total_discounted_count = np.sum(self.rewards_states.discounted_count[:, 0])
        log_total_discounted_count = (
            np.log(total_discounted_count) if total_discounted_count > 0 else 0
        )

        discounted_count = self.rewards_states.discounted_count
        self.rewards_states.ucb_values = self.rewards_states.q_values + np.sqrt(
            self.algorithm.config.exploration_coefficient
            * log_total_discounted_count
            / np.maximum(discounted_count, 1)
        )
