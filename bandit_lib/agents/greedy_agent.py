from __future__ import annotations

import numpy as np

from .base import BaseAgent, BaseAlgorithm
from .schemas import (
    GreedyConfig,
    GreedyRewardStates,
)

from bandit_lib.env import Environment
from bandit_lib.utils import ProcessDataLogger
from bandit_lib.agents.schemas import MetricsConfig


class GreedyAlgorithm(BaseAlgorithm["GreedyAgent", GreedyConfig]):
    def __init__(self, config: GreedyConfig) -> None:
        super().__init__(config, agent_type=GreedyAgent)
        self._inited: bool = False
        
    def init(self) -> None:
        if self._inited:
            return
        self.optimistic_init_history = [0] * self.agent.env.arm_num
        self.optimistic_inited = (
            False if self.config.optimistic_initialization_enabled else True
        )
        self._inited = True

    def run(self) -> int:
        self.init()
        arm_index = int(np.argmax(self.agent.rewards_states.q_values))  # greedy action
        arm_index = self.epsilon(arm_index)  # epsilon-greedy action
        arm_index = self.optimistic_init(arm_index)  # optimistic initialization action
        return arm_index

    def optimistic_init(self, arm_index: int) -> int:
        if not self.optimistic_inited:
            for i in range(self.agent.env.arm_num):
                if (
                    self.optimistic_init_history[i]
                    >= self.config.optimistic_initialization_value
                ):
                    continue
                self.optimistic_init_history[i] += 1
                return i
            self.optimistic_inited = True
        return arm_index

    def epsilon(self, arm_index) -> int:
        if self.config.epsilon:
            epsilon_rand = self.agent.rng.randint(0, 1)
            if epsilon_rand < self.config.epsilon:
                new_arm_index = self.agent.rng.randint(0, self.agent.env.arm_num - 1)
                return new_arm_index
            if (
                self.config.enable_epsilon_decay
                and self.config.epsilon > self.config.epsilon_min_value
            ):
                self.config.epsilon *= self.config.epsilon_decay_factor
                self.config.epsilon = max(
                    self.config.epsilon, self.config.epsilon_min_value
                )
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
