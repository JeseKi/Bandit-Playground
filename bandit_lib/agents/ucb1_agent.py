from __future__ import annotations

import numpy as np

from .base import BaseAgent, BaseAlgorithm
from .schemas import (
    UCB1Config,
    UCB1RewardStates,
)

from bandit_lib.env import Environment
from bandit_lib.utils import ProcessDataLogger
from bandit_lib.agents.schemas import MetricsConfig


class UCB1Algorithm(BaseAlgorithm["UCB1Agent", UCB1Config]):
    def __init__(self, config: UCB1Config) -> None:
        super().__init__(config, agent_type=UCB1Agent)

    def run(self) -> int:
        return int(np.argmax(self.agent.rewards_states.ucb1_values))


class UCB1Agent(BaseAgent[UCB1RewardStates, UCB1Algorithm]):
    def __init__(
        self,
        name: str,
        env: Environment,
        algorithm: UCB1Algorithm,
        metrics_config: MetricsConfig = MetricsConfig(),
        process_data_logger: ProcessDataLogger | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            name, env, algorithm, metrics_config, process_data_logger, seed
        )

        self.rewards_states = UCB1RewardStates.create(
            env.arm_num, metrics_config.sliding_window_size
        )

    def act(self) -> int:
        return self.algorithm.run()

    def pull(self, arm_index: int) -> int:
        return self.env.get_arm_result(arm_index, current_step=self.steps)

    def update(self, arm_index: int, reward: int) -> None:
        super().update(arm_index, reward)

        # calculate q value
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

        # calculate ucb1 value
        log_steps = np.log(self.steps) if self.steps > 0 else 0.0
        counts_np = self.rewards_states.rewards[arm_index, 0]
        self.rewards_states.ucb1_values = q + np.sqrt(
            2 * log_steps / np.maximum(counts_np, 1)
        )
