"""
Thompson Sampling Agent
"""

from __future__ import annotations

from .base import BaseAgent, BaseAlgorithm
from .schemas import (
    ThompsonSamplingConfig,
    ThompsonSamplingRewardStates,
)

from bandit_lib.env import Environment
from bandit_lib.utils import ProcessDataLogger
from bandit_lib.agents.schemas import MetricsConfig


class ThompsonSamplingAlgorithm(
    BaseAlgorithm["ThompsonSamplingAgent", ThompsonSamplingConfig]
):
    def __init__(self, config: ThompsonSamplingConfig) -> None:
        super().__init__(config, agent_type=ThompsonSamplingAgent)

    def run(self) -> int:
        r = self.agent.rewards_states
        beta = self.agent.nprng.beta(r.alpha, r.beta)
        index = int(beta.argmax())
        return index


class ThompsonSamplingAgent(
    BaseAgent[ThompsonSamplingRewardStates, ThompsonSamplingAlgorithm]
):
    def __init__(
        self,
        name: str,
        env: Environment,
        algorithm: ThompsonSamplingAlgorithm,
        metrics_config: MetricsConfig = MetricsConfig(),
        process_data_logger: ProcessDataLogger | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            name, env, algorithm, metrics_config, process_data_logger, seed
        )
        self.rewards_states = ThompsonSamplingRewardStates.create(
            env.arm_num, metrics_config.sliding_window_size
        )

    def act(self) -> int:
        return self.algorithm.run()

    def pull(self, arm_index: int) -> int:
        return self.env.get_arm_result(arm_index, current_step=self.steps)

    def update(self, arm_index: int, reward: int) -> None:
        super().update(arm_index, reward)

        # decay alpha and beta
        if self.algorithm.config.enable_decay_alpha:
            self.rewards_states.alpha *= self.algorithm.config.discount_factor
            self.rewards_states.beta *= self.algorithm.config.discount_factor

            # handle underflow
            underflow_mask_alpha = self.rewards_states.alpha < 1e-10
            underflow_mask_beta = self.rewards_states.beta < 1e-10
            self.rewards_states.alpha[underflow_mask_alpha] = 1e-10
            self.rewards_states.beta[underflow_mask_beta] = 1e-10

        # update alpha and beta
        self.rewards_states.alpha[arm_index, 0] += reward
        self.rewards_states.beta[arm_index, 0] += 1 - reward
