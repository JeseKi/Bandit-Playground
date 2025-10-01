from __future__ import annotations

from typing import List, TypeVar, Generic, Type
from abc import ABC, abstractmethod
import random

import numpy as np

from .schemas import AlgorithmConfig, BaseRewardStates, Metrics, MetricsConfig
from bandit_lib.env import Environment
from bandit_lib.utils import ProcessDataLogger

AgentReward_T = TypeVar("AgentReward_T", bound=BaseRewardStates)
AgentAlgorithm_T = TypeVar("AgentAlgorithm_T", bound="BaseAlgorithm")
AlgorithmConfig_T = TypeVar("AlgorithmConfig_T", bound=AlgorithmConfig)
Agent_T = TypeVar("Agent_T", bound="BaseAgent")


class BaseAlgorithm(ABC, Generic[Agent_T, AlgorithmConfig_T]):
    def __init__(self, config: AlgorithmConfig_T, agent_type: Type[Agent_T]) -> None:
        self._agent: Agent_T | None = None
        self._target_type: Type[Agent_T] = agent_type
        self.config: AlgorithmConfig_T = config

    @property
    def agent(self) -> Agent_T:
        if self._agent is None:
            raise ValueError("Agent is not set")
        return self._agent

    @agent.setter
    def agent(self, agent: Agent_T) -> None:
        if not isinstance(agent, self._target_type):
            raise ValueError(f"Agent must be of type {self._target_type}")
        self._agent = agent

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError


class BaseAgent(ABC, Generic[AgentReward_T, AgentAlgorithm_T]):
    def __init__(
        self,
        name: str,
        env: Environment,
        algorithm: AgentAlgorithm_T,
        metrics_config: MetricsConfig = MetricsConfig(),
        seed: int = 42,
    ) -> None:
        "self.rewards_states must be set in the subclass. Such as `self.rewards_states = BaseRewardStates.create(self.env.arm_num)`."
        # config
        self.name: str = name
        self.env: Environment = env
        self.algorithm: AgentAlgorithm_T = algorithm
        self.algorithm.agent = self
        self.seed: int = seed
        self.rng: random.Random = random.Random(seed)
        self.nprng: np.random.Generator = np.random.default_rng(seed)
        self.metrics_config: MetricsConfig = metrics_config

        # state
        self.steps: int = 0
        self.optimal_arm_chosen_times: int = 0
        self.metrics: List[Metrics] = []
        self.convergence_step: int = -1
        self.process_data_logger: ProcessDataLogger | None

        self._convergenced: bool = False

        # init
        self.rewards_states: AgentReward_T  # must be set in the subclass

    def set_logger(self, logger: ProcessDataLogger) -> None:
        self.process_data_logger = logger

    def step(self) -> None:
        """Step the agent."""
        arm_index = self.act()
        reward = self.pull(arm_index=arm_index)
        self.update(arm_index=arm_index, reward=reward)

    @abstractmethod
    def act(self) -> int:
        """Action. Return the arm index to pull. Return a arm index."""
        raise NotImplementedError

    @abstractmethod
    def pull(self, arm_index: int) -> int:
        """Pull the arm. Unlike `act`, this function will return the reward."""
        raise NotImplementedError

    def update(self, arm_index: int, reward: int) -> None:
        self.steps += 1
        # update rewards
        self.rewards_states.rewards[arm_index, 0] += 1
        self.rewards_states.rewards[arm_index, 1] += reward
        self.rewards_states.sliding_window_rewards.append((1, reward))

        # update metrics
        if arm_index == self.env.best_arm_index:
            self.optimal_arm_chosen_times += 1

        if (
            not self.env.config.enable_dynamic
            and not self._convergenced
            and self.optimal_arm_rate()
            >= self.metrics_config.optimal_arm_rate_threshold
        ):
            self._convergenced = True
            self.convergence_step = self.steps

        if (
            self.process_data_logger is not None
            and self.process_data_logger.should_record(self.steps)
        ):
            metrics = Metrics(
                current_step=self.steps,
                regret_rate=self.regret_rate(),
                regret=self.regret(),
                reward_rate=self.reward_rate(),
                reward=self.total_reward(),
                optimal_arm_rate=self.optimal_arm_rate(),
                sliding_window_reward_rate=self.sliding_window_reward_rate(),
                convergence_step=self.convergence_step,
            )
            self.process_data_logger.record(metrics)

    def regret(self) -> float:
        """Calculate the regret of the agent."""
        best_reward = self.env.best_arm.reward_probability * self.steps
        return best_reward - self.total_reward()

    def regret_rate(self) -> float:
        """Calculate the regret rate of the agent."""
        try:
            if self.steps == 0:
                return 0
            best_reward = self.env.best_arm.reward_probability * self.steps
            return self.regret() / best_reward

        except ZeroDivisionError:
            print("ZeroDivisionError", self.steps, self.env.best_arm.reward_probability)
            for arm in self.env.arms:
                if arm.reward_probability == 0:
                    print(arm.reward_probability, self.env.arms.index(arm), [arm.reward_probability for arm in self.env.arms])
            return 0

    def reward_rate(self) -> float:
        """Calculate the reward rate of the agent."""
        if self.steps == 0:
            return 0
        return self.total_reward() / self.steps

    def total_reward(self) -> int:
        """Calculate the total reward of the agent."""
        r = int(np.sum(self.rewards_states.rewards[:, 1]))
        return r

    def optimal_arm_rate(self) -> float:
        """Calculate the optimal arm rate of the agent."""
        return self.optimal_arm_chosen_times / self.steps

    def sliding_window_reward_rate(self) -> float:
        """Calculate the sliding window reward rate of the agent."""
        rewards: int = np.sum(
            [reward for _, reward in self.rewards_states.sliding_window_rewards]
        )
        pull_counts: int = np.sum(
            [pull_count for pull_count, _ in self.rewards_states.sliding_window_rewards]
        )
        return rewards / pull_counts
