from __future__ import annotations

from typing import List
import random

import numpy as np

from .schemas import EnvConfig, PiecewizeMethod


class Arm:
    def __init__(self, reward_probability: float, seed: int) -> None:
        self.reward_probability = reward_probability
        self.rng = np.random.default_rng(seed)

    def pull(self) -> bool:
        return self.rng.random() < self.reward_probability


class Environment:
    def __init__(
        self,
        arm_num: int,
        config: EnvConfig = EnvConfig(),
        seed: int = 42,
    ) -> None:
        """Initialize the environment

        Args:
            arm_num (int): Number of arms.
            config (EnvConfig): Configuration for the environment.
        """
        # config
        self.seed: int = seed
        self.rng: random.Random = random.Random(self.seed)
        self.nprng: np.random.Generator = np.random.default_rng(self.seed)
        self.config: EnvConfig = config

        # state
        self.best_arm_index: int = -1

        # init
        self.arm_num = arm_num
        self.arms: List[Arm] = []
        self._reset()

    @property
    def best_arm(self) -> Arm:
        return self.arms[self.best_arm_index]

    def get_arm_result(self, arm_index: int, current_step: int) -> bool:
        self._dynamic(current_step)
        return self.arms[arm_index].pull()

    def _reset(self) -> None:
        self.arms = []
        seed = int(self.rng.random() * 1000000)

        for i in range(self.arm_num):
            self.arms.append(
                Arm(reward_probability=(i + 1) / (self.arm_num + 1), seed=seed)
            )

        rng = random.Random(seed)
        rng.shuffle(self.arms)
        best_arm = max(self.arms, key=lambda x: x.reward_probability)
        self.best_arm_index = self.arms.index(best_arm)

    def _dynamic(self, current_step: int) -> None:
        if not self.config.enable_dynamic:
            return

        self._random_walk(current_step)
        self._piecewize(current_step)
        best_arm = max(self.arms, key=lambda x: x.reward_probability)
        self.best_arm_index = self.arms.index(best_arm)

    def _random_walk(self, current_step: int) -> None:
        if current_step != 0 and self.config.random_walk_internal % current_step != 0:
            return

        m = self.rng.sample(self.arms, self.config.random_walk_arm_num)
        samples = self.nprng.normal(
            0,
            self.config.random_walk_std,
            self.config.random_walk_arm_num,
        )
        for machine, sample in zip(m, samples):
            r = machine.reward_probability + sample
            if r < 0 or r > 1:
                r = machine.reward_probability - sample
            machine.reward_probability = r

    def _piecewize(self, current_step: int) -> None:
        if current_step != 0 and current_step % self.config.piecewize_internal != 0:
            return

        if self.config.piecewize_method == PiecewizeMethod.NONE:
            return
        elif self.config.piecewize_method == PiecewizeMethod.DETERMINISTIC_REWARD_DRIFT:
            self._deterministic_reward_drift()
        elif self.config.piecewize_method == PiecewizeMethod.PERMUTATION:
            self._reward_permutation()
        elif self.config.piecewize_method == PiecewizeMethod.RESET:
            self._reset()
        else:
            raise ValueError(
                f"Invalid piecewize method: {self.config.piecewize_method}"
            )

    def _reward_permutation(self) -> None:
        arms_sorted = sorted(self.arms, key=lambda m: m.reward_probability)
        n = len(arms_sorted)

        for i, arm in enumerate(arms_sorted):
            arm.reward_probability = (n - 1 - i) / n

    def _deterministic_reward_drift(self) -> None:
        for arm in self.arms:
            r = arm.reward_probability + 0.5
            if r > 1:
                r = arm.reward_probability - 0.5
            arm.reward_probability = r
