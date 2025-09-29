from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PiecewizeMethod(Enum):
    PERMUTATION = "permutation"
    RESET = "reset"
    DETERMINISTIC_REWARD_DRIFT = "deterministic_reward_drift"


class EnvConfig(BaseModel):
    enable_dynamic: bool = Field(
        default=False, description="Enable dynamic environment."
    )
    random_walk_internal: int = Field(
        default=1, description="Internal for random walk", ge=1
    )
    random_walk_arm_num: int = Field(
        default=1, description="Number of arms for random walk", ge=1
    )
    random_walk_std: float = Field(
        default=0.01, description="Standard deviation for random walk", ge=0
    )
    piecewize_internal: int = Field(
        default=1, description="Internal for piecewize method", ge=1
    )
    piecewize_method: PiecewizeMethod = Field(
        default=PiecewizeMethod.PERMUTATION, description="Method for piecewize method"
    )
    seed: int = Field(default=42, description="Seed for the environment.")
