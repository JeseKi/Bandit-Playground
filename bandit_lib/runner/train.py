from typing import List, Set, Tuple, Type, cast
from multiprocessing import Pool
import numpy as np

from bandit_lib.agents import (
    BaseRewardStates,
    Metrics,
    BaseAlgorithm,
    AlgorithmType,
    AlgorithmConfig,
    GreedyAlgorithm,
    UCB1Algorithm,
    ThompsonSamplingAlgorithm,
    GreedyConfig,
    UCB1Config,
    ThompsonSamplingConfig,
)
from bandit_lib.agents.base import Agent_T

from bandit_lib.env import Environment, EnvConfig


def agent_factory(
    agent_type: Type[Agent_T],
    name: str,
    env: Environment,
    algorithm: BaseAlgorithm,
    seed: int,
) -> Agent_T:
    return agent_type(
        seed=seed,
        name=name,
        env=env,
        algorithm=algorithm,
    )


def algorithm_factory(
    algorithm_type: AlgorithmType,
    algorithm_config: AlgorithmConfig,
) -> BaseAlgorithm:
    if algorithm_type == AlgorithmType.GREEDY:
        algorithm_config = cast(GreedyConfig, algorithm_config)
        return GreedyAlgorithm(config=algorithm_config)
    elif algorithm_type == AlgorithmType.UCB1:
        algorithm_config = cast(UCB1Config, algorithm_config)
        return UCB1Algorithm(config=algorithm_config)
    elif algorithm_type == AlgorithmType.THOMPSON_SAMPLING:
        algorithm_config = cast(ThompsonSamplingConfig, algorithm_config)
        return ThompsonSamplingAlgorithm(config=algorithm_config)
    else:
        raise ValueError(f"Invalid algorithm type: {algorithm_type}")


def batch_train(
    agent_type: Type[Agent_T],
    name: str,
    arm_num: int,
    env_config: EnvConfig,
    algorithm_type: AlgorithmType,
    algorithm_config: AlgorithmConfig,
    repeat_times: int,
    step_num: int,
    base_seed: int = 42,
    worker_num: int = 4,
) -> Tuple[List[Agent_T], BaseRewardStates, List[Metrics]]:
    base_seed = base_seed
    env_seed_offset = 9973
    agents: List[Agent_T] = []
    for i in range(repeat_times):
        env = Environment(
            arm_num=arm_num, config=env_config, seed=base_seed + i * env_seed_offset
        )
        agent = agent_factory(
            agent_type=agent_type,
            name=name,
            env=env,
            algorithm=algorithm_factory(algorithm_type, algorithm_config),
            seed=base_seed + i * env_seed_offset,
        )
        agents.append(agent)

    with Pool(worker_num) as p:
        results: List[Tuple[Agent_T, BaseRewardStates, List[Metrics]]] = p.starmap(
            train, zip(agents, [step_num] * repeat_times)
        )

    rewards_states, avg_metrics = calculate_metrics(results)
    return agents, rewards_states, avg_metrics


def train(
    agent: Agent_T, step_num: int
) -> Tuple[Agent_T, BaseRewardStates, List[Metrics]]:
    for i in range(step_num):
        agent.step()
    return agent, agent.rewards_states, agent.metrics


def calculate_metrics(
    results: List[Tuple[Agent_T, BaseRewardStates, List[Metrics]]],
) -> Tuple[BaseRewardStates, List[Metrics]]:
    rewards: List[BaseRewardStates] = [result[1] for result in results]
    rewards_states = BaseRewardStates.create(
        arm_num=results[0][0].env.arm_num,
        sliding_window_size=results[0][0].metrics_config.sliding_window_size,
    )
    rewards_states.rewards = np.mean([reward.rewards for reward in rewards], axis=0)
    metrics_list: List[List[Metrics]] = [result[2] for result in results]
    avg_metrics: List[Metrics] = []
    for metrics in metrics_list:
        steps: Set[float] = set()
        for metric in metrics:
            steps.add(metric.current_step)
        if len(steps) >= 2:
            raise ValueError("Metrics must have the same step number.")
        avg_regret_rate = np.mean([metric.regret_rate for metric in metrics])
        avg_regret = np.mean([metric.regret for metric in metrics])
        avg_reward_rate = np.mean([metric.reward_rate for metric in metrics])
        avg_reward = np.mean([metric.reward for metric in metrics])
        avg_optimal_arm_rate = np.mean([metric.optimal_arm_rate for metric in metrics])
        avg_sliding_window_reward_rate = np.mean(
            [metric.sliding_window_reward_rate for metric in metrics]
        )
        avg_convergence_step = np.mean([metric.convergence_step for metric in metrics])
        avg_metrics.append(
            Metrics(
                regret_rate=float(avg_regret_rate),
                regret=float(avg_regret),
                reward_rate=float(avg_reward_rate),
                reward=float(avg_reward),
                optimal_arm_rate=float(avg_optimal_arm_rate),
                sliding_window_reward_rate=float(avg_sliding_window_reward_rate),
                convergence_step=float(avg_convergence_step),
            )
        )
    return rewards_states, avg_metrics
