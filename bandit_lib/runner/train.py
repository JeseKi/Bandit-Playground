from typing import List, Tuple, Type, cast
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
    MetricsConfig,
)
from bandit_lib.agents.base import Agent_T
from bandit_lib.utils import ProcessDataLogger
from bandit_lib.env import Environment, EnvConfig


def agent_factory(
    agent_type: Type[Agent_T],
    name: str,
    env: Environment,
    algorithm: BaseAlgorithm,
    metrics_config: MetricsConfig = MetricsConfig(),
    seed: int = 42,
) -> Agent_T:
    return agent_type(
        seed=seed,
        name=name,
        env=env,
        algorithm=algorithm,
        metrics_config=metrics_config,
    )


def algorithm_factory(
    algorithm_config: AlgorithmConfig,
) -> BaseAlgorithm:
    if algorithm_config.algorithm_type == AlgorithmType.GREEDY:
        algorithm_config = cast(GreedyConfig, algorithm_config)
        return GreedyAlgorithm(config=algorithm_config)
    elif algorithm_config.algorithm_type == AlgorithmType.UCB1:
        algorithm_config = cast(UCB1Config, algorithm_config)
        return UCB1Algorithm(config=algorithm_config)
    elif algorithm_config.algorithm_type == AlgorithmType.THOMPSON_SAMPLING:
        algorithm_config = cast(ThompsonSamplingConfig, algorithm_config)
        return ThompsonSamplingAlgorithm(config=algorithm_config)
    else:
        raise ValueError(f"Invalid algorithm type: {algorithm_config.algorithm_type}")


def batch_train(
    run_id: str,
    agent_type: Type[Agent_T],
    name: str,
    arm_num: int,
    env_config: EnvConfig,
    algorithm_config: AlgorithmConfig,
    repeat_times: int,
    step_num: int,
    metrics_config: MetricsConfig = MetricsConfig(),
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
            algorithm=algorithm_factory(algorithm_config=algorithm_config),
            metrics_config=metrics_config,
            seed=base_seed + i * env_seed_offset,
        )
        agent.set_logger(
            logger=ProcessDataLogger(
                run_id=run_id,
                total_steps=step_num,
                agent=agent,
            )
        )
        agents.append(agent)

    with Pool(worker_num) as p:
        results: List[Tuple[Agent_T, BaseRewardStates, List[Metrics]]] = p.starmap(
            train, zip(agents, [step_num] * repeat_times)
        )

    trained_agents: List[Agent_T] = [result[0] for result in results]
    rewards_states, avg_metrics = calculate_metrics(results)
    return trained_agents, rewards_states, avg_metrics


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

    metrics_history_avg: List[Metrics] = []
    for step_metrics in zip(*metrics_list):
        step_metrics = cast(Tuple[Metrics], step_metrics)
        avg_regret_rate = np.mean([metric.regret_rate for metric in step_metrics])
        avg_regret = np.mean([metric.regret for metric in step_metrics])
        avg_reward_rate = np.mean([metric.reward_rate for metric in step_metrics])
        avg_reward = np.mean([metric.reward for metric in step_metrics])
        avg_optimal_arm_rate = np.mean(
            [metric.optimal_arm_rate for metric in step_metrics]
        )
        avg_sliding_window_reward_rate = np.mean(
            [metric.sliding_window_reward_rate for metric in step_metrics]
        )
        avg_convergence_step = np.mean(
            [metric.convergence_step for metric in step_metrics]
        )
        metrics_history_avg.append(
            Metrics(
                current_step=float(step_metrics[0].current_step),
                regret_rate=float(avg_regret_rate),
                regret=float(avg_regret),
                reward_rate=float(avg_reward_rate),
                reward=float(avg_reward),
                optimal_arm_rate=float(avg_optimal_arm_rate),
                sliding_window_reward_rate=float(avg_sliding_window_reward_rate),
                convergence_step=float(avg_convergence_step),
            )
        )
    return rewards_states, metrics_history_avg
