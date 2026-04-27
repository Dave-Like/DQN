import os
import gymnasium as gym
import highway_env
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from gymnasium.vector import AsyncVectorEnv
from DQNAgent import DQNAgent
from trainer import train_agent
from DuelingDQNAgent import DuelingDQNAgent

NUM_ENVS = min(8, max(1, (os.cpu_count() or 2) // 2))


def create_highway_env():
    env = gym.make("highway-v0")
    env.unwrapped.config.update({
        "vehicles_count": 12,
        "duration": 50,
        "normalize_reward": True,
        "collision_reward": -10,
        "high_speed_reward": 1.0,
        "lane_change_reward": 0.05,
        "policy_frequency": 4,
        "offscreen_rendering": False,
        "render_fps": 0,
        "survival_reward": 0,
        "max_episode_steps":40
    })
    env = FlattenObservation(env)
    return env


def main():
    num_envs = NUM_ENVS
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.set_num_interop_threads(max(1, min(8, os.cpu_count() or 1)))

    env = AsyncVectorEnv([create_highway_env for _ in range(num_envs)], shared_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=192,
        exploration_type="epsilon",
        buffer_capacity=400000,
        batch_size=192,
        gamma=0.98,
        lr=4e-4,
        target_update_freq=300,
        updates_per_step=2,
        update_interval=1,
        min_replay_size=2000,
        per_alpha=0.2,
        per_beta_start=0.4,
        per_beta_increment=1e-4,
        per_priority_epsilon=1e-3,
        per_td_error_clip=2.0,
        reward_clip=10.0,
        grad_clip_norm=10.0,
        device=device,
    )

    print("预填充经验池...")
    for _ in range(200):
        states, _ = env.reset()
        actions = np.array([env.single_action_space.sample() for _ in range(num_envs)])
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        agent.store_batch_transitions(states, actions, rewards, next_states, terminated, truncated)

    print(f"\nCPU/GPU并行训练（{num_envs}个环境）...")
    reward_history = train_agent(
        env=env,
        agent=agent,
        num_episodes=3500,
        log_freq=50,
        algo_name="DuelingDQN",
        env_name="highway-v0",
        exploration_decay_episodes=1,
    )
    env.close()


if __name__ == "__main__":
    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()
