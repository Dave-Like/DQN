import os
import gymnasium as gym
import highway_env
import torch
from gymnasium.wrappers import FlattenObservation
from gymnasium.vector import AsyncVectorEnv
from DQNAgent import DQNAgent
from trainer import train_agent

NUM_ENVS = min(8, max(1, (os.cpu_count() or 2) // 2))


def create_highway_env():
    env = gym.make("highway-v0")
    env.unwrapped.config.update({
        "vehicles_count": 15,
        "duration": 80,
        "normalize_reward": False,
        "collision_reward": -80,
        "high_speed_reward": 3,
        "lane_change_reward": 0.2,
        "policy_frequency": 5,
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
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        exploration_type="epsilon",
        buffer_capacity=400000,
        batch_size=512,
        gamma=0.95,
        lr=1e-3,
        target_update_freq=500,
        updates_per_step=1,
        device=device,
    )

    print("预填充经验池...")
    for _ in range(200):
        states, _ = env.reset()
        actions = env.action_space.sample()
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        dones = [t or tr for t, tr in zip(terminated, truncated)]
        agent.store_batch_transitions(states, actions, rewards, next_states, dones)

    print(f"\nCPU/GPU并行训练（{num_envs}个环境）...")
    reward_history = train_agent(
        env=env,
        agent=agent,
        num_episodes=5000,
        log_freq=50,
        algo_name="DQN",
        env_name="highway-v0",
    )
    env.close()


if __name__ == "__main__":
    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()
