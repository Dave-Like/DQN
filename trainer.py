import numpy as np
from plot_utils import plot_rewards

def train_agent(env, agent, num_episodes, log_freq, algo_name, env_name):
    reward_history = []
    total_episodes = 0
    episode_rewards = np.zeros(env.num_envs, dtype=np.float32)

    states, _ = env.reset()
    while total_episodes < num_episodes:
        actions = agent.select_action(states)
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        agent.store_batch_transitions(states, actions, rewards, next_states, dones)
        agent.maybe_update()

        episode_rewards += np.array(rewards, dtype=np.float32)

        for idx, done in enumerate(dones):
            if done:
                reward_history.append(float(episode_rewards[idx]))
                episode_rewards[idx] = 0.0
                total_episodes += 1
                agent.decay_exploration()
                if total_episodes >= num_episodes:
                    break

        if total_episodes >= num_episodes:
            break

        states = next_states

        if len(reward_history) and total_episodes % log_freq == 0 and total_episodes == len(reward_history):
            print(f"Episode: {total_episodes} | Avg Reward: {np.mean(reward_history[-log_freq:]):.1f}")

    plot_rewards(
        rewards=reward_history,
        algo_name=algo_name,
        env_name=env_name,
        window=100,
        save_path=f"{algo_name}_{env_name}_rewards.png",
        show_plot=True
    )

    return reward_history