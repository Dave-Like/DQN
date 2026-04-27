import numpy as np
from plot_utils import plot_rewards

def train_agent(env, agent, num_episodes, log_freq, algo_name, env_name, exploration_decay_episodes=None):
    reward_history = []
    total_episodes = 0
    completed_since_decay = 0
    decay_interval = exploration_decay_episodes or max(8, env.num_envs)
    episode_rewards = np.zeros(env.num_envs, dtype=np.float32)

    states, _ = env.reset()
    while total_episodes < num_episodes:
        actions = agent.select_action(states)
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        agent.store_batch_transitions(states, actions, rewards, next_states, terminated, truncated)
        agent.maybe_update()

        episode_rewards += np.array(rewards, dtype=np.float32)

        episode_completed = False
        for idx, done in enumerate(dones):
            if done:
                reward_history.append(float(episode_rewards[idx]))
                episode_rewards[idx] = 0.0
                total_episodes += 1
                completed_since_decay += 1
                episode_completed = True
                if total_episodes >= num_episodes:
                    break

        if episode_completed and completed_since_decay >= decay_interval:
            agent.decay_exploration()
            completed_since_decay = 0

        if total_episodes >= num_episodes:
            break

        states = next_states

        if len(reward_history) and total_episodes % log_freq == 0 and total_episodes == len(reward_history):
            explorer = getattr(agent, "explorer", None)
            epsilon = getattr(explorer, "eps", None)
            if epsilon is not None:
                print(f"Episode: {total_episodes} | Episode Reward Average: {np.mean(reward_history[-log_freq:]):.1f} | Epsilon: {epsilon:.3f}")
            else:
                print(f"Episode: {total_episodes} | Episode Reward Average: {np.mean(reward_history[-log_freq:]):.1f}")

    plot_rewards(
        rewards=reward_history,
        algo_name=algo_name,
        env_name=env_name,
        window=100,
        save_path=f"{algo_name}_{env_name}_rewards.png",
        show_plot=True
    )

    return reward_history