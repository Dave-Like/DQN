import numpy as np
from plot_utils import plot_rewards  

def train_agent(
    env,
    agent,
    num_episodes=1100,
    log_freq=20,
    algo_name="DQN",
    env_name="highway-v0",
    render=False  
):
    reward_history = []
    
    for episode in range(num_episodes):
        # 重置环境
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            if render:
                env.render()
            # 选动作
            action = agent.select_action(state)
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            # 训练
            agent.train_step()

            # 更新状态
            state = next_state
            episode_reward += reward

        # 探索策略衰减
        agent.decay_exploration()

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        reward_history.append(episode_reward)
        
        if episode % log_freq == 0:
            print(f"Episode: {episode:3d} | Reward: {episode_reward:6.1f} ")

    env.close()
    plot_rewards(reward_history, algo_name, env_name)  

    return reward_history