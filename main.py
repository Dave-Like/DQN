import gymnasium as gym
import highway_env
import torch
from gymnasium.wrappers import FlattenObservation
from highway_env.envs import HighwayEnv
from DQNAgent import DQNAgent
from trainer import train_agent

def main():

    env = HighwayEnv()
    
    env.config.update({
        "offscreen_rendering": False,  
        "render_fps": 0,               
        "policy_frequency": 1,         
        "vehicles_count": 10,          
        "duration": 40,                
        "reward_speed_range": [20, 30],
        "collision_reward": -100,      
        "high_speed_reward": 1,        
        "lane_centering_reward": 0.5,  
    })
    env.reset()  

    env = FlattenObservation(env)  
    state_dim = env.observation_space.shape[0]  
    action_dim = env.action_space.n             

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,                
        exploration_type="epsilon",    
        buffer_capacity=100000,        
        batch_size=256,                 
        gamma=0.99,                    
        lr=5e-4,                       
        target_update_freq=10,         
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("\nbegin")
    reward_history = train_agent(
        env=env,
        agent=agent,
        num_episodes=500,             
        log_freq=50,                   
        algo_name="DQN",               
        env_name="highway-v0",         
        render=False                  
    )


if __name__ == "__main__":
    main()