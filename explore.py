import numpy as np
import torch

class Exploration:
    @staticmethod
    def create(type, action_dim, state_dim, device):  
        if type == "epsilon":
            return EpsilonGreedy(action_dim, device)  
        elif type == "boltzmann":
            return Boltzmann(action_dim, device)      
        elif type == "thompson":
            return ThompsonSampling(action_dim, state_dim, device)
        else:
            raise ValueError(f"fault：{type}")

# 1. epsilon-greedy
class EpsilonGreedy:
    def __init__(self, action_dim, device, init_eps=0.9, min_eps=0.01, decay_rate=0.995):
        self.action_dim = action_dim
        self.device = device  
        self.eps = init_eps
        self.min_eps = min_eps
        self.decay_rate = decay_rate

    def select_action(self, q_net, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = q_net(state)
            return q_values.argmax().item()

    def reset(self):
        self.eps = max(self.min_eps, self.eps * self.decay_rate)

# 2. Boltzmann
class Boltzmann:
    def __init__(self, action_dim, device, init_tau=1.0, min_tau=0.1, decay_rate=0.99):
        self.action_dim = action_dim
        self.device = device  
        self.tau = init_tau
        self.min_tau = min_tau
        self.decay_rate = decay_rate

    def select_action(self, q_net, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = q_net(state)
        probs = torch.softmax(q_values / self.tau, dim=1)
        action = np.random.choice(self.action_dim, p=probs.detach().cpu().numpy()[0])
        return action

    def reset(self):
        self.tau = max(self.min_tau, self.tau * self.decay_rate)


class ThompsonSampling:
    def __init__(self, action_dim, state_dim, device, noise_scale=0.1):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.noise_scale = noise_scale  
        self.episode_action_count = np.zeros(action_dim)  

    def select_action(self, q_net, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = q_net(state)  
            dynamic_noise = self.noise_scale / (1 + self.episode_action_count.mean()/10)
            thompson_q = q_values + dynamic_noise * torch.randn_like(q_values)
        
        action = thompson_q.argmax(dim=1).item()
        return action

    def reset(self):
        self.episode_action_count = np.zeros(self.action_dim)

