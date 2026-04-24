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


class EpsilonGreedy:
    def __init__(self, action_dim, device, init_eps=0.9, min_eps=0.01, decay_rate=0.995):
        self.action_dim = action_dim
        self.device = device
        self.eps = init_eps
        self.min_eps = min_eps
        self.decay_rate = decay_rate

    def select_action(self, q_net, states):
        states = np.asarray(states)
        single = False
        if states.ndim == 1:
            states = states[np.newaxis, ...]
            single = True

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = q_net(states_tensor)
        greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        if single:
            if np.random.rand() < self.eps:
                return int(np.random.randint(self.action_dim))
            return int(greedy_actions[0])

        random_mask = np.random.rand(states.shape[0]) < self.eps
        random_actions = np.random.randint(self.action_dim, size=states.shape[0])
        return np.where(random_mask, random_actions, greedy_actions)

    def decay(self):
        self.eps = max(self.min_eps, self.eps * self.decay_rate)

    def reset(self):
        self.decay()


class Boltzmann:
    def __init__(self, action_dim, device, init_tau=1.0, min_tau=0.1, decay_rate=0.99):
        self.action_dim = action_dim
        self.device = device
        self.tau = init_tau
        self.min_tau = min_tau
        self.decay_rate = decay_rate

    def select_action(self, q_net, states):
        states = np.asarray(states)
        single = False
        if states.ndim == 1:
            states = states[np.newaxis, ...]
            single = True

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = q_net(states_tensor)
        probs = torch.softmax(q_values / self.tau, dim=1).cpu().numpy()
        actions = np.array([np.random.choice(self.action_dim, p=p) for p in probs])
        return int(actions[0]) if single else actions

    def decay(self):
        self.tau = max(self.min_tau, self.tau * self.decay_rate)

    def reset(self):
        self.decay()


class ThompsonSampling:
    def __init__(self, action_dim, state_dim, device, noise_scale=0.1):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.noise_scale = noise_scale
        self.episode_action_count = np.zeros(action_dim)

    def select_action(self, q_net, states):
        states = np.asarray(states)
        single = False
        if states.ndim == 1:
            states = states[np.newaxis, ...]
            single = True

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = q_net(states_tensor)
        dynamic_noise = self.noise_scale / (1 + self.episode_action_count.mean() / 10.0)
        thompson_q = q_values + dynamic_noise * torch.randn_like(q_values)
        actions = thompson_q.argmax(dim=1).cpu().numpy()
        for action in actions:
            self.episode_action_count[int(action)] += 1
        return int(actions[0]) if single else actions

    def decay(self):
        self.episode_action_count = np.zeros(self.action_dim)

    def reset(self):
        self.decay()
