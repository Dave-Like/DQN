import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from explore import Exploration
from ReplayBuffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        exploration_type="epsilon",
        buffer_capacity=100000,
        batch_size=256,
        gamma=0.99,
        lr=5e-4,
        target_update_freq=10,
        updates_per_step=1,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.updates_per_step = updates_per_step
        self.update_step = 0

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.explorer = Exploration.create(
            type=exploration_type,
            action_dim=action_dim,
            state_dim=state_dim,
            device=self.device,
        )

    def select_action(self, states):
        return self.explorer.select_action(self.q_net, states)

    def maybe_update(self):
        if len(self.memory) < self.batch_size:
            return

        for _ in range(self.updates_per_step):
            self.train_step()
            self.update_step += 1
            if self.update_step % self.target_update_freq == 0:
                self.update_target_network()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_exploration(self):
        if hasattr(self.explorer, "decay"):
            self.explorer.decay()
        elif hasattr(self.explorer, "reset"):
            self.explorer.reset()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def store_batch_transitions(self, states, actions, rewards, next_states, dones):
        self.memory.push_batch(states, actions, rewards, next_states, dones)
