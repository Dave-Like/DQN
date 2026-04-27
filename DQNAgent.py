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
        update_interval=1,
        min_replay_size=None,
        per_alpha=0.4,
        per_beta_start=0.6,
        per_beta_increment=5e-5,
        per_priority_epsilon=1e-6,
        per_td_error_clip=5.0,
        reward_clip=10.0,
        grad_clip_norm=10.0,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.updates_per_step = updates_per_step
        self.update_interval = max(1, update_interval)
        self.min_replay_size = max(self.batch_size, min_replay_size or (self.batch_size * 8))
        self.reward_clip = reward_clip
        self.grad_clip_norm = grad_clip_norm
        self.update_step = 0
        self.collect_step = 0

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta=per_beta_start,
            beta_increment_per_sampling=per_beta_increment,
            priority_epsilon=per_priority_epsilon,
            td_error_clip=per_td_error_clip,
        )
        self.explorer = Exploration.create(
            type=exploration_type,
            action_dim=action_dim,
            state_dim=state_dim,
            device=self.device,
        )

    def select_action(self, states):
        return self.explorer.select_action(self.q_net, states)

    def maybe_update(self):
        self.collect_step += 1
        if self.collect_step % self.update_interval != 0:
            return

        if len(self.memory) < self.min_replay_size:
            return

        for _ in range(self.updates_per_step):
            self.train_step()
            self.update_step += 1
            if self.update_step % self.target_update_freq == 0:
                self.update_target_network()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, terminateds, turncateds, indices, weights = self.memory.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = np.logical_or(terminateds, turncateds).astype(np.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        td = target_q - current_q
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        td_errors = td.detach().abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_exploration(self):
        if hasattr(self.explorer, "decay"):
            self.explorer.decay()
        elif hasattr(self.explorer, "reset"):
            self.explorer.reset()

    def store_transition(self, state, action, reward, next_state, terminated, turncated=False):
        self.memory.push(state, action, reward, next_state, terminated, turncated)

    def store_batch_transitions(self, states, actions, rewards, next_states, terminateds, turncateds=None):
        if turncateds is None:
            turncateds = np.zeros_like(terminateds, dtype=np.bool_)
        self.memory.push_batch(states, actions, rewards, next_states, terminateds, turncateds)
