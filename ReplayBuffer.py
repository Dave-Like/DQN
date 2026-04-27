import random
import numpy as np
import pickle
from abc import ABC, abstractmethod


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def add(self, priority, data):
        tree_index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_index, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, tree_index, value):
        left = 2 * tree_index + 1
        right = left + 1

        if left >= len(self.tree):
            return tree_index

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    def get(self, value):
        tree_index = self._retrieve(0, value)
        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class BaseBuffer(ABC):
    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def push(self, state, action, reward, next_state, terminated, turncated):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def update_priorities(self, indices, priorities):
        pass

    def __len__(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        capacity,
        alpha=0.2,
        beta=0.4,
        beta_increment_per_sampling=1e-4,
        priority_epsilon=1e-3,
        td_error_clip=2.0,
    ):
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.priority_epsilon = priority_epsilon
        self.td_error_clip = td_error_clip
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, terminated, turncated):
        transition = (state, action, reward, next_state, terminated, turncated)
        self.tree.add(self.max_priority ** self.alpha, transition)

    def push_batch(self, states, actions, rewards, next_states, terminateds, turncateds):
        for state, action, reward, next_state, terminated, turncated in zip(
            states, actions, rewards, next_states, terminateds, turncateds
        ):
            self.push(state, action, reward, next_state, terminated, turncated)

    def sample(self, batch_size, beta=None):
        if self.tree.n_entries == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        total_priority = float(self.tree.total_priority)
        if total_priority <= 0.0:
            raise ValueError("Cannot sample when total priority is non-positive.")

        beta = self.beta if beta is None else beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        indices = []
        priorities = []
        batch = []
        segment = total_priority / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)
            value = random.random() * (end - start) + start
            value = min(value, total_priority - 1e-8)
            tree_index, priority, data = self.tree.get(value)

            if data is None or priority <= 0.0:
                found_valid = False
                for _ in range(8):
                    retry_value = random.random() * total_priority
                    tree_index, priority, data = self.tree.get(retry_value)
                    if data is not None and priority > 0.0:
                        found_valid = True
                        break
                if not found_valid:
                    raise ValueError("Failed to draw a valid prioritized sample from SumTree.")

            indices.append(tree_index)
            priorities.append(priority)
            batch.append(data)

        sampling_probabilities = np.array(priorities, dtype=np.float32) / total_priority
        sampling_probabilities = np.clip(sampling_probabilities, self.priority_epsilon, None)
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        max_weight = float(np.max(weights))
        if not np.isfinite(max_weight) or max_weight <= 0.0:
            weights = np.ones_like(weights, dtype=np.float32)
        else:
            weights /= max_weight

        states, actions, rewards, next_states, terminateds, turncateds = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(terminateds, dtype=np.bool_),
            np.array(turncateds, dtype=np.bool_),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities):
        priorities = np.abs(np.asarray(priorities, dtype=np.float32))
        priorities = np.minimum(priorities, self.td_error_clip) + self.priority_epsilon
        if priorities.size:
            self.max_priority = max(self.max_priority, float(priorities.max()))
        for tree_index, priority in zip(indices, priorities):
            self.tree.update(tree_index, float(priority ** self.alpha))

    def __len__(self):
        return self.tree.n_entries

    def clear(self):
        self.tree = SumTree(self.capacity)
