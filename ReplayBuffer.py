import random
import numpy as np
import pickle
from collections import deque
from abc import ABC, abstractmethod
#虚基类
class BaseBuffer(ABC):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


#均匀实现
class ReplayBuffer(BaseBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )

