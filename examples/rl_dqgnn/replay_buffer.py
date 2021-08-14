from collections import deque, namedtuple
import random
from torch_geometric.data import Data, Batch
import torch
import numpy as np
from examples.rl_dqgnn.sum_tree import SumTree

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device=device
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = Batch.from_data_list([e.state for e in experiences])
        next_states = Batch.from_data_list([e.next_state for e in experiences])
        actions = torch.tensor([e.action for e in experiences]).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences]).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:

    def __init__(self, buffer_size, batch_size, device, seed):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.prio_exp = namedtuple("PrioExp", field_names=["exp", "sampling_prob", "idx"])
        self.device=device
        self.seed = random.seed(seed)
        self.max_priority = 1
        self.cnt=0

    def add(self, state, action, reward, next_state, done):
        self.cnt+=1
        self.tree.add(self.max_priority, self.experience(state, action, reward, next_state, done))

    def sample(self):
        batch_size=self.batch_size

        segment = self.tree.total() / batch_size

        prio_exps = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data_index) = self.tree.get(s)
            prio_exp = self.prio_exp(self.tree.data[data_index], p / self.tree.total(), idx)
            prio_exps.append(prio_exp)
        while len(prio_exps) < batch_size:
            # This should rarely happen
            prio_exps.append(random.choice(prio_exps))

        states = Batch.from_data_list([e.exp.state for e in prio_exps])
        next_states = Batch.from_data_list([e.exp.next_state for e in prio_exps])
        actions = torch.tensor([e.exp.action for e in prio_exps]).to(self.device)
        rewards = torch.tensor([e.exp.reward for e in prio_exps]).to(self.device)
        dones = torch.tensor([e.exp.done for e in prio_exps]).float().to(self.device)
        sampling_probs = torch.tensor([e.sampling_prob for e in prio_exps]).float().to(self.device)
        idxs = np.array([e.idx for e in prio_exps])
        return states, actions, rewards, next_states, dones, sampling_probs, idxs

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.cnt