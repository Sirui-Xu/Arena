import numpy as np
import random
from collections import namedtuple, deque
import os, sys
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset


dqn_root = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(dqn_root))
sys.path.append(project_root)
sys.path.append(dqn_root)


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network


class DQGNN_agent():

    def __init__(self, qnet_local, qnet_target, device, seed):
        self.device=device
        self.seed = random.seed(seed)
        torch.manual_seed(seed)

        self.qnetwork_local = qnet_local.to(device)
        self.qnetwork_target = qnet_target.to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        self.qnetwork_local.eval()
        # Batch the inputs.
        state = copy.deepcopy(state)
        state.batch = torch.zeros(len(state.x)).long()
        state = state.to(self.device)
        with torch.no_grad():
            best_action = self.qnetwork_local(state, 1).argmax() # Batch size is always one.
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return best_action.cpu().item(), 'best'
        else:
            return random.choice(np.arange(6)), 'random' # Hard coded!

    def learn(self, experiences, gamma):
        graphs, actions, rewards, next_graphs, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(
            next_graphs.to(self.device), BATCH_SIZE).max(1)[0].detach()

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(graphs.to(self.device), BATCH_SIZE)
        Q_expected = Q_expected.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        states = next(iter(DataLoader(
            GraphDataset([e.state for e in experiences]), batch_size=self.batch_size, shuffle=False)))
        next_states = next(iter(DataLoader(
            GraphDataset([e.next_state for e in experiences]), batch_size=self.batch_size, shuffle=False)))
        actions = torch.tensor([e.action for e in experiences]).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences]).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)