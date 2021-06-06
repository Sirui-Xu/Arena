import numpy as np
import random
from collections import namedtuple, deque
import os, sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader

dqn_root = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(dqn_root))
sys.path.append(project_root)
sys.path.append(dqn_root)
dataset_root = dqn_root + '/saved_data'

from models import *
from data_utils import ReplayGraphDataset, input_to_graph_input, parse_edge_outputs

Qnet_fn=TBD

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

class DQGNN_agent():

    def __init__(self, state_input_dim, state_output_dim, device, seed):
        self.state_input_dim = state_input_dim
        self.state_output_dim = state_output_dim
        self.device=device
        self.seed = random.seed(seed)
        torch.manual_seed(seed)

        self.qnetwork_local = Qnet_fn(state_input_dim, state_output_dim).to(device)
        self.qnetwork_target = Qnet_fn(state_input_dim, state_output_dim).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state_dict, action, reward, next_state_dict, done):
        # Save experience in replay memory
        self.memory.add(state_dict, action, reward, next_state_dict, done)

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
        graph = state.to_graph_input(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            best_action = self.qnetwork_local(graph, mode='argmax')
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return best_action.cpu().data.numpy(), 'best'
        else:
            return random.choice(np.arange(state.get_shape()[0])), 'random'

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        graphs, actions, rewards, next_graphs, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_graphs, mode='max').detach().unsqueeze(dim=1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(graphs, actions, mode='action')
        #print('actions:\n', actions)
        #print('Q_targets:\n',Q_targets)
        #print('Q_expected:\n', Q_expected)
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done",
                                                                "edge_num", "node_num", "edge_index", "edge_attr"])
        self.device=device
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #print('adding to replay buffer: \nstate=%s\nnext_state=%s\naction=%d, reward=%f'%(state.squeeze(), next_state.squeeze(), action, reward))
        state, edge_num, node_num, edge_index, edge_attr = state.list_comprehension()
        next_state, _, _, _, _ = next_state.list_comprehension()

        e = self.experience(np.expand_dims(state, axis=-1), action, reward,
                            np.expand_dims(next_state, axis=-1), done,
                            edge_num, node_num, edge_index, edge_attr)
        self.memory.append(e)

    def pack_batch_input(self, edge_nums, node_nums, states, edge_indexes, edge_attrs):
        batch_dataset = ReplayGraphDataset(dataset_root,
                                           states, edge_nums, node_nums, edge_indexes, edge_attrs)

        loader = DataLoader(batch_dataset, batch_size=batch_dataset.__len__())
        datas = [data for data in loader]
        return datas[0]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        edge_nums = torch.cat([torch.tensor([e.edge_num]).long().to(self.device) for e in experiences if e is not None])
        node_nums = torch.cat([torch.tensor([e.node_num]).long().to(self.device) for e in experiences if e is not None])
        edge_indexes = [torch.from_numpy(e.edge_index).long().to(self.device) for e in experiences if e is not None]
        edge_attrs = [torch.from_numpy(e.edge_attr).float().to(self.device) for e in experiences if e is not None]
        states = [torch.from_numpy(e.state).float().to(self.device) for e in experiences if e is not None]
        graphs = self.pack_batch_input(edge_nums, node_nums, states, edge_indexes, edge_attrs)

        next_states = [torch.from_numpy(e.next_state).float().to(self.device) for e in experiences if e is not None]
        next_graphs = self.pack_batch_input(edge_nums, node_nums, next_states, edge_indexes, edge_attrs)


        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
        #    device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (graphs, actions, rewards, next_graphs, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)