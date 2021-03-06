import numpy as np
import random
from collections import namedtuple, deque
import os, sys
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader, Batch
from torch.utils.data import Dataset
from examples.rl_dqgnn.replay_buffer import *


dqn_root = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(dqn_root))
sys.path.append(project_root)
sys.path.append(dqn_root)


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
#LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
#TARGET_UPDATE_FREQ = 500


class DQGNN_agent():

    def __init__(self, qnet_local, qnet_target, lr,
                 hard_update, target_update_freq, double_q, PER,
                 replay_eps, replay_alpha, replay_beta, device, seed):
        self.LR=lr
        self.hard_update=hard_update
        self.target_update_freq = target_update_freq
        self.double_q = double_q
        self.PER = PER
        self.replay_eps = replay_eps
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.device=device
        self.seed = random.seed(seed)
        torch.manual_seed(seed)

        self.qnetwork_local = qnet_local.to(device)
        self.qnetwork_target = qnet_target.to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        if not PER:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed)
        else:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        if self.hard_update:
            if self.t_step % self.target_update_freq ==0:
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        self.qnetwork_local.eval()
        # Batch the inputs.

        #state = copy.deepcopy(state)
        #state.batch = torch.zeros(len(state.x)).long()
        #state = state.to(self.device)
        state=Batch.from_data_list([state]).to(self.device)
        with torch.no_grad():
            best_action = self.qnetwork_local(state).argmax() # Batch size is always one.
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return best_action.cpu().item(), 'best'
        else:
            return random.choice(np.arange(6)), 'random' # Hard coded!

    def learn(self, experiences, gamma):
        if not self.PER:
            graphs, actions, rewards, next_graphs, dones = experiences
        else:
            graphs, actions, rewards, next_graphs, dones, sampling_probs, idxs = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(
                next_graphs.to(self.device)).detach()
        if not self.double_q:
            Q_targets_next = Q_targets_next.max(1)[0]
        else:
            best_actions = torch.argmax(self.qnetwork_local(next_graphs.to(self.device)), dim=-1)
            Q_targets_next = Q_targets_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(graphs.to(self.device))
        Q_expected = Q_expected.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = Q_targets - Q_expected

        if self.PER:
            priorities = loss.abs().add(self.replay_eps).pow(self.replay_alpha)
            self.memory.update_priorities(zip(idxs, priorities.cpu().detach().numpy()))
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)
        loss = loss.pow(2).mul(0.5).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if not self.hard_update:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

