import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max
        self.current_step = 0

    def __call__(self):
        val = self.current
        self.current = self.bound(self.current + self.inc * self.current_step, self.end)
        return val

    def tick(self):
        self.current_step+=1

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class GNNCategoricalActor(Actor):

    def __init__(self, gnn_func, gnn_kwargs, device):
        super().__init__()
        self.logits_net = gnn_func(**gnn_kwargs).to(device)
        self.device = device

    def _distribution(self, obs):
        if isinstance(obs, Data):
            batch_obs = Batch.from_data_list([obs]).to(self.device)
            logits = self.logits_net(batch_obs)
            #print('probs:', torch.exp(logits) / torch.exp(logits).sum())
        else:
            #BATCH_SIZE=len(obs)
            #loader=DataLoader(obs, batch_size=BATCH_SIZE, shuffle=False)
            #for batch in loader:
            #    logits = self.logits_net(batch, BATCH_SIZE)
            batch_obs = Batch.from_data_list(obs).to(self.device)
            logits = self.logits_net(batch_obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class GNNCategoricalEpsActor(Actor):

    def __init__(self, gnn_func, gnn_kwargs, device):
        super().__init__()
        self.logits_net = gnn_func(**gnn_kwargs).to(device)
        self.device = device
        self.eps = LinearSchedule(0.9,0.00,20)
        print('caution: hard-coding max step in epsilon schedule')

    def _distribution(self, obs):
        if isinstance(obs, Data):
            batch_obs = Batch.from_data_list([obs]).to(self.device)
            logits = self.logits_net(batch_obs)
        else:
            batch_obs = Batch.from_data_list(obs).to(self.device)
            logits = self.logits_net(batch_obs)
        probs = torch.exp(logits)
        probs = probs / torch.sum(probs)
        eps = self.eps()
        probs_eps = (1-eps) * probs + eps * torch.ones_like(probs, device=self.device) / probs.shape[-1]
        return Categorical(probs = probs_eps)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def tick(self):
        self.eps.tick()


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.

class GNNCritic(nn.Module):

    def __init__(self, gnn_func, gnn_kwargs, device):
        super().__init__()
        self.v_net = gnn_func(**gnn_kwargs).to(device)
        self.device = device

    def forward(self, obs):
        if isinstance(obs, Data):
            obs_batch = Batch.from_data_list([obs]).to(self.device)
            values = torch.squeeze(self.v_net(obs_batch), -1) # Critical to ensure v has right shape.
        else:
            #v_begin_time = time.time()
            #BATCH_SIZE=len(obs)
            #loader=DataLoader(obs, batch_size=BATCH_SIZE, shuffle=False)
            #for batch in loader:
            #    values = torch.squeeze(self.v_net(batch, BATCH_SIZE), -1)
            obs_batch = Batch.from_data_list(obs).to(self.device)
            values = torch.squeeze(self.v_net(obs_batch), -1)
            #print(f'v iter: {time.time()-v_begin_time}')
        return values


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class GNNActorCritic(nn.Module):

    def __init__(self, gnn_func, pi_kwargs, v_kwargs, device):
        super().__init__()

        self.pi = GNNCategoricalActor(gnn_func, pi_kwargs, device)
        self.v = GNNCritic(gnn_func, v_kwargs, device)

    def step(self, obs, verbose=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            if(verbose):
                print('probs: ', pi.probs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def tick(self):
        pass

class GNNEpsActorCritic(nn.Module):

    def __init__(self, gnn_func, pi_kwargs, v_kwargs, device):
        super().__init__()

        self.pi = GNNCategoricalEpsActor(gnn_func, pi_kwargs, device)
        self.v = GNNCritic(gnn_func, v_kwargs, device)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def tick(self):
        self.pi.tick()
