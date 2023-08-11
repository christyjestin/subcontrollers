import numpy as np
import scipy.signal

from gymnasium.spaces import Tuple, Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def get_combined_shape(length, shape = None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation = nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if (j < len(sizes) - 2) else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def as_vector(tup):
    assert isinstance(tup, tuple) and len(tup) == 2
    return np.concatenate((tup[0], np.array([tup[1]])))

# ensures that the input is a composite Tuple space consisting of a Box 1D continuous and a single Discrete space
def validate_space(space):
    assert isinstance(space, Tuple) and len(space.spaces) == 2, \
        f"The space {space} should include both continuous and discrete components"
    box, discrete = space.spaces
    assert isinstance(box, Box) and len(box.shape) == 1, "The first subspace should be a continuous vector"
    assert np.all(box.high < np.inf) and np.all(box.low == -box.high), \
        "The continuous space must be bounded and symmetric around 0"
    assert isinstance(discrete, Discrete) and discrete.n == 2, "The second subspace should be a discrete binary"
    return True

def get_space_dim(space):
    assert validate_space(space), "The space must be of the form described above in validate_space"
    return space.spaces[0].shape[0] + 1

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([get_space_dim(observation_space)] + list(hidden_sizes), activation, activation)
        action_box = action_space.spaces[0]
        num_continuous_inputs = action_box.shape[0]
        self.action_split = [num_continuous_inputs, 1]

        self.mu_layer = nn.Linear(hidden_sizes[-1], num_continuous_inputs)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], num_continuous_inputs)
        self.logit_layer = nn.Linear(hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.act_limit = action_box.high

    def forward(self, obs, deterministic = False, with_logprob = True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        p = self.sigmoid(self.logit_layer(net_out))

        # Pre-squash distribution and sample
        continuous_dist = Normal(mu, std)
        # Deterministic case is only used for evaluating policy at test time
        continuous_action = mu if deterministic else continuous_dist.rsample()

        with torch.no_grad():
            r = 0.5 if deterministic else torch.rand_like(p)
            discrete_action = (r < p).int() # sample 1 with probability p

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = continuous_dist.log_prob(continuous_action).sum(axis = -1)
            logp_pi -= 2 * (np.log(2) - continuous_action - F.softplus(-2 * continuous_action)).sum(axis = 1)
            logp_pi += discrete_action * torch.log(p) + (1 - discrete_action) * torch.log(1 - p)
        else:
            logp_pi = None

        continuous_action = self.act_limit * torch.tanh(continuous_action)
        return torch.cat((continuous_action, discrete_action)), logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim = -1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes = (256, 256), activation = nn.ReLU):
        super().__init__()
        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    @torch.no_grad()
    def action(self, observation, deterministic = False):
        action, _ = self.pi(observation, deterministic, with_logprob = False)
        continuous_action, discrete_action = torch.split(action, [2, 1])
        return (continuous_action.numpy(), discrete_action.bool()[0].item())