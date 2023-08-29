import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from gymnasium.spaces import Tuple, Box, Discrete

from envs import BaseArmEnv

MIN_PROB = torch.tensor(1e-12) 

# lower bound probability to avoid issues from computing the log of 0
def clamp_prob(inp):
    return torch.clamp(inp, min = MIN_PROB)

def KNN(data, labels, new_point, k):
    dists = np.linalg.norm(data - new_point, axis = 1)
    top_idxs = np.argsort(dists)[:k]
    votes = labels[top_idxs]
    vals, counts = np.unique(votes, return_counts = True)
    return vals[np.argmax(counts)]

def sum_to_one(lst):
    return np.array(lst) / np.sum(lst) # returns to an array that sums to 1 (useful for getting probabilities)

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
def validate_space(space, is_obs_space):
    assert isinstance(space, Tuple) and len(space.spaces) == 2, \
        f"The space {space} should include both continuous and discrete components"
    box, discrete = space.spaces
    assert isinstance(box, Box) and len(box.shape) == 1, "The first subspace should be a continuous vector"
    assert is_obs_space or (np.all(box.high < np.inf) and np.all(box.low == -box.high)), \
        "The continuous action space must be bounded and symmetric around 0"
    assert isinstance(discrete, Discrete) and discrete.n == 2, "The second subspace should be a discrete binary"
    return True

def get_space_dim(space):
    return space.spaces[0].shape[0] + 1

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        continuous_act_dim = act_dim - 1
        self.action_split = [continuous_act_dim, 1]

        self.mu_layer = nn.Linear(hidden_sizes[-1], continuous_act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], continuous_act_dim)
        self.logit_layer = nn.Linear(hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.act_limit = torch.tensor(act_limit)

    def forward(self, obs, deterministic = False, with_logprob = True):
        if torch.any(torch.isnan(obs)):
            print('obs', obs)
        net_out = self.net(obs)
        if torch.any(torch.isnan(net_out)):
            print('net_out', net_out)
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
            discrete_action = r < p # sample 1 with probability p

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = continuous_dist.log_prob(continuous_action).sum(axis = -1)
            logp_pi -= 2 * (np.log(2) - continuous_action - F.softplus(-2 * continuous_action)).sum(axis = 1)
            logp_discrete = torch.where(discrete_action, torch.log(clamp_prob(p)), torch.log(clamp_prob(1 - p)))
            logp_pi += torch.squeeze(logp_discrete, dim = -1)
        else:
            logp_pi = None
        continuous_action = self.act_limit * torch.tanh(continuous_action)
        return torch.cat((continuous_action, discrete_action.int()), dim = -1), logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim = -1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class VanillaActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes = (256, 256), activation = nn.ReLU):
        super().__init__()
        validate_space(observation_space, is_obs_space = True)
        validate_space(action_space, is_obs_space = False)

        # build policy and value functions
        obs_dim = get_space_dim(observation_space)
        act_dim = get_space_dim(action_space)
        continuous_act_dim = act_dim - 1
        self.action_split = [continuous_act_dim, 1]
        act_limit = action_space[0].high
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, act_limit, hidden_sizes, activation)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    @torch.no_grad()
    def action(self, observation, deterministic = False):
        action, _ = self.pi(observation, deterministic, with_logprob = False)
        continuous_action, discrete_action = torch.split(action, self.action_split)
        return (continuous_action.numpy(), discrete_action.bool()[0].item())


class MultiActorCritic(nn.Module):
    def __init__(self, envs, num_subcontrollers, hidden_sizes = (256, 256), activation = nn.ReLU):
        super().__init__()
        assert isinstance(num_subcontrollers, int) and num_subcontrollers > 1
        self.num_subcontrollers = num_subcontrollers
        # get dimensions from first environment and then check that they all fit the same base environment
        env = envs[0]
        validate_space(env.observation_space, is_obs_space = True)
        validate_space(env.action_space, is_obs_space = False)
        self.obs_dim = get_space_dim(env.observation_space)
        self.act_dim = get_space_dim(env.action_space)
        continuous_act_dim = self.act_dim - 1
        self.action_split = [continuous_act_dim, 1]
        act_limit = env.action_space[0].high
        assert all(isinstance(env, BaseArmEnv) for env in envs), f'Invalid environments: {envs}'

        num_envs = len(envs)
        self.pis = [SquashedGaussianMLPActor(self.obs_dim, self.act_dim, act_limit, hidden_sizes, activation)
                        for _ in range(self.num_subcontrollers)]
        self.q1s = [MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation) for _ in range(num_envs)]
        self.q2s = [MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation) for _ in range(num_envs)]

    # expand an observation to have an extra initial dimension of size num_subcontrollers
    # this is meant to align the shape of observations and candidate actions
    def expand_observation(self, obs):
        other_dims = [-1] * len(obs.shape)
        return torch.unsqueeze(obs, dim = 0).expand(self.num_subcontrollers, *other_dims)

    @torch.no_grad() # this function is not meant for batch computations because of the final packaging step
    def action(self, observation, env_index, deterministic = False):
        assert observation.shape == (self.obs_dim,), f'Invalid shape for observation: {observation.shape}'

        observation = torch.unsqueeze(observation, dim = 0)
        # sample candidate actions from all subcontrollers
        candidates = torch.stack([pi(observation, deterministic, with_logprob = False)[0] for pi in self.pis])
        subcontroller_index = self.select_subcontroller(observation, env_index, candidates, deterministic)[0].item()
        action = candidates[subcontroller_index, 0]
        continuous_action, discrete_action = torch.split(action, self.action_split) # package the chosen action
        return (continuous_action.numpy(), discrete_action.bool()[0].item()), subcontroller_index

    @torch.no_grad() # use q values as logits to select a subcontroller based on proposed actions
    def select_subcontroller(self, observations, env_index, candidates, deterministic = False):
        assert len(observations.shape) == 2 and observations.shape[1] == self.obs_dim, \
            f'Invalid shape for observations: {observations.shape}'
        assert len(candidates.shape) == 3 and candidates.shape[0] == self.num_subcontrollers and \
            candidates.shape[2] == self.act_dim, f'Invalid shape for candidates: {candidates.shape}'
        assert observations.shape[0] == candidates.shape[1], 'Batch sizes must be aligned'

        observations = self.expand_observation(observations) # s x n x o (candidates is s x n x a)
        q1_vals = self.q1s[env_index](observations, candidates) # s x n
        q2_vals = self.q2s[env_index](observations, candidates) # s x n
        logits, _ = torch.min(torch.stack((q1_vals, q2_vals)), dim = 0) # s x n
        if deterministic:
            return torch.argmax(logits, dim = 0) # n,
        else:
            # multinomial requires that the rows contain probabilities, so we have to transpose the tensor first
            probs = torch.softmax(logits, dim = 0).T # n x s
            return torch.multinomial(probs, 1).flatten() # n,

    @torch.no_grad()
    def next_action_for_backup(self, observations, env_index):
        assert len(observations.shape) == 2 and observations.shape[1] == self.obs_dim, \
            f'Invalid shape for observations: {observations.shape}'

        n = observations.shape[0] # batch size
        candidates, logprobs = zip(*[pi(observations) for pi in self.pis])
        candidates, logprobs = torch.stack(candidates), torch.stack(logprobs) # s x n x a, s x n
        indices = self.select_subcontroller(observations, env_index, candidates)
        return candidates[indices, torch.arange(n)], logprobs[indices, torch.arange(n)] # n x a, n


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents. This buffer is based on the latest Gym API
    which differentiates between terminated and truncated episodes. However, the buffer will only track
    terminated episodes because our environments do not truncate.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.observation_buffer = np.zeros(get_combined_shape(size, obs_dim), dtype = np.float32)
        self.next_observation_buffer = np.zeros(get_combined_shape(size, obs_dim), dtype = np.float32)
        self.action_buffer = np.zeros(get_combined_shape(size, act_dim), dtype = np.float32)
        self.reward_buffer = np.zeros(size, dtype = np.float32)
        self.terminated_buffer = np.zeros(size, dtype = np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, observation, action, reward, next_observation, terminated):
        self.observation_buffer[self.ptr] = observation
        self.next_observation_buffer[self.ptr] = next_observation
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.terminated_buffer[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size = 32):
        return self.package_batch(idxs = np.random.randint(0, self.size, size = batch_size))

    def package_batch(self, idxs):
        batch = dict(observation = self.observation_buffer[idxs], 
                     next_observation = self.next_observation_buffer[idxs], 
                     action = self.action_buffer[idxs], 
                     reward = self.reward_buffer[idxs], 
                     terminated = self.terminated_buffer[idxs])
        return {k: torch.as_tensor(v, dtype = torch.float32) for k, v in batch.items()}


class SubcontrollerReplayBuffer(ReplayBuffer):
    '''An extension of the ReplayBuffer that also stores the index of the subcontroller that is used'''

    def __init__(self, obs_dim, act_dim, num_subcontrollers, size):
        super().__init__(obs_dim, act_dim, size)
        self.subcontroller_index_buffer = np.zeros(size, dtype = np.int16)
        # the ith element is a list containing the indices of the environment steps where the ith subcontroller was used
        self.steps_by_subcontroller = [[] for _ in range(num_subcontrollers)]

    def store(self, observation, action, reward, next_observation, terminated, subcontroller_index):
        # special case for early steps where we aren't assigning subcontrollers yet
        if subcontroller_index == -1:
            super().store(observation, action, reward, next_observation, terminated)
            return

        # we're overwriting old data if the buffer's full, so we should update the subcontroller records accordingly
        if self.size == self.max_size:
            prev_subcontroller_index = self.subcontroller_index_buffer[self.ptr]
            self.steps_by_subcontroller[prev_subcontroller_index].remove(self.ptr)
        self.subcontroller_index_buffer[self.ptr] = subcontroller_index
        self.steps_by_subcontroller[subcontroller_index].append(self.ptr)
        # N.B. the super call must come after the subcontroller logic because the super call will increment the ptr
        super().store(observation, action, reward, next_observation, terminated)

    def sample_q_batch(self, batch_size = 32):
        return super().sample_batch(batch_size = batch_size)

    def sample_pi_batch(self, subcontroller_index, batch_size):
        if batch_size == 0:
            return None # empty batch
        list_idxs = np.random.randint(0, len(self.steps_by_subcontroller[subcontroller_index]), size = batch_size)
        # reindex into the list containing steps that belong to the given subcontroller; we're indexing into
        # a list instead of an array, so this list comprehension is a workaround to mimic array indexing
        buffer_idxs = np.array([self.steps_by_subcontroller[subcontroller_index][i] for i in list_idxs])
        return self.package_batch(buffer_idxs)

    def num_steps(self, subcontroller_index):
        return len(self.steps_by_subcontroller[subcontroller_index])