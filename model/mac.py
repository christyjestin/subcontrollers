from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import wandb

from model.core import *

class MAC:
    """
    Multi Actor-Critic (SAC)
    Notes:
        - Entropy
            This implementation only considers the entropy of each subcontroller individually, rather than the overall
            entropy of the resulting Multi Actor Critic controller. Practically this means that we consider the log
            probability of an action from a certain subcontroller rather than its overall log probability (marginalizing
            over all subcontrollers) or the log probability of choosing both the subcontroller and the action. The logic
            behind this is that the probability of choosing a subcontroller is entirely defined by the Q values and the
            temperature, and we don't want entropy calculations to interfere with learning the Q function: thus the
            entropy calculations must not consider the probability of choosing a subcontroller.

    Args:
        env_fns: A list of functions that create a copy of the environment. The environment must satisfy the OpenAI Gym API.

        exp_name (str): The name of the experiment for WandB logging

        num_subcontrollers (int): The number of subcontrollers used in this MultiActorCritic setup

        actor_critic: The constructor method for a PyTorch Module with an ``act`` method, a ``pi`` module, a ``q1``
            module, and a ``q2`` module. The ``act`` method and ``pi`` module should accept batches of observations
            as inputs, and ``q1`` and ``q2`` should accept a batch of observations and a batch of actions as inputs.
            When called, ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each observation.
            ``q1``       (batch,)          | Tensor containing one current estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current estimate of Q* for the provided 
                                           | observations and actions. (Critical: make sure to flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policygiven observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of actions in ``a``. Importantly:
                                           | gradients should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) for the agent and the 
            environment in each epoch.

        num_epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target networks. Target networks are 
            updated towards main networks according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection, before running real policy.
            Helps exploration.

        update_after (int): Number of env interactions to collect before starting to do gradient descent 
        updates. Ensures replay buffer is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse between gradient descent updates. 
        Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps 
        is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic policy at the end of each epoch.

        save_freq (int): How often (in terms of gap between epochs) to save the current policy and value function.
    """

    def __init__(self, env_fns, exp_name, num_subcontrollers, actor_critic = MultiActorCritic, ac_kwargs = dict(), 
                 seed = 0, steps_per_epoch = 4000, num_epochs = 100, replay_size = int(1e6), gamma = 0.99, 
                 polyak = 0.995, lr = 1e-3, alpha = 0.2, batch_size = 100, start_steps = 10000, update_after = 1000, 
                 update_every = 50, num_test_episodes = 10, save_freq = 1):
        self.num_envs = len(env_fns)
        self.num_subcontrollers = num_subcontrollers
        self.exp_name = exp_name
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.save_freq = save_freq

        self.logger = wandb.init(project = "subcontrollers", name = exp_name, config = locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.envs, self.test_envs = [env_fn() for env_fn in env_fns], [env_fn() for env_fn in env_fns]

        self.ac = actor_critic(self.envs, self.num_subcontrollers, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(self.ac.q1s.parameters(), self.ac.q2s.parameters())
        obs_dim = get_space_dim(self.envs[0].observation_space)
        act_dim = get_space_dim(self.envs[0].action_space)
        # Each environment has its own buffer
        self.replay_buffers = [SubcontrollerReplayBuffer(obs_dim = obs_dim, act_dim = act_dim, size = replay_size) 
                                for _ in range(self.num_envs)]
        counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.alert(title = 'Number of parameters', text = f'pi:{counts[0]}, q1: {counts[1]}, q2: {counts[2]}', 
                          level = 'INFO')

        self.pi_optimizer = Adam(self.ac.pis.parameters(), lr = lr)
        self.q_optimizer = Adam(self.q_params, lr = lr)

    def compute_loss_q(self, data, task_index):
        observation, action, reward, next_observation, terminated = data['observation'], data['action'], \
                                                data['reward'], data['next_observation'], data['terminated']

        q1 = self.ac.q1s[task_index](observation, action)
        q2 = self.ac.q2s[task_index](observation, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logprob_next_action = self.ac.next_action_for_backup(next_observation, task_index)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1s[task_index](next_observation, next_action)
            q2_pi_targ = self.ac_targ.q2s[task_index](next_observation, next_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward + self.gamma * (1 - terminated) * (q_pi_targ - self.alpha * logprob_next_action)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = {f'Q1Vals ({task_index})': q1.detach().numpy(), f'Q2Vals ({task_index})': q2.detach().numpy()}
        return loss_q, q_info

    def compute_loss_pi(self, subcontroller_index, batches):
        loss_vals = []
        for (env_index, batch) in batches:
            if batch is None:
                continue
            observation = batch['observation']
            pi, logprob_pi = self.ac.pis[subcontroller_index](observation)
            q1_pi = self.ac.q1s[env_index](observation, pi)
            q2_pi = self.ac.q2s[env_index](observation, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            # Entropy-regularized policy loss
            loss_vals.append(self.alpha * logprob_pi - q_pi)
        loss_pi = torch.cat(loss_vals).mean()
        pi_info = dict(LogPi = logprob_pi.detach().numpy()) # Useful info for logging
        return loss_pi, pi_info

    def update(self):
        complete_q_info = {}
        # First run one gradient descent step for each Q1 and Q2
        self.q_optimizer.zero_grad()
        for env_index in range(self.num_envs):
            batch = self.replay_buffers[env_index].sample_q_batch(self.batch_size)
            loss_q, q_info = self.compute_loss_q(batch, env_index)
            complete_q_info.update(q_info)
            loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort computing Q gradients during policy learning
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for each subcontroller
        self.pi_optimizer.zero_grad()
        for subcontroller_index in range(self.num_subcontrollers):
            env_probs = sum_to_one([buffer.num_steps(subcontroller_index) for buffer in self.replay_buffers])
            batch_sizes = np.random.multinomial(self.batch_size, pvals = env_probs)
            batches = [buffer.sample_pi_batch(subcontroller_index, batch_size) for buffer, batch_size in
                        zip(self.replay_buffers, batch_sizes)]
            loss_pi, pi_info = self.compute_loss_pi(subcontroller_index, batches)
            loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step
        for p in self.q_params:
            p.requires_grad = True

        self.logger.log({'backprop': {'LossPi': loss_pi.item(), **pi_info, 'LossQ': loss_q.item(), **complete_q_info}})

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # N.B. we're using in-place operations "mul_", "add_" to update target params, 
                # as opposed to the operations "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, observation, env_index, deterministic = False):
        return self.ac.action(torch.as_tensor(as_vector(observation), dtype = torch.float32), env_index, deterministic)

    def test_agent(self, env):
        for _ in range(self.num_test_episodes):
            (observation, _), terminated, episode_return, episode_length = env.reset(), False, 0, 0
            while not terminated:
                # Take deterministic actions at test time
                action = self.get_action(observation, deterministic = True)
                observation, reward, terminated, _, _ = env.step(action)
                episode_return += reward
                episode_length += 1
                self.logger.log({env.name: {'test': {'episode return': episode_return, 'episode length': episode_length}}})

    # TODO: 
    def assign_subcontroller(self, env_index, observation, action):
        pass

    def run(self):
        total_steps = self.steps_per_epoch * self.num_epochs
        start_time = time.time()
        # persistent data structures need to be indexed by environment/task
        episode_returns, episode_lengths = [0] * self.num_envs, [0] * self.num_envs
        observations = [self.envs[i].reset() for i in range(self.num_envs)]

        for t in range(total_steps):
            # Environment interactions
            for env_index in range(self.num_envs):
                # Until start_steps have elapsed, randomly sample actions from a uniform
                # distribution for better exploration. Afterwards, use the learned policy.
                if t > self.start_steps:
                    action = self.get_action(observations[env_index])
                else:
                    action = self.envs[env_index].action_space.sample()
                    subcontroller_index = self.assign_subcontroller(env_index, observations[env_index], action)

                next_observation, reward, terminated, _, _ = self.envs[env_index].step(action)
                episode_returns[env_index] += reward
                episode_lengths[env_index] += 1

                self.replay_buffers[env_index].store(as_vector(observations[env_index]), as_vector(action), reward, 
                                                     as_vector(next_observation), terminated, subcontroller_index)

                # Super critical, easy to overlook step: make sure to update most recent observation!
                observations[env_index] = next_observation

                # End of trajectory handling
                if terminated:
                    env_name = self.envs[env_index].name
                    self.logger.log({env_name: {'train': {'episode return': episode_returns[env_index], 
                                                          'episode length': episode_lengths[env_index]}}})
                    observations[env_index], _ = self.envs[env_index].reset()
                    episode_returns[env_index], episode_lengths[env_index] = 0, 0

            # Backprop (the order of the for loops is chosen so that the environments take turns to backprop)
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    self.update()

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save the model
                if (epoch % self.save_freq == 0) or (epoch == self.num_epochs):
                    torch.save(self.ac.state_dict(), 'temp/model.pt')
                    self.logger.log_artifact('temp/model.pt', name = f"{self.exp_name}_epoch_{epoch}.pt")

                # Test the performance of the deterministic version of the agent
                for env_index in range(self.num_envs):
                    self.test_agent(self.test_envs[env_index], env_index)
                self.logger.log({'epoch_time': time.time() - start_time})
                start_time = time.time()