from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import wandb

from model.core import *

class SAC:
    """
    Soft Actor-Critic (SAC)

    Args:
        env_fn : A function which creates a copy of the environment. The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch nn.Module with an ``action`` method, a ``pi`` module, a \
            ``q1`` module, and a ``q2`` module. The ``pi`` module should accept batches of observations as inputs, \
            and ``q1`` and ``q2`` should accept a batch of observations and a batch of actions as inputs. The output \
            of the q modules should have shape (batch,) and contain one current estimate of Q* for the provided \
            observations and actions. CRITICAL: make sure to flatten this output.

            Calling ``pi`` should return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of actions in ``a``. Importantly:
                                           | gradients should be able to flow back into ``a``.
            ===========  ================  ======================================

            The action method should output a single action and is not meant for batch computation: this is because of \
            a final packaging step that makes the action compatible with our Gym environments.

        ac_kwargs (dict): Any kwargs appropriate for the VanillaActorCritic object you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) for the agent and the \
            environment in each epoch.

        num_epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target networks. Target networks are \
            updated towards main networks according to:\
            .. math:: \\theta_{\\text{targ}} \\leftarrow \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta \
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to \
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection, before running real policy. \
            Helps exploration.

        update_after (int): Number of env interactions to collect before starting to do gradient descent \
        updates. Ensures replay buffer is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse between gradient descent updates. \
        Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic policy at the end of each epoch.

        save_freq (int): How often (in terms of gap between epochs) to save the current policy and value function.
    """

    def __init__(self, env_fn, exp_name, actor_critic = VanillaActorCritic, ac_kwargs = dict(), seed = 0, 
                 steps_per_epoch = 4000, num_epochs = 100, replay_size = int(1e6), gamma = 0.99, polyak = 0.995, 
                 lr = 1e-3, alpha = 0.2, batch_size = 100, start_steps = 10000, update_after = 1000, 
                 update_every = 50, num_test_episodes = 10, save_freq = 1):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = get_space_dim(self.env.observation_space)
        act_dim = get_space_dim(self.env.action_space)

        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac.to(self.device)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, self.device)
        counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.alert(title = 'Number of parameters', text = f'pi:{counts[0]}, q1: {counts[1]}, q2: {counts[2]}', 
                          level = 'INFO')

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr = lr)
        self.q_optimizer = Adam(self.q_params, lr = lr)


    def compute_q_loss(self, data):
        observation, action, reward, next_observation, done = data['observation'], data['action'], \
                                                data['reward'], data['next_observation'], data['done']

        q1 = self.ac.q1(observation, action)
        q2 = self.ac.q2(observation, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logprob_next_action = self.ac.pi(next_observation)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(next_observation, next_action)
            q2_pi_targ = self.ac_targ.q2(next_observation, next_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward + self.gamma * (1 - done) * (q_pi_targ - self.alpha * logprob_next_action)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals = q1.detach().cpu().numpy(), Q2Vals = q2.detach().cpu().numpy()) # Useful info for logging
        return loss_q, q_info

    def compute_pi_loss(self, data):
        observation = data['observation']
        pi, logprob_pi = self.ac.pi(observation)
        q1_pi = self.ac.q1(observation, pi)
        q2_pi = self.ac.q2(observation, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logprob_pi - q_pi).mean()
        pi_info = dict(LogPi = logprob_pi.detach().cpu().numpy()) # Useful info for logging
        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_q_loss(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort computing Q gradients during policy learning
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_pi_loss(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        self.logger.log({'backprop': {'LossPi': loss_pi.item(), **pi_info, 'LossQ': loss_q.item(), **q_info}})

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # N.B. we're using in-place operations "mul_", "add_" to update target params, 
                # as opposed to the operations "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, observation, deterministic = False):
        observation = torch.as_tensor(as_vector(observation), dtype = torch.float32, device = self.device)
        return self.ac.action(observation, deterministic)

    def test_agent(self):
        for _ in range(self.num_test_episodes):
            (observation, _), done, episode_return, episode_length = self.test_env.reset(), False, 0, 0
            while not done:
                # Take deterministic actions at test time
                action = self.get_action(observation, deterministic = True)
                observation, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            self.logger.log({'test': {'episode return': episode_return, 'episode length': episode_length}})

    def run(self):
        total_steps = self.steps_per_epoch * self.num_epochs
        start_time = time.time()
        (observation, _), episode_return, episode_length = self.env.reset(), 0, 0

        for t in range(total_steps):
            # Until start_steps have elapsed, randomly sample actions from a uniform
            # distribution for better exploration. Afterwards, use the learned policy.
            if t > self.start_steps:
                action = self.get_action(observation)
            else:
                action = self.env.action_space.sample()

            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            self.replay_buffer.store(as_vector(observation), as_vector(action), reward, as_vector(next_observation), 
                                     done)

            # Super critical, easy to overlook step: make sure to update most recent observation!
            observation = next_observation

            # End of trajectory handling
            if done:
                self.logger.log({'train': {'episode return': episode_return, 'episode length': episode_length}})
                (observation, _), episode_return, episode_length = self.env.reset(), 0, 0

            # Backprop
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data = batch)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.num_epochs):
                    torch.save(self.ac.state_dict(), 'temp/model.pt')
                    self.logger.log_artifact('temp/model.pt', name = f"actor_critic_epoch_{epoch}.pt")

                # Test the performance of the deterministic version of the agent
                self.test_agent()
                self.logger.log({'epoch_time': time.time() - start_time})
                start_time = time.time()