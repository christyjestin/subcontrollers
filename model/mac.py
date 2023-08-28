from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import time
import wandb
from tqdm import tqdm

from model.core import *

NUM_KERNEL_PCA_COMPONENTS = 10 # size of latent dimension
NUM_NEIGHBORS = 9 # k for kNN

class MAC:
    """
    Multi Actor-Critic (SAC)
    Notes:
        - Entropy
            This implementation only considers the entropy of each subcontroller individually, rather than the overall \
            entropy of the resulting Multi Actor Critic controller. Practically this means that we consider the log \
            probability of an action from a certain subcontroller rather than its overall log probability (marginalizing \
            over all subcontrollers) or the log probability of choosing both the subcontroller and the action. The logic \
            behind this is that the probability of choosing a subcontroller is entirely defined by the Q values and the \
            temperature, and we don't want entropy calculations to interfere with learning the Q function: thus the \
            entropy calculations must not consider the probability of choosing a subcontroller.

    Args:
        env_fns: A list of functions that create a copy of the environment. The environment must satisfy the OpenAI Gym API.

        exp_name (str): The name of the experiment for WandB logging

        num_subcontrollers (int): The number of subcontrollers used in this MultiActorCritic setup

        actor_critic: The constructor method for a PyTorch nn.Module with an ``action`` method and three lists of modules \
            in ``pis``, ``q1s``, ``q2s``. Each member of the lists should match the spec for their respective module type \
            given in the ``SAC`` class. The ``action`` method also matches the spec given in ``SAC``.

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) for the agent and the \
            environment in each epoch.

        num_epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target networks. Target networks are \
            updated towards main networks according to: \
            .. math:: \\theta_{\\text{targ}} \\leftarrow \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta \
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to \
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection, before running real policy. \
            Helps exploration.

        update_after (int): Number of env interactions to collect before starting to do gradient descent updates. \
        Ensures replay buffer is full enough for useful updates. This is also the point where we'll use clustering \
        methods to assign subcontrollers to the initial observations from random exploration.

        update_every (int): Number of env interactions that should elapse between gradient descent updates. \
        Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic policy at the end of each epoch.

        save_freq (int): How often (in terms of gap between epochs) to save the current policy and value function.
        
        use_wandb (bool): Whether to use WandB logging
    """

    def __init__(self, env_fns, exp_name, num_subcontrollers, actor_critic = MultiActorCritic, ac_kwargs = dict(), 
                 seed = 0, steps_per_epoch = 4000, num_epochs = 100, replay_size = int(1e6), gamma = 0.99, 
                 polyak = 0.995, lr = 1e-3, alpha = 0.2, batch_size = 100, start_steps = 10000, update_after = 1000, 
                 update_every = 50, num_test_episodes = 10, save_freq = 1, use_wandb = False):
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
        self.use_wandb = use_wandb

        if self.use_wandb:
            self.logger = wandb.init(project = "subcontrollers", name = exp_name, config = locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.envs, self.test_envs = [env_fn() for env_fn in env_fns], [env_fn() for env_fn in env_fns]

        self.ac = actor_critic(self.envs, self.num_subcontrollers, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(*[q.parameters() for q in (self.ac.q1s + self.ac.q2s)])
        self.pi_params = itertools.chain(*[pi.parameters() for pi in self.ac.pis])
        obs_dim = get_space_dim(self.envs[0].observation_space)
        act_dim = get_space_dim(self.envs[0].action_space)
        # Each environment has its own buffer
        self.replay_buffers = [SubcontrollerReplayBuffer(obs_dim, act_dim, self.num_subcontrollers, replay_size) 
                                for _ in range(self.num_envs)]
        counts = tuple(count_vars(module) for module in [*self.ac.pis, *self.ac.q1s, *self.ac.q2s])
        if self.use_wandb:
            self.logger.alert(title = 'Number of parameters', level = 'INFO', 
                              text = f'pi:{counts[0]}, q1: {counts[1]}, q2: {counts[2]}')

        self.pi_optimizer = Adam(self.pi_params, lr = lr)
        self.q_optimizer = Adam(self.q_params, lr = lr)

        # data structures for assigning subcontrollers (this is done by clustering observations from random exploration)
        self.kernel_pcas = [KernelPCA(n_components = NUM_KERNEL_PCA_COMPONENTS, kernel = 'rbf', eigen_solver = 'arpack')
                            for _ in range(self.num_envs)]
        self.standard_scalers = [StandardScaler() for _ in range(self.num_envs)]
        # we stop using KNN to assign subcontrollers after random exploration, 
        # so the number of points we need to store is exactly start_steps
        self.knn_points = [np.zeros((self.start_steps, NUM_KERNEL_PCA_COMPONENTS), dtype = np.float32)
                           for _ in range(self.num_envs)]

        self.test_subcontroller_counts = [[0] * self.num_subcontrollers for _ in range(self.num_envs)]

    def compute_q_loss(self, data, task_index):
        observation, action, reward, next_observation, terminated = data['observation'], data['action'], \
                                                data['reward'], data['next_observation'], data['terminated']

        q1 = self.ac.q1s[task_index](observation, action)
        q2 = self.ac.q2s[task_index](observation, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, next_action_logprobs = self.ac.next_action_for_backup(next_observation, task_index)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1s[task_index](next_observation, next_action)
            q2_pi_targ = self.ac_targ.q2s[task_index](next_observation, next_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward + self.gamma * (1 - terminated) * (q_pi_targ - self.alpha * next_action_logprobs)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = {f'Q1Vals ({task_index})': q1.detach().numpy(), f'Q2Vals ({task_index})': q2.detach().numpy()}
        return loss_q, q_info

    def compute_pi_loss(self, subcontroller_index, batches):
        loss_vals = []
        logprobs = []
        for env_index, batch in enumerate(batches):
            if batch is None:
                continue # empty batch
            observation = batch['observation']
            pi, logprob = self.ac.pis[subcontroller_index](observation)
            q1_pi = self.ac.q1s[env_index](observation, pi)
            q2_pi = self.ac.q2s[env_index](observation, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            # Entropy-regularized policy loss
            loss_vals.append(self.alpha * logprob - q_pi)
            logprobs.append(logprob)
        loss_pi = torch.cat(loss_vals).mean()
        logprob_pi = torch.cat(logprobs).mean()
        pi_info = dict(LogPi = logprob_pi.detach().numpy()) # Useful info for logging
        return loss_pi, pi_info

    def update(self):
        # First run one gradient descent step for each Q1 and Q2
        full_q_info = {}
        self.q_optimizer.zero_grad()
        for env_index in range(self.num_envs):
            batch = self.replay_buffers[env_index].sample_q_batch(self.batch_size)
            loss_q, q_info = self.compute_q_loss(batch, env_index)
            full_q_info.update(q_info)
            loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort computing Q gradients during policy learning
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for each subcontroller
        full_pi_info = {}
        self.pi_optimizer.zero_grad()
        for subcontroller_index in range(self.num_subcontrollers):
            # give more weight to envs that use the subcontroller more often; this sampling scheme is equivalent (in
            # probability) to directly drawing from all environment steps that use the subcontroller (across all tasks)
            env_probs = sum_to_one([buffer.num_steps(subcontroller_index) for buffer in self.replay_buffers])
            # number of samples to take from each environment
            batch_sizes = np.random.multinomial(self.batch_size, pvals = env_probs)
            batches = [buffer.sample_pi_batch(subcontroller_index, batch_size) for buffer, batch_size in
                        zip(self.replay_buffers, batch_sizes)]
            loss_pi, pi_info = self.compute_pi_loss(subcontroller_index, batches)
            full_pi_info.update(pi_info)
            loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step
        for p in self.q_params:
            p.requires_grad = True

        if self.use_wandb:
            self.logger.log({'backprop': {'LossPi': loss_pi.item(), **full_pi_info, 'LossQ': loss_q.item(), 
                                          **full_q_info}})

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # N.B. we're using in-place operations "mul_", "add_" to update target params, 
                # as opposed to the operations "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, observation, env_index, deterministic = False):
        return self.ac.action(torch.as_tensor(as_vector(observation), dtype = torch.float32), env_index, deterministic)

    def test_agent(self, env, env_index):
        for _ in range(self.num_test_episodes):
            (observation, _), terminated, episode_return, episode_length = env.reset(), False, 0, 0
            while not terminated:
                # Take deterministic actions at test time
                action, subcontroller_index = self.get_action(observation, env_index, deterministic = True)
                self.test_subcontroller_counts[env_index][subcontroller_index] += 1
                observation, reward, terminated, _, _ = env.step(action)
                episode_return += reward
                episode_length += 1
            if self.use_wandb:
                count_dict = {i: v for i, v in enumerate(self.test_subcontroller_counts[env_index])}
                self.logger.log({env.name: {'test': {'episode return': episode_return, 
                                                     'episode length': episode_length, 
                                                     'subcontroller_counts': count_dict}}})

    # We assign a subcontroller for the new observation in two steps:
    # 1. Apply a nonlinear transformation to reduce the dimensionality of the observation
    # 2. Run KNN on the reduced observation to assign a subcontroller (the neighbors in KNN are
    #    all previous observations — all of which have already been assigned to a subcontroller)
    def assign_subcontroller(self, env_index, observation):
        idx = self.replay_buffers[env_index].ptr
        scaled = self.standard_scalers[env_index].transform(observation.reshape(1, -1))
        transformed = self.kernel_pcas[env_index].transform(scaled)
        # run KNN on transformed data
        labels = self.replay_buffers[env_index].subcontroller_index_buffer[:idx]
        assignment = KNN(data = self.knn_points[env_index][:idx], labels = labels, new_point = transformed, 
                         k = NUM_NEIGHBORS)
        # store transformed data point for future runs of KNN
        self.knn_points[env_index][idx] = transformed.flatten()
        return assignment

    # See `assign_subcontroller` for context; this function initiates the process in 3 steps:
    # 1. Fit a nonlinear transformation by running Kernel PCA on initial observations
    # 2. Run a clustering algorithm to produce initial subcontroller assignments for observations
    # 3. Save these assignments to use in the KNN classification scheme for later observations
    # this function also returns a list containing the counts for each subcontroller
    def start_assigning_subcontrollers(self, env_index):
        # standardize and run kernel PCA to apply a nonlinear transformation to observations
        data = self.replay_buffers[env_index].observation_buffer[:self.update_after]
        scaled_data = self.standard_scalers[env_index].fit_transform(data)
        transformed_data = self.kernel_pcas[env_index].fit_transform(scaled_data)
        # run clustering algorithm
        HAC = AgglomerativeClustering(n_clusters = self.num_subcontrollers, linkage = 'ward', compute_full_tree = False)
        assignments = HAC.fit_predict(transformed_data)
        # store transformed data to run KNN in future timesteps
        self.knn_points[env_index][:self.update_after] = transformed_data
        # store assigned subcontrollers
        self.replay_buffers[env_index].subcontroller_index_buffer[:self.update_after] = assignments
        for i, v in enumerate(assignments):
            self.replay_buffers[env_index].steps_by_subcontroller[v].append(i)
        _, counts = np.unique(assignments, return_counts = True)
        return counts

    def run(self):
        total_steps = self.steps_per_epoch * self.num_epochs
        start_time = time.time()
        # persistent data structures need to be indexed by environment/task
        episode_returns, episode_lengths = [0] * self.num_envs, [0] * self.num_envs
        train_subcontroller_counts = [[0] * self.num_subcontrollers for _ in range(self.num_envs)]
        observations = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        clean_exp_name = '_'.join(self.exp_name.split()) # replace spaces with underscores

        for t in tqdm(range(total_steps)):
            if t == self.start_steps:
                print('Finished exploration')
            elif t == self.update_after:
                print('Started assigning subcontrollers and backpropagating')
            # Environment interactions for all tasks
            for env_index, env in enumerate(self.envs):
                # Until start_steps have elapsed, randomly sample actions from a uniform
                # distribution for better exploration. Afterwards, use the learned policy.
                if t >= self.start_steps:
                    action, subcontroller_index = self.get_action(observations[env_index], env_index)
                    train_subcontroller_counts[env_index][subcontroller_index] += 1
                else:
                    action = env.action_space.sample()
                    # can't assign a subcontroller until we've fit our clustering methods
                    if t >= self.update_after:
                        subcontroller_index = self.assign_subcontroller(env_index, as_vector(observations[env_index]))
                    else:
                        subcontroller_index = -1

                next_observation, reward, terminated, _, _ = env.step(action)
                episode_returns[env_index] += reward
                episode_lengths[env_index] += 1

                self.replay_buffers[env_index].store(as_vector(observations[env_index]), as_vector(action), reward, 
                                                     as_vector(next_observation), terminated, subcontroller_index)

                # Super critical, easy to overlook step: make sure to update most recent observation!
                observations[env_index] = next_observation

                # End of trajectory handling
                if terminated:
                    if self.use_wandb:
                        count_dict = {i: v for i, v in enumerate(train_subcontroller_counts[env_index])}
                        self.logger.log({env.name: {'train': {'episode return': episode_returns[env_index], 
                                                              'episode length': episode_lengths[env_index], 
                                                              'subcontroller_counts': count_dict}}})
                    observations[env_index], _ = env.reset()
                    episode_returns[env_index], episode_lengths[env_index] = 0, 0

                # Fit clustering methods and retroactively assign subcontrollers
                if t + 1 == self.update_after:
                    train_subcontroller_counts[env_index] = self.start_assigning_subcontrollers(env_index)

            # Backprop (happens outside of the environment loop i.e. all environments backprop together)
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    self.update()

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save the model
                if ((epoch % self.save_freq == 0) or (epoch == self.num_epochs)) and self.use_wandb:
                    torch.save(self.ac.state_dict(), 'temp/model.pt')
                    self.logger.log_artifact('temp/model.pt', name = f"{clean_exp_name}_epoch_{epoch}.pt")

                # Test the performance of the deterministic version of the agent
                for env_index in range(self.num_envs):
                    self.test_agent(self.test_envs[env_index], env_index)
                if self.use_wandb:
                    self.logger.log({'epoch_time': time.time() - start_time})
                start_time = time.time()