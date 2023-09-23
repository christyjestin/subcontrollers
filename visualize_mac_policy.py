import argparse
import torch
import os
import matplotlib as mpl
from tabulate import tabulate

from envs import *
from model import MultiActorCritic
from model.core import as_vector

parser = argparse.ArgumentParser()
parser.add_argument('--task', action = 'store', choices = ['throw', 'catch', 'set'], required = True)
parser.add_argument('--num_subcontrollers', action = 'store', type = int, default = 8)
parser.add_argument('--critic_dir', action = 'store', default = 'temp/')
parser.add_argument('--actor_dir', action = 'store', default = 'temp')
parser.add_argument('--hidden_size', type = int, default = 256)
parser.add_argument('--num_layers', type = int, default = 2)
parser.add_argument('--deterministic', action = 'store_true')

env_map = {
    'throw': ThrowEnv,
    'catch': CatchEnv,
    'set': SetEnv
}

# searches `paths` for a path containing all strings in `strings`
def find_file(paths, strings):
    for string in strings:
        paths = list(filter(lambda path: string in path, paths))
    assert len(paths) == 1, f'Too many or too few results were found for the query {strings}: {paths}'
    return paths[0]

# returns a color in the RGBA format used in MJCF files
def cmap(subcontroller_index, num_subcontrollers):
    return mpl.cm.spring(subcontroller_index / num_subcontrollers)

def main(task, num_subcontrollers, critic_dir, actor_dir, hidden_sizes, deterministic):
    env = env_map[task](render_mode = 'human')
    model = MultiActorCritic([env], num_subcontrollers, hidden_sizes = hidden_sizes)

    # load critics
    paths = os.listdir(critic_dir)
    q1_path = find_file(paths, [task, 'q1'])
    q2_path = find_file(paths, [task, 'q2'])
    model.q1s[0].load_state_dict(torch.load(f'{critic_dir}/{q1_path}'))
    model.q2s[0].load_state_dict(torch.load(f'{critic_dir}/{q2_path}'))
    # load actors
    paths = os.listdir(actor_dir)
    pi_paths = [find_file(paths, ['pi', str(i)]) for i in range(num_subcontrollers)]
    for i, pi_path in enumerate(pi_paths):
        model.pis[i].load_state_dict(torch.load(f'{actor_dir}/{pi_path}'))
    # put model in eval mode
    for module in (model.q1s + model.q2s + model.pis):
        module.eval()

    num_extra_timesteps = 5
    print(f'Note that the simulation will show {num_extra_timesteps} extra timesteps after the episode terminates')
    print(f'The current policy is {"" if deterministic else "non-"}deterministic')
    episode_length, total_reward = 0, 0
    observation, _ = env.reset()
    rewards = []
    info_logs = []
    subcontrollers = []
    for _ in range(1000):
        obs = torch.as_tensor(as_vector(observation), dtype = torch.float32)
        action, subcontroller_index = model.action(obs, env_index = 0, deterministic = deterministic)
        observation, net_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(net_reward)
        info_logs.append(info)
        subcontrollers.append(subcontroller_index)
        total_reward += net_reward
        episode_length += 1
        if done:
            print(f"The total reward for this episode is {total_reward}, and the episode length is {episode_length}")
            for _ in range(5):
                env.passive_step()
            print(tabulate([[*info.values(), reward, index] for info, reward, index in zip(info_logs, rewards, subcontrollers)], 
                           headers = ['rewards', 'control', 'changing_fist', 'total_cost', 'net', 'subcontroller']))
            input("Press Enter to Continue")
            rewards = []
            info_logs = []
            subcontrollers = []
            episode_length, total_reward = 0, 0
            observation, _ = env.reset()
    env.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.task, args.num_subcontrollers, args.critic_dir, args.actor_dir, 
         [args.hidden_size] * args.num_layers, args.deterministic)