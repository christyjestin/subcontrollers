import argparse
import torch

from envs import *
from model import VanillaActorCritic
from model.core import as_vector

parser = argparse.ArgumentParser()
parser.add_argument('--task', action = 'store', choices = ['throw', 'catch', 'set'], required = True)
parser.add_argument('--model_path', action = 'store', required = True)
parser.add_argument('--hidden_size', type = int, default = 256)
parser.add_argument('--num_layers', type = int, default = 2)
parser.add_argument('--deterministic', action = 'store_true')

env_map = {
    'throw': ThrowEnv,
    'catch': CatchEnv,
    'set': SetEnv
}

def main(env_type, model_path, hidden_sizes, deterministic):
    env = env_map[env_type](render_mode = 'human')
    model = VanillaActorCritic(env.observation_space, env.action_space, hidden_sizes = hidden_sizes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    num_extra_timesteps = 5
    print(f'Note that the simulation will show {num_extra_timesteps} extra timesteps after the episode terminates')
    print(f'The current policy is {"" if deterministic else "non-"}deterministic')
    episode_length, total_reward = 0, 0
    observation, _ = env.reset()
    for _ in range(1000):
        action = model.action(torch.as_tensor(as_vector(observation), dtype = torch.float32), deterministic)
        observation, net_reward, terminated, _, _ = env.step(action)
        total_reward += net_reward
        episode_length += 1
        if terminated:
            print(f"The total reward for this episode is {total_reward}, and the episode length is {episode_length}")
            for _ in range(5):
                env.passive_step()
            episode_length, total_reward = 0, 0
            observation, _ = env.reset()
    env.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.task, args.model_path, [args.hidden_size] * args.num_layers, args.deterministic)