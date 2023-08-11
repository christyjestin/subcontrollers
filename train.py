import argparse
import torch

from envs import *
from model import SAC, MLPActorCritic

env_fn_map = {'throw': ThrowEnv, 'catch': CatchEnv, 'set': SetEnv}

parser = argparse.ArgumentParser()
parser.add_argument("--task", action='store', help="Enter task", choices=['throw', 'catch', 'set'], required=True)
parser.add_argument('--hidden_size', type = int, default = 256)
parser.add_argument('--num_layers', type = int, default = 2)
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--seed', '-s', type = int, default = 0)
parser.add_argument('--num_epochs', type = int, default = 50)
parser.add_argument('--exp_name', type = str, default = 'sac')

def main():
    args = parser.parse_args()
    sac = SAC(env_fn = env_fn_map[args.task], actor_critic = MLPActorCritic, gamma = args.gamma, seed = args.seed, 
                ac_kwargs = {'hidden_sizes': [args.hidden_size] * args.num_layers}, num_epochs = args.num_epochs)
    sac.run()

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    main()