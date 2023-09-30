import argparse
import torch

from envs import *
from model import MAC

env_fn_map = {'throw': ThrowEnv, 'catch': CatchEnv, 'set': SetEnv}

parser = argparse.ArgumentParser()
parser.add_argument('--tasks', action = 'store', help = 'Enter tasks', choices = ['throw', 'catch', 'set'], 
                    nargs = '+', required = True)
parser.add_argument('--exp_name', type = str, default = 'test run')
parser.add_argument('--num_subcontrollers', type = int, default = 8)
parser.add_argument('--no_wandb', dest = 'use_wandb', action = 'store_false', help = 'Disable WandB logging')
parser.add_argument('--hidden_size', type = int, default = 256)
parser.add_argument('--num_layers', type = int, default = 2)
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--alpha', type = float, default = 0.2)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--num_epochs', type = int, default = 50)

def main():
    args = parser.parse_args()
    assert len(args.tasks) == len(set(args.tasks)) and len(args.tasks) > 1, \
        f'There must be more than one task, and they must be unique: {args.tasks}'

    env_fns = [env_fn_map[task] for task in args.tasks]
    mac = MAC(env_fns = env_fns, exp_name = args.exp_name, num_subcontrollers = args.num_subcontrollers, 
              use_wandb = args.use_wandb, num_epochs = args.num_epochs, gamma = args.gamma, seed = args.seed, 
              ac_kwargs = {'hidden_sizes': [args.hidden_size] * args.num_layers})
    print('Starting run...')
    mac.run()

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    main()