import time
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# Introducing algorithms and environments
from pettingzoo.mpe import simple_spread_v3
from maddpg.maddpg import MADDPG
# Import tool files
from utils import functions
from hyper_parameters import hyper_parameters


# Accept hyperparameters
hyper_paras = hyper_parameters()
parse_args_maddpg = hyper_paras.parse_args_maddpg()
# Instantiation environments and algorithmsren
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=200, continuous_actions=True, render_mode="human")
maddpg = MADDPG(parse_args=parse_args_maddpg)

maddpg.load_checkpoint()

# max epoch
num_epoch = 10
# max step for each game
max_step = 50

# epoch loop
for epoch in range(num_epoch):
    # reset env
    obs_env, infos = env.reset()
    # step loop
    for step in range(max_step):

        # Convert the data format of the state space to dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        # get action
        actions = maddpg.choose_action(actor_state)
        print(actions)
        # Convert action to dict
        action_env = functions.action_array_to_dict(actions)
        # step action
        obs_next_env, rewards_env, terminations_env, truncations, infos = env.step(action_env)
        # Get new state data
        critic_state_next, actor_state_next = functions.obs_dict_to_array(obs_next_env)
        # state transfer
        obs_env = obs_next_env

        # time.sleep(0.1)


env.close()

