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

# Create a subdirectory named with the current time to distinguish different log files
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
# create writer
writer = SummaryWriter(log_dir=f'writer/{current_time}')

# Accept hyperparameters
hyper_paras = hyper_parameters()
parse_args_maddpg = hyper_paras.parse_args_maddpg()
# Instantiation environments and algorithms
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=200, continuous_actions=True, render_mode=None)
maddpg = MADDPG(parse_args=parse_args_maddpg)


# max epoch
num_epoch = 20000
# max step for a game
max_step = 50
# Used to record the loss of each parameter iteration
step_for_loss = 0
# Used for reward recording data
reward_record_agent_0 = 0
reward_record_agent_1 = 0
reward_record_agent_2 = 0
max_reward = -1000000

"""
Explore section
Fill half of the experience pool
"""

print("exploration")

# Add a logic here to explore
# epoch loop
for epoch in range(1000):
    print("epoch :", epoch)
    # reset env
    obs_env, infos = env.reset()
    #print(infos)
    # step loop
    for step in range(max_step):

        # Convert the data format of the state space to dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        #print(f"critic state: {critic_state}")
        #print(f"actor state: {actor_state}")
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
        rewards = [rewards_env["agent_0"], rewards_env["agent_1"],  rewards_env["agent_2"]]
        terminal = [terminations_env["agent_0"], terminations_env["agent_1"], terminations_env["agent_2"]]
        # Storing data
        maddpg.buffer.store_transition(
            critic_state, actor_state, actions, rewards, critic_state_next, actor_state_next, terminal
        )

print("begin training")

"""
training part
"""
# epoch loop
for epoch in range(num_epoch):
    # reset env
    obs_env, infos = env.reset()
    # step loop
    for step in range(max_step):

        step_for_loss = step_for_loss + 1

        # Convert the data format of the state space to dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        # get action
        actions = maddpg.choose_action(actor_state)
        # Convert action to dict
        action_env = functions.action_array_to_dict(actions)
        # step action
        obs_next_env, rewards_env, terminations_env, truncations, infos = env.step(action_env)
        # Get new state data
        critic_state_next, actor_state_next = functions.obs_dict_to_array(obs_next_env)
        # state transfer
        obs_env = obs_next_env
        rewards = [rewards_env["agent_0"], rewards_env["agent_1"], rewards_env["agent_2"]]
        terminal = [terminations_env["agent_0"], terminations_env["agent_1"], terminations_env["agent_2"]]
        # Storing data
        maddpg.buffer.store_transition(
            critic_state, actor_state, actions, rewards, critic_state_next, actor_state_next, terminal
        )
        # training
        maddpg.learn(writer, step_for_loss)

        # Record reward data
        reward_record_agent_0 = reward_record_agent_0 + rewards[0]
        reward_record_agent_1 = reward_record_agent_1 + rewards[1]
        reward_record_agent_2 = reward_record_agent_2 + rewards[2]

    reward_record = reward_record_agent_0 + reward_record_agent_1 + reward_record_agent_2

    print('Ep: {} sum_eward: {}'.format(epoch+1, reward_record))

    writer.add_scalar('reward/sum_reward', reward_record, (epoch+1))

    # If the sum of rewards is greater than the sum of previous maximum rewards, save the model
    if reward_record >= max_reward:
        max_reward = reward_record
        # save the model
        maddpg.save_checkpoint()

    # Record the maximum cumulative reward
    writer.add_scalar('reward/max_reward', max_reward, (epoch+1))

    # The cumulative rewards for one game are cleared
    reward_record_agent_0 = 0
    reward_record_agent_1 = 0
    reward_record_agent_2 = 0


env.close()

