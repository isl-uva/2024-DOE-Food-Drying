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
from Dymola_Env import DymolaEnv, scale_and_clamp_values
import random
import json


# Create a subdirectory named with the current time to distinguish different log files
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
# create writer
writer = SummaryWriter(log_dir=f'writer/{current_time}')

# Accept hyperparameters
hyper_paras = hyper_parameters()
parse_args_maddpg = hyper_paras.parse_args_maddpg()
# Instantiation environments and algorithms
env = DymolaEnv()
maddpg = MADDPG(parse_args=parse_args_maddpg)


#action boundary value
# action_lower_bounds=[0.25,0.25,1,0.3,273.15+20,273.15+20]
# action_upper_bounds=[0.8,0.8,100,0.6,333.15,353.15]
action_lower_bounds = [0.25, 0.25, 1, 0.3, 273.15, 273.15]
action_upper_bounds = [0.8, 0.8, 50, 0.6, 333.15, 273.15+65]

# max epoch
num_epoch = 10000
# max step for a game
max_step = int(env.end_time/env.step_size)
# Used to record the loss of each parameter iteration
step_for_loss = 0
# Used for reward recording data
reward_record_agent_0 = 0
reward_record_agent_1 = 0
reward_record_agent_2 = 0
max_reward = -1000000
# num initial explore
explore_range = 1

"""
save error results
Fill half of the experience pool
"""
def save_variables(filename, variables):
    with open(filename, 'w') as f:
        json.dump(variables, f)

"""
Explore section
Fill half of the experience pool
"""

print("exploration")

# Add a logic here to explore
# epoch loop
for epoch in range(10):
    print("epoch :", epoch)
    # reset env
    obs_env, infos = env.reset()
    random_step_size = random.randint(1, explore_range)
    print(f'random_step_size: {random_step_size}')

    for i in range(random_step_size):
        action_output = [0.4, env.action_space[0].sample().item(), env.action_space[1].sample().item(), 0.4, 273.15 + 46,
                  env.action_space[2].sample().item()]
        obs_env, _, _, _ = env.step(action_output)

    print(obs_env)
    # step loop
    #need more train step here

    for step in range(i+1,max_step):

        # Convert the data format of the state space to dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        #print(f"critic state: {critic_state}")
        #print(f"actor state: {actor_state}")
        # get action
        actions = maddpg.choose_action(actor_state)
        # print(f'actions: {type(actions)}, {actions}')
        # Convert action to dict
        action_env = functions.action_array_to_dict(actions)
        # step action
        # print(f'action rotation initial: {actions[1].item()}')

        action_input = [(0.4-0.25)/(0.8-0.25), actions[0].item(), actions[1].item(), (0.4-0.3)/0.3, 46/60,
                  actions[2].item()]
        # print(action_input)

        action_output=scale_and_clamp_values(action_input,action_lower_bounds,action_upper_bounds)
        print(f'step: {step}, action: {action_output}')

        obs_next_env, rewards_env, terminations_env, infos = env.step(action_output)
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
    try:
        print("epoch :", epoch)
        action_list=[]
        # reset env
        obs_env, infos = env.reset()

        random_step_size = random.randint(1, explore_range)
        print(f'random_step_size: {random_step_size}')

        for i in range(random_step_size):
            action_output = [0.4, env.action_space[0].sample().item(), env.action_space[1].sample().item(), 0.4, 273.15 + 46,
                      env.action_space[2].sample().item()]
            action_list.append(action_output)
            print(action_output)
            save_variables('variables.json', action_list)

            obs_env, _, _, _ = env.step(action_output)

        print(obs_env)
        # print(env.current_time)
        # step loop
        for step in range(i+1,max_step):
            print(f'step:{step}')
            step_for_loss = step_for_loss + 1

            # Convert the data format of the state space to dict array
            critic_state, actor_state = functions.obs_dict_to_array(obs_env)
            # get action
            actions = maddpg.choose_action(actor_state)
            # Convert action to dict
            action_env = functions.action_array_to_dict(actions)

            #convert action to correct range
            action_input = [(0.4 - 0.25) / (0.8 - 0.25), actions[0].item(), actions[1].item(), (0.4 - 0.3) / 0.3, 46 / 60,
                            actions[2].item()]
            print(action_input)
            action_output = scale_and_clamp_values(action_input, action_lower_bounds, action_upper_bounds)
            action_list.append(action_output)
            print(action_output)
            save_variables('variables.json', action_list)

            # step action
            obs_next_env, rewards_env, terminations_env, infos = env.step(action_output)
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

        print('Ep: {} sum_reward: {}'.format(epoch+1, reward_record))

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

    except Exception as e:
        variables = {
            'action_list': action_list,
            'error': str(e)
        }
        save_variables('errorinfo.json', variables)
        print(f"An error occurred: {e}. Variables saved to 'variables.json'.")
        break


env.close()

