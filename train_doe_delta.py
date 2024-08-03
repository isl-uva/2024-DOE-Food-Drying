import time
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# Introducing algorithms and environments
from pettingzoo.mpe import simple_spread_v3
from maddpg_delta.maddpg import MADDPG
# Import tool files
from utils_delta import functions
from hyper_parameters_delta import hyper_parameters
from Dymola_Env_Delta import DymolaEnv
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
action_lower_bounds = np.array([0.25, 0.25, 10, 0.1, 273.15, 273.15])
action_upper_bounds = np.array([0.8, 0.8, 50, 0.4, 333.15, 273.15+65])
action_delta_bounds = (action_upper_bounds - action_lower_bounds)*0.2
control_baseline= np.array([0.4, 0.4, 10, 0.2, 273.15+46, 323])


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

    control_current = control_baseline

    for i in range(random_step_size):
        action_output = [0.4, env.action_space[0].sample().item(), env.action_space[1].sample().item(), 0.4, 273.15 + 46,
                  env.action_space[2].sample().item()]
        print(f'random_step_size: {random_step_size}, action:{action_output}')
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

        action_input = np.array([0.5, actions[0].item(), actions[1].item(), 0.5, 0.5,
                  actions[2].item()])
        print(f'action_origin:{action_input}')

        action_output=control_current+(2*action_input-1)*action_delta_bounds
        action_output=np.clip(action_output,action_lower_bounds,action_upper_bounds)
        control_current=action_output
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
            control_current = control_baseline
            print(f'step:{step}')
            step_for_loss = step_for_loss + 1

            # Convert the data format of the state space to dict array
            critic_state, actor_state = functions.obs_dict_to_array(obs_env)
            # get action
            actions = maddpg.choose_action(actor_state)
            # Convert action to dict
            action_env = functions.action_array_to_dict(actions)

            #convert action to correct range
            action_input = np.array([0.5, actions[0].item(), actions[1].item(), 0.5, 0.5,
                                     actions[2].item()])
            print(action_input)

            action_output = control_current + (2 * action_input - 1) * action_delta_bounds
            action_output = np.clip(action_output, action_lower_bounds, action_upper_bounds)
            control_current = action_output
            print(f'step: {step}, action: {action_output}')
            action_list.append(action_output.tolist())
            print(action_output)
            save_variables('variables20240802.json', action_list)

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

