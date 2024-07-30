# Import tool files
from maddpg.maddpg import MADDPG
from utils import functions
from hyper_parameters import hyper_parameters
from Dymola_Env import DymolaEnv, scale_and_clamp_values
import matplotlib.pyplot as plt
import json

# Accept hyperparameters
hyper_paras = hyper_parameters()
parse_args_maddpg = hyper_paras.parse_args_maddpg()
# Instantiation environments and algorithmsren
env = DymolaEnv()
maddpg = MADDPG(parse_args=parse_args_maddpg)

maddpg.load_checkpoint()

# action boundary value
# action_lower_bounds = [0.25, 0.25, 10, 0.3, 273.15+30, 273.15+40]
# action_lower_bounds = [0.25, 0.4, 10, 0.3, 273.15+30, 273.15+40]
# action_upper_bounds = [0.8, 0.8, 100, 0.6, 333.15, 353.15]

# 2024/07/22
action_lower_bounds = [0.25, 0.25, 10, 0.3, 273.15, 273.15]
action_upper_bounds = [0.8, 0.8, 100, 0.6, 333.15, 273.15+65]

# max epoch
num_epoch = 1
# max step for each game
max_step = int(env.end_time / env.step_size)

# epoch loop
for epoch in range(num_epoch):
    # save data for plot
    time = []
    Temp_fc = []
    RH_fc = []
    Energy = []
    Flow_r = []
    N = []
    Tset2 = []
    score_all = []
    score = 0
    count = 0
    # reset env
    obs_env, infos = env.reset()
    # step loop
    for step in range(max_step):
        print(f'step: {step}')
        # Convert the data format of the state space to dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        # get action
        actions = maddpg.choose_action(actor_state)
        # Convert action to dict
        action_env = functions.action_array_to_dict(actions)
        # convert action to correct range
        action_input = [(0.4 - 0.25) / (0.8 - 0.25), actions[0].item(), actions[1].item() - 1, (0.5 - 0.3) / 0.3,
                        (46 - 30) / 30,
                        actions[2].item()]
        action_output = scale_and_clamp_values(action_input, action_lower_bounds, action_upper_bounds)

        #action_output=[0.4,0.4,10,0.5,273.15+46,323]

        # step action
        obs_next_env, rewards_env, terminations_env, infos = env.step(action_output)
        # Get new state data
        critic_state_next, actor_state_next = functions.obs_dict_to_array(obs_next_env)
        # state transfer
        obs_env = obs_next_env

        reward = rewards_env['agent_0']

        score += reward

        count += 1

        score_all.append(score)
        time.append(env.current_time / 60.0)
        Temp_fc.append(env.state[27] - 273.15)
        RH_fc.append(100 * env.state[28])
        Energy.append(env.energy)
        Flow_r.append(action_output[1])
        N.append(action_output[2])
        Tset2.append(action_output[5] - 273.15)

        # time.sleep(0.1)



# Plot the results
plt.figure(1,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, Temp_fc)
plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.grid(True)
plt.show(block=False)


# Plot the results
plt.figure(2,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, RH_fc)
plt.xlabel('Time (min)')
plt.ylabel('Relative Humidity (%)')
plt.title('Relative Humidity Over Time')
plt.grid(True)
plt.show(block=False)

# Plot the results
plt.figure(3,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, Energy)
plt.xlabel('Time (min)')
plt.ylabel('Energy (J)')
plt.title('Energy Over Time')
plt.grid(True)
plt.show(block=False)


# Plot the results
plt.figure(4,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, Flow_r)
plt.xlabel('Time (min)')
plt.ylabel('Flowrate (kg/s)')
plt.title('Air Flowrate Over Time')
plt.grid(True)
plt.show(block=False)

# Plot the results
plt.figure(5,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, N)
plt.xlabel('Time (min)')
plt.ylabel('DW Rotation Speed (%)')
plt.title('DW Rotation Speed Over Time')
plt.grid(True)
plt.show(block=False)

# Plot the results
plt.figure(6,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, Tset2)
plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.title('Tset2 Over Time')
plt.grid(True)
plt.show(block=False)

# Plot the results
plt.figure(7,figsize=(10, 6))
#plt.plot(time, moisture_content, label=f'T={T}°C')
plt.plot(time, score_all)
plt.xlabel('Time (min)')
plt.ylabel('score')
plt.title('Score Over Time')
plt.grid(True)
plt.show(block=False)

print("end")


