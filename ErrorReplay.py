from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from Dymola_Env import DymolaEnv, scale_and_clamp_values
import json
# Define the file path
file_path = 'errorcontrol2.json'
# file_path = 'errorcontrol.json'
# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)
# Define the file path
# file_path = 'ErrorInput.txt'
#
# # Initialize an empty list to store all rows
# data = []
print(len(data))
# # Read the file
# with open(file_path, 'r') as file:
#     for line in file:
#         # Strip any surrounding whitespace and parse the line as a list of floats
#         row = eval(line.strip())
#         data.append(row)

env = DymolaEnv()
episodes = 100

for episode in range(1, episodes + 1):
    print(f'episode:{episode}')
    time = []
    Temp_fc = []
    RH_fc = []
    Energy = []
    Flow_r = []
    N = []
    Tset2 = []
    score_all = []

    n_state, info = env.reset()
    done = False
    score = 0
    count = 0

    while not done:
        # action=[0.4,env.action_space[0].sample().item(),env.action_space[1].sample().item(),0.5,273.15+46,env.action_space[2].sample().item()]
        action = data[count]
        # print(action)
        n_state, reward_env, done_env, info = env.step(action)
        reward = reward_env['agent_0']
        done = done_env['agent_0']

        score += reward

        count += 1

        score_all.append(score)
        time.append(env.current_time / 60.0)
        Temp_fc.append(env.state[27] - 273.15)
        RH_fc.append(100 * env.state[28])
        Energy.append(env.energy)
        Flow_r.append(action[1])
        N.append(action[2])
        Tset2.append(action[5] - 273.15)

        if count>=len(data):
            break