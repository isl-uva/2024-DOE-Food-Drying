from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt
from pyfmi import load_fmu



# # Example usage
# a = [1000, 0.2, 0.2]  # Ratios, with the first value out of bounds
# lower_bounds = [0, 0, 0]
# upper_bounds = [100, 50, 40]
#
# scaled_a = scale_and_clamp_values(a, lower_bounds, upper_bounds)
# print("Scaled and clamped input list:", scaled_a)

class DymolaEnv(Env):
    def __init__(self):
        # define action space
        self.action_space = Tuple((
            Box(low=np.array([0.25]), high=np.array([0.8]), dtype=np.float32),
            # Agent 0: Flow_r, Regeneration air flowrate
            Box(low=np.array([10]), high=np.array([50]), dtype=np.float32),
            # Agent 1: N, Rotation speed of Desiccant wheel
            Box(low=np.array([273.15]), high=np.array([273.15+65]), dtype=np.float32),
        # Agent 2: Tset2, Temperature setpoint of Heater 2
        ))
        # define observation space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

        # load fmu file DWHP1_fmu 1.fmu is an error file
        self.model = load_fmu("DWHP1_fmu 1.fmu", kind='cs', log_level=2)

        self.output = 30  ## numbers of outputs
        self.start_time = 0  #
        self.end_time = 8 * 3600  #
        self.current_time = self.start_time
        self.step_size = 300  # Define your own step size, unit is second, 300s = 5min
        self.done = False
        self.energy = 0
        self.num_agents = len(self.action_space)
        self.power_ref = 6000.0
        self.temp_ref = 20.0
        self.rh_ref = 1.0
        self.params_old=[0.4, 10, 323]


        # for training
        self.Temp_d = 273.15 + 46
        self.RH_d = 0.19

        # for testing
        # self.Temp_d = 273.15 + 46
        # self.RH_d = 0.25



        # read weather data
        weather_TP=[]
        weather_RH=[]

        # with open("modelica_input_data.txt", 'r') as file:
        with open("modelica_input_data.txt", 'r') as file:
            for line in file:
                # Split the line into individual elements
                parts = line.strip().split('\t')
                # Append the elements to respective lists
                weather_TP.append(float(parts[1]))
                weather_RH.append(float(parts[2]))

        self.weather_TP=weather_TP
        self.weather_RH=weather_RH

        # Create a list with 30 elements, all of which are 0, using iteration
        zero_list = []
        for _ in range(self.output):
            zero_list.append(0)

        self.state = zero_list
        # Initialize the model with the start time
        self.model.reset()
        self.model.initialize(self.start_time)
        print("success boot")

        # set start temp
        # self.state=np.array([38+random.uniform(-3, 3), 38+random.uniform(-3, 3), 38+random.uniform(-3, 3)])
        # Temparture, Mo
        # self.state=np.array([25.0, 1.0])
        # set shower length
        self.time_id = 0
        self.dt = 0.2
        self.error_thres = 0.001

    def step(self, action):
        # set action parameters
        paras = action

        self.model.set('FLOW_p', paras[0])
        self.model.set('FLOW_r', paras[1])
        self.model.set('N', paras[2])
        self.model.set('SP_HP', paras[3])
        self.model.set('Tset1', paras[4])
        self.model.set('Tset2', paras[5])

        self.model.do_step(self.current_time, self.step_size, True)

        # Read all avaliable data from model
        for i in range(self.output):
            self.state[i] = self.model.get(f"y{i}")[0]

        # Define the number of agents and the number of elements in each list
        num_agents = self.num_agents


        # Use a dictionary comprehension to create the dictionary
        # agents_dict = {f"agent_{i}": [None] * list_size for i in range(num_agents)}
        agents_dict={}
        for i in range(num_agents):
            agents_dict[f"agent_{i}"] = []

        # regeneration fan observation
        agents_dict['agent_0'].append((self.weather_TP[int(self.current_time / 3600)] - self.Temp_d)/self.temp_ref)
        agents_dict['agent_0'].append((self.weather_RH[int(self.current_time / 3600)] - self.RH_d)/self.rh_ref)
        agents_dict['agent_0'].append((self.state[6]-self.Temp_d)/self.temp_ref)
        agents_dict['agent_0'].append((self.state[7]-self.RH_d)/self.rh_ref)
        # Add absolute status
        # agents_dict['agent_0'].append(self.weather_TP[int(self.current_time / 3600)])
        # agents_dict['agent_0'].append(self.weather_RH[int(self.current_time / 3600)])
        # agents_dict['agent_0'].append(self.state[6])
        # agents_dict['agent_0'].append(self.state[7])
        #Add agent power
        agents_dict['agent_0'].append(self.params_old[0])



        # Dessicant Wheel Observation
        agents_dict['agent_1'].append((self.state[21]-self.Temp_d)/self.temp_ref)
        agents_dict['agent_1'].append((self.state[22]-self.RH_d)/self.rh_ref)
        agents_dict['agent_1'].append((self.state[24]-self.Temp_d)/self.temp_ref)
        agents_dict['agent_1'].append((self.state[25]-self.RH_d)/self.rh_ref)
        # Add absolute status
        # agents_dict['agent_1'].append(self.state[21])
        # agents_dict['agent_1'].append(self.state[22])
        # agents_dict['agent_1'].append(self.state[24])
        # agents_dict['agent_1'].append(self.state[25])
        # Add agent power
        agents_dict['agent_1'].append(self.params_old[1]/50)

        # Heater 2 observation
        agents_dict['agent_2'].append((self.state[9]-self.Temp_d)/self.temp_ref)
        agents_dict['agent_2'].append((self.state[10]-self.RH_d)/self.rh_ref)
        agents_dict['agent_2'].append((self.state[21]-self.Temp_d)/self.temp_ref)
        agents_dict['agent_2'].append((self.state[22]-self.RH_d)/self.rh_ref)
        # Add absolute status
        # agents_dict['agent_2'].append(self.state[9])
        # agents_dict['agent_2'].append(self.state[10])
        # agents_dict['agent_2'].append(self.state[21])
        # agents_dict['agent_2'].append(self.state[22])
        # Add agent power
        agents_dict['agent_2'].append((self.params_old[2]-self.Temp_d)/self.temp_ref)

        self.params_old[0]=paras[1]
        self.params_old[1]=paras[2]
        self.params_old[2] = paras[5]

        # Return the observation
        self.obs_env=agents_dict

        #Update System Time
        self.current_time += self.step_size

        # Check if system is done
        if self.current_time >= self.end_time:
            self.done = True
            self.model.terminate()
            print("dymola model end")
        else:
            self.done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        Temp_d = self.Temp_d
        RH_d = self.RH_d

        Temp_fc = self.state[27]
        RH_fc = self.state[28]

        humidity_error = abs(RH_d - RH_fc)
        temp_error = abs(Temp_d - Temp_fc)

        # calculate power
        power = 0
        for i in range(6):
            power += self.state[i]

        self.energy += power * self.step_size

        # calculate reward
        if humidity_error < 0.01:
            hum_reward = 1.0
        else:
            hum_reward = -1.0 - humidity_error

        if temp_error < 1:
            temp_reward = 1.0
        else:
            temp_reward = -1.0 - temp_error / 20

        power_reward = (self.power_ref - power) / self.power_ref

        reward_all = hum_reward + temp_reward
        # reward_all = hum_reward + temp_reward + power_reward

        reward = {}

        for i in range(self.num_agents):
            key = f'agent_{i}'
            # value = reward_all.copy()
            value = reward_all
            reward[key] = value

        done = {}

        for i in range(self.num_agents):
            key = f'agent_{i}'
            value = self.done
            done[key] = value

        info = {}

        # return step information
        return self.obs_env, reward, done, info

    def render(self):
        print(self.state)
        pass

    def reset(self):
        # Initialize the model with the start time

        if self.current_time >= self.end_time:
            self.model = load_fmu("DWHP1_fmu 1.fmu", kind='cs', log_level=2)
            self.model.reset()
            self.model.initialize(self.start_time)
            self.current_time = 0
            self.energy = 0
            self.done = False
            print("Normal end: time reset")

        if self.current_time>0:
            if not self.done:
                self.model.terminate()
                self.model = load_fmu("DWHP1_fmu 1.fmu", kind='cs', log_level=2)
                self.model.reset()
                self.model.initialize(self.start_time)
                self.current_time = 0
                self.energy = 0
                self.done = False
                print("Error end: time reset")


        # Define the number of agents and the number of elements in each list
        num_agents = self.num_agents


        # Use a dictionary comprehension to create the dictionary
        # agents_dict = {f"agent_{i}": [None] * list_size for i in range(num_agents)}
        agents_dict={}
        for i in range(num_agents):
            agents_dict[f"agent_{i}"] = []

        # regeneration fan observation
        agents_dict['agent_0'].append(self.weather_TP[int(self.current_time / 3600)] - self.Temp_d)
        agents_dict['agent_0'].append(self.weather_RH[int(self.current_time / 3600)] - self.RH_d)
        agents_dict['agent_0'].append(self.state[6] - self.Temp_d)
        agents_dict['agent_0'].append(self.state[7] - self.RH_d)
        # Add absolute status
        # agents_dict['agent_0'].append(self.weather_TP[int(self.current_time / 3600)])
        # agents_dict['agent_0'].append(self.weather_RH[int(self.current_time / 3600)])
        # agents_dict['agent_0'].append(self.state[6])
        # agents_dict['agent_0'].append(self.state[7])
        # Add agent power
        agents_dict['agent_0'].append(0.4)

        # Dessicant Wheel Observation
        agents_dict['agent_1'].append(self.state[21] - self.Temp_d)
        agents_dict['agent_1'].append(self.state[22] - self.RH_d)
        agents_dict['agent_1'].append(self.state[24] - self.Temp_d)
        agents_dict['agent_1'].append(self.state[25] - self.RH_d)
        # Add absolute status
        # agents_dict['agent_1'].append(self.state[21])
        # agents_dict['agent_1'].append(self.state[22])
        # agents_dict['agent_1'].append(self.state[24])
        # agents_dict['agent_1'].append(self.state[25])
        # Add agent power
        agents_dict['agent_1'].append(10)

        # Heater 2 observation
        agents_dict['agent_2'].append(self.state[9] - self.Temp_d)
        agents_dict['agent_2'].append(self.state[10] - self.RH_d)
        agents_dict['agent_2'].append(self.state[21] - self.Temp_d)
        agents_dict['agent_2'].append(self.state[22] - self.RH_d)
        # Add absolute status
        # agents_dict['agent_2'].append(self.state[9])
        # agents_dict['agent_2'].append(self.state[10])
        # agents_dict['agent_2'].append(self.state[21])
        # agents_dict['agent_2'].append(self.state[22])
        # Add agent power
        agents_dict['agent_2'].append(323-self.Temp_d)

        self.obs_env = agents_dict

        info = []


        print("reset activate")
        return self.obs_env, info

