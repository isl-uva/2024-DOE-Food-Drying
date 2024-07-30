from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt
from pyfmi import load_fmu



def scale_and_clamp_values(input_list, lower_bounds, upper_bounds):
    """
    Scale the elements of input_list to be a ratio between the specified bounds,
    with clamping to ensure values are within [0, 1] range.

    Args:
    - input_list (list of floats/integers): The list of ratios to scale.
    - lower_bounds (list of floats/integers): The lower bounds for each element.
    - upper_bounds (list of floats/integers): The upper bounds for each element.

    Returns:
    - list: The scaled and clamped values within the bounds.
    """
    if len(input_list) != len(lower_bounds) or len(input_list) != len(upper_bounds):
        raise ValueError("The length of the input list must match the length of the bounds lists.")

    scaled_list = []
    for i, ratio in enumerate(input_list):
        # Clamp the ratio to be within [0, 1]
        clamped_ratio = max(0, min(1, ratio))
        scaled_value = lower_bounds[i] + clamped_ratio * (upper_bounds[i] - lower_bounds[i])
        scaled_list.append(scaled_value)

    return scaled_list


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
            Box(low=np.array([1]), high=np.array([100]), dtype=np.float32),
            # Agent 1: N, Rotation speed of Desiccant wheel
            Box(low=np.array([273.15]), high=np.array([353.15]), dtype=np.float32),
        # Agent 2: Tset2, Temperature setpoint of Heater 2
        ))
        # define observation space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

        # load fmu file
        self.model = load_fmu("DWHP1_fmu 1.fmu", kind='cs', log_level=2)

        self.output = 30  ## numbers of outputs
        self.start_time = 0  #
        self.end_time = 10 * 3600  #
        self.current_time = self.start_time
        self.step_size = 300  # Define your own step size, unit is second, 300s = 5min
        self.done = False
        self.energy = 0
        self.num_agents = len(self.action_space)
        self.power_ref = 6000.0
        self.Temp_d = 273.15 + 46
        self.RH_d = 0.2

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

        # Caluculate Observation
        observation_all = []

        for i in range(self.output):
            self.state[i] = self.model.get(f"y{i}")[0]
            if i >= 6 and i % 3 != 2:
                observation_all.append(self.state[i])

        agents_dict = {}

        for i in range(self.num_agents):
            key = f'agent_{i}'
            value = observation_all.copy()
            agents_dict[key] = value

        agents_dict['agent_0'].append(self.state[1])
        agents_dict['agent_1'].append(self.state[2])
        agents_dict['agent_2'].append(self.state[5])

        self.obs_env = agents_dict

        self.current_time += self.step_size

        # Check if shower is done
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
            hum_reward = -1.0

        if temp_error < 1:
            temp_reward = 1.0
        else:
            temp_reward = -1.0

        power_reward = (self.power_ref - power) / self.power_ref
        # print(f'power_reward: {type(power_reward)} {power_reward}')

        # reward_all = hum_reward + temp_reward + power_reward
        reward_all = 2*hum_reward + temp_reward
        # reward_all = hum_reward

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



        # Caluculate Observation
        observation_all = []

        for i in range(self.output):
            self.state[i] = self.model.get(f"y{i}")[0]
            if i >= 6 and i % 3 != 2:
                observation_all.append(self.state[i])

        agents_dict = {}

        for i in range(self.num_agents):
            key = f'agent_{i}'
            value = observation_all.copy()
            agents_dict[key] = value

        agents_dict['agent_0'].append(self.state[1])
        agents_dict['agent_1'].append(self.state[2])
        agents_dict['agent_2'].append(self.state[5])

        self.obs_env = agents_dict

        info = []


        print("reset activate")
        return self.obs_env, info

