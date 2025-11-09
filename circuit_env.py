import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os
import yaml
import math
from ngspice_interface import DUT as DUT_NGSpice
from utils.plotting import plotLearning, plot_running_maximum

class CircuitEnv(gym.Env):
    PER_LOW, PER_HIGH = -np.inf, +np.inf
    
    def __init__(self, config=None, circuit_name='TwoStage', run_id='rllib_baseline',
                 success_threshold=0.0, simulator='ngspice'):
        self.run_id = run_id
        self.max_steps_per_episode = 10
        self.env_steps = 0
        self.episode_steps = 0
        self.success_threshold = success_threshold
        self.circuit_name = circuit_name

        project_path = os.getcwd()
    
        yaml_directory = os.path.join(project_path, f"{simulator}_interface", 'files', 'yaml_files')
        circuit_yaml_path = os.path.join(yaml_directory, f'{circuit_name}.yaml')
        with open(circuit_yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.Loader)
        
        self.dict_params = yaml_data['params']
        self.dict_targets = yaml_data['targets']

        self.hard_constraints = yaml_data['hard_constraints']
        self.optimization_targets = yaml_data['optimization_targets']

        # number of input action components
        self.n_actions = len(self.dict_params) #20 [min,max,step]

        # number of output observation components
        self.obs_dim = len(self.dict_targets) #7

        self.param_ranges = {}
        for name, value in self.dict_params.items():
           self.param_ranges[name] = {'min': value[0], 'max': value[1], 'step': value[2]}
        
        #simulate engine
        self.simulation_engine = DUT_NGSpice(circuit_yaml_path)

        #print(f"\n Initialized {circuit_name} with simulator {simulator} \n")

        # Initialization of action space & observation space
        """
            low and high = minimum and maximum possible values for each action variable
            
        """
        # Action space defines what actions the agent can take
        act_high = np.array([1 for _ in range(self.n_actions)])
        act_low = np.array([-1 for _ in range(self.n_actions)])
         #creat a continuous action space in the range [-1,1] for each parameters
         # BOX just selected continuous values within the range [-1,1]
        self.action_space = Box(low=act_low, high=act_high)
        
        #Observation space gives the "states" of the system
        obs_high = np.array([CircuitEnv.PER_HIGH]*self.obs_dim, dtype=np.float32)
        obs_low = np.array([CircuitEnv.PER_LOW]*self.obs_dim, dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high)

        # spec reward weights
        self.spec_weights = yaml_data['spec_weights']

        self.reward_history = []
        self.score_history = []

        self.counter = 0
        self.pvt_corner = {'process': 'TT', 'voltage': 1.2, 'temp': 27}
    
    def action_refine(self, action):
        """
        TODO: Implement a function that converts normalized actions to actual parameter values.
        
        This function should:
        1. Take a flattened numpy array of actions (values between -1 and 1)
        2. Convert each action value to the actual parameter value from self.parameter_ranges
        
        You have two options;
            1) Continuous mapping: convert the action directly to the parameter value based on the min-max values
            2) Discretize mapping: create a vector of possible values using the min-max-step from self.parameter_ranges, 
                                    then, map the action to an index of that vector and retreive the corresponding actual value.

        Args:
            action: numpy array of normalized values between -1 and 1
            
        Returns:
            dict: Dictionary mapping parameter names to their actual sizing values
                e.g., {'mp1': 13, 'wp1':2.0e-06, 'lp1':9.0e-08, ...}
        
        """
        
        # continuos mapping
        actual_action = {}
        action = np.asarray(action, dtype=np.float32).flatten()
        # print("Action:", action)
        # print("Action shape:", action.shape)
        for i,name in enumerate(list(self.dict_params.keys())):
            actual_action[name] = (action[i]+1)*0.5*(self.param_ranges[name]["max"]-self.param_ranges[name]["min"]) + self.param_ranges[name]["min"]
        return actual_action
    
    
    def simulate(self, params):
        """
        TODO: Create/Simulate netlist with the given parameters and return the measured metrics.

        Args:
            params (dict): A dictionary containing the parameters for the simulation.

        Returns:
            dict: A dictionary containing the measured metrics from the simulation.
        """
        new_netlist_path = self.simulation_engine.create_new_netlist(params,
                                                                    process=self.pvt_corner['process'],
                                                                    temp_pvt=self.pvt_corner['temp'],
                                                                    vdd=self.pvt_corner['voltage'])
        info = self.simulation_engine.simulate(new_netlist_path)
        #print(f"\nNew netlist created at: {new_netlist_path}")
        return self.simulation_engine.measure_metrics()
    
    def normalize_specs(self, spec_dict):
        """
        TODO: Normalize the specifications in `spec_dict` based on target specifications.

        This function normalizes the values in `spec_dict` by comparing them to the
        target specifications stored in `self.dict_targets`. The normalization is 
        performed using the formula:
            normalized_value = (spec_value - goal_value) / (spec_value + goal_value) 
        
        Args:
            spec_dict (dict): A dictionary containing the specifications to be normalized.
                              The keys should match those in `self.dict_targets`.
        
        Returns:
            dict: A dictionary containing the normalized specifications, with the same keys
                  as `spec_dict`.
        """
        normalized_spec = {}
        for name in self.dict_targets.keys():
                normalized_spec[name] = (spec_dict[name]-self.dict_targets[name])/(spec_dict[name]+self.dict_targets[name])
        return normalized_spec
    
    def evaluate(self, action):
        self.param_values = self.action_refine(action)
        #print(f"\n Env-Evaluate: Refined action: {action}")
        self.real_specs = self.simulate(self.param_values)
        #print(f"\n Env-Evaluate: Real specs: {self.real_specs}")
        self.cur_norm_specs = self.normalize_specs(self.real_specs)
        #print(f"\n Env-Evaluate: Normalized specs: {self.cur_norm_specs}")
    
    def reset(self, *, seed=None, options=None):
        """
        TODO: Reset the environment to an initial state and return the initial observation.

        Parameters:
            No mandatory input parameters (leave the input signature as is).
            seed (int, optional): A seed for the random number generator to ensure reproducibility.
            options (dict, optional): Additional options for the reset process.

        Returns:
            np.ndarray: The initial observation of the environment, which is the normalized current specifications.

        This method performs the following steps:
        1. Initializes the episode steps counter to zero.
        2. Generates a random action within the range [-1, 1] for each action parameter.
        3. Evaluates the environment with the generated random action.
        4. Resets the episode score to zero.
        5. Constructs the initial observation by returning the normalized current specifications.

        Note:
        - The observation is returned as a NumPy array of type float32.
        """
        # 1. Initialize episode steps counter to zero
        self.env_steps = 0
        self.episode_steps = 0
        # 2. Generates a random action within the range [-1,1] for each action parameter
        if seed is not None:
            np.random.seed(seed)
        random_action = np.random.uniform(-1,1,self.n_actions)
        # 3. Evaluate the environment with the generated random action
        # print(f"\n Resetting environment with random action: {random_action}")
        self.evaluate(random_action)
        # 4. Resets the episode score to zero
        self.score_history.clear
        #return normalized observation with fp32 np.array
        return np.array(list(self.cur_norm_specs.values()), dtype= np.float32)
        
    
    def step(self, action):
        """
        TODO: Perform a single step in the environment using the given action.

        This function should:
        1. Evaluate the given action.
        2. Compute the reward and check if the hard constraints are satisfied.
        3. Update the current observation.
        4. Append the reward to the reward history and update the score.
        5. Create output directories if they do not exist.
        6. If the goal state is reached, plot the running maximum reward.
        7. Increment the environment and episode step counters.
        8. Check if the maximum steps per episode have been reached, setting the done flag if true.
        9. Every 10 steps, update the score history, also plot the learning curve, and reset the score.
        10. Return the current observation, reward, done flag, and additional information.

        Args:
            action: The action to be taken in the environment. An array of values between -1 and 1.

        Returns:
            tuple: A tuple containing:
            - ob (np.ndarray): The current observation of the environment.
            - reward (float): The reward obtained from taking the action.
            - done (bool): A flag indicating whether the episode has ended.
            - info (dict): Additional information, including whether the goal state was reached.
        """
        #1. Evaluate the given action
        self.evaluate(action)
        #2. Compute the reward and check if the hard constraints are satisfied
        reward, hard_satisfied = self.reward_computation(self.cur_norm_specs)
        #3. Update the current observation
        ob = np.array(list(self.cur_norm_specs.values()), dtype=np.float32)
        #4. Append the reward to the reward history and update the score
        self.reward_history.append(reward)
        self.reward_history[-1] += reward
        #5. Create output directories if they do not exist
        output_dir = os.path.join(os.getcwd(), 'output_figs',str(self.run_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #6. If the goal state is reached, plot the running maximum reward
        if hard_satisfied:
            print("\n hard satisfied, plot reward")
            self.save_solution(reward)
            plot_running_maximum(self.reward_history,self.run_id)
            
        #7. Increment the environment and episode step counters
        self.env_steps += 1
        self.episode_steps += 1
        #8. Check if the maximum steps per episode have been reached, setting the done flag
        done = False
        print(f" Reward: {reward}, Hard satisfied: {hard_satisfied}")
        if self.env_steps >= self.max_steps_per_episode:
            done = True
        #9. Every 10 steps, update the score history, also plot the learning curve
        self.counter += 1
        if (self.counter) % 10 == 0:
            self.score_history.append(self.reward_history[-1])
            plotLearning(self.score_history,run_id=self.run_id)
            self.reward_history[-1] = 0.0
        #10. Return the current observation, reward, done flag, and additional information
        info = {'hard_satisfied': hard_satisfied}
        return ob, reward, done, info
        
    
    def reward_computation(self, norm_specs):
        """
        TODO: Compute the reward based on normalized specifications and hard constraints.

        Args:
            norm_specs (dict): A dictionary containing normalized specifications.

        Returns:
            tuple: A tuple containing:
            - reward (float): The computed reward value.
            - hard_satisfied (bool): A boolean indicating whether the hard constraints are satisfied.

        The function performs the following steps:
        1. Initialize the reward to 0.0 and hard_satisfied to False.
        2. Iterate over the hard constraints and adjust the reward based on the specifications.
        3. Check if all hard constraints are satisfied or the total reward passes the success threshold of 0.
            - If so, set hard_satisfied to True, add a bonus reward (+0.3), and adjust the reward based on optimization targets.
            - If not, add weighted reward components of the optimization targets.
        4. Return the computed reward and the hard_satisfied flag.
        """
        # 1. Initialize the reward to 0.0 and hard_satisfied to False.
        reward = 0.0
        reward_target = 0.0
        reward_hard = 0.0
        hard_satisfied = False
        # 2. Iterate over the hard constraints and adjust the reward based on the specifications.
        for name in self.hard_constraints:
            if name == 'noise': #minimize noise
                reward_hard -= max(0.0, norm_specs[name]*self.spec_weights[name])
            else: #maximize gain,pm,ugbw and slew rate
                reward_hard += min(0.0, norm_specs[name]*self.spec_weights[name])
        for name in self.optimization_targets:
                reward_target -= norm_specs[name]*self.spec_weights[name]
        # 3. Check if all hard constraints are satisfied or the total reward passes the success
        if reward_hard >= self.success_threshold: #all hard constraints are satisfied
            hard_satisfied = True
            reward = 0.3+ reward_target
        else: # some hard constraints are not satisfied
            hard_satisfied = False
            reward = reward_hard + 0.05*reward_target
        # 4. Return the computed reward and the hard_satisfied flag.
        return reward, hard_satisfied

def save_solution(self, reward):
        folder = './solutions'
        os.makedirs(folder, exist_ok=True)
        csv_path = os.path.join(folder, 'solutions.csv')

        # combine specs dict
        specs_dict = {**self.real_specs}

        # create a row with 'Specs' and 'reward'
        row = {
            'Params': json.dumps({k: float(v) for k, v in self.param_values.items()}),
            'Specs': json.dumps({k: float(v) for k, v in specs_dict.items()}),  # dict as string
            'reward': float(reward)
        }

        # append to CSV
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Params','Specs', 'reward'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

if __name__ == '__main__':
    env = CircuitEnv(
        circuit_name='TwoStage', 
        run_id=0, 
        simulator='ngspice', 
        success_threshold=0.0
        )
    print("Initial actions:",env.action_space)
    print("Initial observation:",env.observation_space)

    ob = env.reset()
    print("Initial observation: ", ob)
    print("Initial parameters: ", env.param_values)
    print("Initial specs: ", env.real_specs)

    action = np.random.uniform(-1, 1, [env.n_actions])
    ob, reward, done, info = env.step(action)
    print("Next parameters: ", env.param_values)
    print("Next observation: ", ob)
    print("Next specs: ", env.real_specs)
    print("Reward: ", reward)
    print("Done: ", done)
    

