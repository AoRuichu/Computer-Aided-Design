import os
import torch
import numpy as np
from td3 import TD3, ReplayBuffer
from circuit_env import CircuitEnv
from datetime import datetime
import numpy as np
import os
from ngspice_interface import DUT as DUT_NGSpice

class Args:
    def __init__(self, **kwargs):
        self.seed = kwargs.get('seed', 123456)
        self.noise_sigma = kwargs.get('noise_sigma', ...)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        self.target_update_interval = kwargs.get('target_update_interval', 1)
        self.actor_update_interval = kwargs.get('actor_update_interval', 2)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.batch_size = kwargs.get('batch_size', ...)
        self.w = kwargs.get('w', ...)
        self.pi_lr = kwargs.get('pi_lr', ...)
        self.q_lr = kwargs.get('q_lr', ...)
        self.replay_size = kwargs.get('replay_size', 1_000_000)
        self.cuda = kwargs.get('cuda', False)
        self.no_discretize = kwargs.get('no_discretize', False)
        self.run_id = kwargs.get('run_id', datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))

class TD3Runner:
    def __init__(self, args=None):
        if args is None:
            args = Args()
        self.args = args
        # Set random seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # Initial environment
        self.env = CircuitEnv(run_id=self.args.run_id)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        # Intial agent
        self.agent = TD3(state_dim, self.env.action_space, self.args)
        # Initial pool for env
        self.env_pool = ReplayBuffer(state_dim, action_dim, max_size=self.args.replay_size)
        self.total_steps = 0

        self.rl_sizer_declaration = {
                                    "name": "rl_sizer_tool",
                                    "description": """A tool to optimally size circuit parameters using Reinforcement Learning.
                                                    The tool takes in an integer as the number of simulation budget steps, and returns the best
                                                    circuit parameters found within that budget.""",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "budget_steps": {
                                                "type": "integer",
                                                "description": "The number of simulation budget steps for the RL Sizer to use."
                                            }
                                        },
                                        "required": ["budget_steps"]
                                    }
                                }

    def train_after_warmup(self, budget_steps):
        while self.total_steps < budget_steps:
            obs = self.env.reset()
            done = False
            while not done:
                action = self.agent.select_action(obs)
                next_state, reward, done, _ = self.env.step(action)
                self.env_pool.push(obs, action, reward, next_state, done)
                obs = next_state
                self.train_policy()
                self.total_steps += 1
                # clean up the no_backup folder every 100 steps
                if self.total_steps % 100 == 0:
                    os.system('./clean.sh')

    def warmup_exploration(self):
        while self.total_steps < self.args.w:
            obs = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.env_pool.push(obs, action, reward, next_state, done)
                obs = next_state
                self.total_steps += 1

    def train_policy(self):
        state, action, reward, next_state, done = self.env_pool.sample(self.args.batch_size)
        batch = (state, action, reward, next_state, done)
        self.agent.update_parameters(memory=batch, update=self.total_steps)
    
    
    def convert_parameters_to_action(self, param_dict):
        action = []
        for i, name in  enumerate(self.env.dict_params.keys()):
            p_min = self.env.param_ranges[name]['min']
            p_max = self.env.param_ranges[name]['max']
            norm_value = 2 * (param_dict[name] - p_min) / (p_max - p_min) - 1
            action.append(norm_value)
        return np.array(action)
    
    def find_best_design_in_pool(self):
        best_reward = -float('inf')
        best_params = None
        for i in range(self.env_pool.size):
            _, action, reward, _, _ = self.env_pool.sample(1)
            if reward.squeeze()[0] > best_reward:
                best_reward = reward.squeeze()[0]
                # convert action back to real parameter values
                param_dict = {}
                for j, name in enumerate(self.env.dict_params.keys()):
                    p_min = self.env.param_ranges[name]['min']
                    p_max = self.env.param_ranges[name]['max']
                    real_value = ((action[0][j] + 1) / 2) * (p_max - p_min) + p_min
                    param_dict[name] = real_value.item()
                best_params = param_dict
        specs = self.env.simulate(best_params)
        return {"params": best_params, "specs": specs}

    def add_llm_experience_to_RL_memory(self, specs1, refined_params, specs2):
        state = self.env.normalize_specs(specs1)
        next_state = self.env.normalize_specs(specs2)
        # convert real param values to [-1, 1] range using the max/min values of each parameter in circuit_env.py
        action = self.convert_parameters_to_action(refined_params)
        # compute reward
        reward, _ = self.env.reward_computation(next_state)
        state = np.array(list(state.values()), dtype=np.float32)
        next_state = np.array(list(next_state.values()), dtype=np.float32)
        done = False # always False for manual experience addition
        self.env_pool.push(state, action, reward, next_state, done)



# ------------------------------------------- CIRCUIT EVALUATOR CLASS ---------------------------------------- ##
# This class may be needed for evaluating LLM manual parameter adjustments outside of the RL environment
class CircuitEvaluator():    
    def __init__(self, circuit_name='TwoStage', run_id='agentic_workflow', simulator='ngspice'):
        self.run_id = run_id
        self.circuit_name = circuit_name

        project_path = os.getcwd()
        yaml_directory = os.path.join(project_path, f"{simulator}_interface", 'files', 'yaml_files')
        circuit_yaml_path = os.path.join(yaml_directory, f'{circuit_name}.yaml')
        self.simulation_engine = DUT_NGSpice(circuit_yaml_path)
        self.pvt_corner = {'process': 'TT', 'voltage': 1.2, 'temp': 27}
    
    
    def simulate(self, params):
        new_netlist_path = self.simulation_engine.create_new_netlist(
            parameters=params,
            process=self.pvt_corner['process'],
            temp_pvt=self.pvt_corner['temp'],
            vdd=self.pvt_corner['voltage']
        )
        self.simulation_engine.simulate(new_netlist_path)
        return self.simulation_engine.measure_metrics()
