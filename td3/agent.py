import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Check if Pytorch detects a Cuda-GPU, if true use the GPU else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = GPU or CPU

class Actor(nn.Module):
	"""
	Actor Network 策略网络
	inputs: state
	outputs: action
	
	Structure:
	- Two Fully connected layer with 256 units + ReLU
	- Output layer with tanh activation scaled by max_action
	"""
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action # bound for action output (e.g. parameter)
		

	def forward(self, state): 
            # print(f"\n state in actor forward: {state}")
            # print("l1.weight mean:", self.l1.weight.mean().item(),
            #     "std:", self.l1.weight.std().item())
            # print("Any NaN in l1.weight:", torch.isnan(self.l1.weight).any().item())
            # print("Any Inf in l1.weight:", torch.isinf(self.l1.weight).any().item())
            # print("l1.bias mean:", self.l1.bias.mean().item(),
            #     "Any NaN in bias:", torch.isnan(self.l1.bias).any().item())
            a = self.l1(state)
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	"""
	Critic Network 价值网络
	inputs: state, action 评估状态-动作的价值
	outputs: Q1, Q2
	
	Two Critic networks(Q1,Q2) to mitigate overestimation bias
	    Single Q will accumulate errors over time and lead to learning out of the right direction and make the converge unstable
		When use two critics, we take the minimum value between them as the target Q value, which reduces overestimation bias.
		
	
	"""
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action): # for critic update
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action): # only for actor update
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
    def __init__(self, state_dim, action_space, args):
		#arguments needed for TD3 agent
        self.max_action = float(action_space.high[0])
        self.discount = args.gamma
        self.tau = args.tau
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.expl_noise = args.noise_sigma # exploration noise added to action selection, 用于训练初期的探索
        self.policy_freq = args.actor_update_interval #update frequency of actor network

        #networks and optimizers
        self.action_dim = action_space.shape[0]
        #Actor update for each training
        self.actor = Actor(state_dim, self.action_dim, self.max_action).to(device)
        #Actor target: copy of Actor but update only when the critic update
        self.actor_target = copy.deepcopy(self.actor) # critic require stable targets to update but the update of actor is too fast
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.pi_lr)

        self.critic = Critic(state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.q_lr)

        self.total_it = 0


    def select_action(self, state):
        """
        TODO: Implement the action selection mechanism for the TD3 agent
        
        Steps to implement:
        1. Convert the state to a tensor if it's a numpy array or CPU tensor
                torch.Tensor = np.ndarray + device + autograd 
        2. Get the deterministic action from the actor network (mu)
        3. Add "exploration noise" using np.random.normal with:
           - mean = 0
           - std = self.max_action * self.expl_noise
           - size = self.action_dim
        4. Clip the final action to be between -self.max_action and self.max_action
        5. Return the action as a numpy array
        
        Args:
            state: The current state observation (numpy array or tensor)
            
        Returns:
            action: The selected action as a numpy array
        """
        print("\nSelecting action...")
        # 1. Convert the state to a tensor if it's a numpy array or CPU tensor
            # if it is a numpy array from the environment
        if isinstance(state,np.ndarray):
            #[1,2,3,4] four elements -> [[1,2,3,4]] batch size 1 , each with 4 features
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            #state = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(device)
        elif isinstance(state,torch.Tensor):
            state = state.to(device) # put cpu tensor to gpu tensor
            if state.dim() == 1:
                state = state.unsqueeze(0)
        
        #2. Get the deterministic action from the actor network (mu) 
            # gpu tensor -> cpu array 
            # flatten: convert 2D -> 1D   
        print(f"\n state in select_action: {state}")
        action = self.actor(state).cpu().data.numpy().flatten()
        print(f"\n action before noise: {action}")
        # 3. Add Gaussian exploration noise
        noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        action = action + noise

        # 4. Clip to valid range
        action = np.clip(action, -self.max_action, self.max_action)
        return action
            


    def update_parameters(self, memory_batch, update):
        """
        TODO: Implement the TD3 learning algorithm
        
        Steps to implement:
        1. Increment total_it counter
        2. Unpack the memory_batch tuple into state, action, next_state, reward, not_done
        
        3. Compute target actions (with torch.no_grad()):
           - Add clipped "policy noise" to target policy
           - Get next actions from target actor network
           - Clip target actions to valid range
        
        4. Compute target Q-values (with torch.no_grad()):
           - Get Q-values from both target critics
           - Take the minimum of both Q-values
           - Compute TD target using reward + discount * min Q-value * not_done
        
        5. Compute current Q-values and critic loss:
           - Get current Q-values from both critics
           - Compute MSE loss between current and target Q-values for both critics
        
        6. Update critics:
           - Zero gradients
           - Backpropagate critic loss
           - Optimize critic
        
        7. Delayed policy update (if total_it % policy_freq == 0):
           - Compute actor loss using first critic's Q-values
           - Update actor
           - Update target networks using soft update (τ)
        
        Args:
            memory_batch: Tuple of (state, action, reward, next_state, not_done) tensors
            update: Update step (not used in this implementation)
        """
        print("\nUpdating parameters...")
        #1. Increment total_it counter
        self.total_it +=1
        #2. Unpack the memory_batch tuple into state, action, next_state, reward, not_done
        state,action,reward,next_state,not_done = memory_batch
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        not_done = not_done.to(device)

        #3. Compute target actions (with torch.no_grad()):
        # with torch.no_grad() means we don't need to track gradients for backprop
        #   - Add clipped "policy noise" to target policy
        #   - Get next actions from target actor network
        #   - Clip target actions to valid range      
        with torch.no_grad():
            policy_noise = (torch.randn_like(action)*self.policy_noise).clip(-self.noise_clip,self.noise_clip)
            next_action= (self.actor_target(next_state)+policy_noise).clip(-self.max_action,self.max_action)
        #4. Compute target Q-values (with torch.no_grad()):
            target_Q1 , target_Q2 = self.critic_target(next_state,next_action)
            min_target_Q = torch.min(target_Q1,target_Q2)
            target_Q = reward + self.discount * min_target_Q * not_done


        #5. Compute current Q-values and critic loss:
        current_Q1,current_Q2= self.critic(state,action)
        critic_loss_Q1 = F.mse_loss(current_Q1,target_Q)
        critic_loss_Q2 = F.mse_loss(current_Q2,target_Q)
        critic_loss = critic_loss_Q1 + critic_loss_Q2

        #6. Update critics:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #7. Delayed policy update (if total_it % policy_freq == 0):
        if self.total_it % self.policy_freq ==0:
            # - Compute actor loss using first critic's Q-values
            actor_loss = -self.critic.Q1(state,self.actor(state)).mean()
            # - Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # - Update target networks using soft update (τ)
            for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                target_param.data.copy_((1-self.tau)*param.data + self.tau*target_param.data)
            for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                target_param.data.copy_((1-self.tau)*param.data + self.tau*target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        