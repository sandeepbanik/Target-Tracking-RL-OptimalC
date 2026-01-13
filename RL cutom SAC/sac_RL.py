from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam, RMSprop
import gym
import core_SAC as core
import torch.nn.functional as F
import random
from torch.distributions.normal import Normal
import pdb
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
#%%

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        #only for estimated states
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs 
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class sac:
    def __init__(self, env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            replay_size=int(1e6), gamma=0.99, 
            polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100):
        
       
    
        # Discount factor.
        self.gamma = gamma
        # Alpha
        self.alpha = alpha
        # Learning rate.
        self.lr = lr
        # Polay 
        self.polyak = polyak

        #batch size
        self.batch_size = batch_size
        # Call the environment.
        self.env = env
    
        obs_dim_buffer = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]
    
        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space,\
                               self.env.action_space, **ac_kwargs).to(device)
        self.ac_targ = deepcopy(self.ac)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
    
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim_buffer,\
                                                  act_dim=act_dim, size=replay_size)
    
        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # Set up optimizers for policy and q-function
        # pdb.set_trace()
       
        self.pi_params = list(self.ac.pi.net.parameters())+ \
           list(self.ac.pi.mu_layer.parameters()) + \
              list(self.ac.pi.log_std_layer.parameters())
        
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr, \
                                 eps = 1e-4, weight_decay= 1e-2) 
        
        
        #self.pi_optimizer = Adam(self.pi_params, lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.pi_optimizer,\
        #                                                  step_size= 150,\
        #                                                      gamma= 1e-2)
       
          
         # Average reward
        
        # self.avg_r = torch.zeros(1).mean().to(device)
        # # self.alpha_r = 0.0001*torch.ones((batch_size)).to(device)
        # self.alpha_r = 0.0001
        

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'].to(device),\
            data['act'].to(device), data['rew'].to(device), \
                data['obs2'].to(device), data['done'].to(device)
    
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)
       
        # Bellman backup for Q functions
        # pdb.set_trace()
        with torch.no_grad():
            # Target actions come from *current* policy
            
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
          
           
            
            backup = r + self.gamma *(1-d) * (q_pi_targ - \
                                                 self.alpha * logp_a2)
    
            

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
       
      
        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
      
        o = data['obs']
        
       
        pi, logp_pi = self.ac.pi(o)
        
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
       

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self,data):
       
        # First run one gradient descent step for Q1 and Q2     
        self.q_optimizer.zero_grad()
        # pdb.set_trace()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()


        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        # pdb.set_trace()
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
      
        #torch.nn.utils.clip_grad_norm_(self.ac.pi.net[0].parameters(), 40.0)
        self.pi_optimizer.step()
        # self.scheduler.step()
        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return {
                    "critic_loss": float(loss_q.item()),
                    "actor_loss": float(loss_pi.item()),
                }


    def get_action(self,o, deterministic=False):
        # pdb.set_trace()
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)