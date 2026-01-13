# -*- coding: utf-8 -*-
"""
training pipeline
"""
#%%
import vehicle_env as vehicle
import pdb
import config as config
import sac_RL as RL
import torch
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
import os
import numpy as np
from mcap_logger import McapLogger, sec_to_ns
import mcap_utils as mc
import time
import numpy as np

#%%
# Create environment
env = vehicle.VehicleEnv()
# pdb.set_trace()
############################
# RL agent.
############################
# Batch size.
batch_size = config.batch_size
# Discount factor.
gamma = config.gamma
alpha = 0.9
# agent = RL.sac(batch_size = batch_size, gamma=gamma,  alpha = alpha,\
#                lr= 1e-5, lr_lstm = 8.10526316e-05)
agent = RL.sac(env, batch_size = batch_size, gamma=gamma,  alpha = alpha,\
               lr= config.learning_rate, polyak=0.995)
ep_itr = config.episodes
traj_itr = config.max_steps
# Vector of learning rate.
ep_alpha = 10
# alpha_rate_1 = np.linspace(0.9,0.1,num = int(ep_itr/ep_alpha)-100)
# alpha_rate_2 = np.linspace(0.1,0.3,num = 100)
# alpha_rate = np.concatenate((alpha_rate_1,alpha_rate_2),axis=0)
alpha_rate = np.linspace(0.9,0.2,num = 250)


path = os.getcwd()
dir_save = path  +  '/SAC'
if not os.path.exists(dir_save):
     os.makedirs(dir_save)
    
model_SAC = dir_save + '/' +str(config.n_agent) + '_agent_final'


# dir_load = path  +  '/SAC' + '/' +str(config.n_agent) + '_agent_v3'
# agent.ac.load_state_dict(torch.load(dir_load))
# # Saving the weights.
torch.save(agent.ac.state_dict(), model_SAC)


#############################
# Save vectors.
#############################en()
# Empty vector for trajectory.
state_str = np.zeros((ep_itr,traj_itr+1,config.n))
state_str_t = np.zeros((ep_itr,2))

# Reward vector.
traj_reward = []
# Episode length
ep_len = []
# Local counter.
l_cnt = -1

#%% Update loop
agent.ac.train()   # set the module in training mode

# ---------------- MCAP logger (one file for whole training run) ----------------
logger = McapLogger("sac_training_final.mcap")

# Schemas and channels (reuse what you already use in single_agent_testing_MCAP_v2)
state_schema_id = logger.register_schema("RobotState", mc.STATE_SCHEMA)
ctrl_schema_id  = logger.register_schema("RobotControl", mc.ACT_SCHEMA)
scene_schema_id = logger.register_schema("foxglove.SceneUpdate", mc.SCENEUPDATE_SCHEMA_MIN)
ep_schema_id    = logger.register_schema("Episode", mc.EPISODE_SCHEMA)

logger.register_channel("/robot/state", state_schema_id)
logger.register_channel("/robot/control", ctrl_schema_id)
logger.register_channel("/robot/scene", scene_schema_id)
logger.register_channel("/robot/episode", ep_schema_id)

# Add schemas/channels for reward + training metrics
train_schema_id = logger.register_schema("TrainMetrics", mc.TRAIN_SCHEMA)   # define this in mcap_utils.py
rew_schema_id   = logger.register_schema("StepReward", mc.REWARD_SCHEMA)    # define this in mcap_utils.py
logger.register_channel("/rl/train", train_schema_id)
logger.register_channel("/rl/reward", rew_schema_id)

# Anchor wall-clock timestamp to convert sim-time -> monotonically increasing log time
t0_wall_ns = sec_to_ns(time.time())
t_ns = t0_wall_ns

idx_alpha = 0
# Start the episode loop.
for episode in range(ep_itr):
    print('Episode {}'.format(episode))
    # Initialize environment
    env.reset()
    
    # Store the target state
    state_str_t[episode,:] = env.target
    # Initial state.
    state_str[episode,0,:] = env.state
    state = env._get_obs()
    # Reward vector.
    episode_reward = []
    # Action vector.
    store_act = []
    
    episode_id = int(episode)
    t0_sim = float(env.current_time)  # if your env has current_time; otherwise use 0.0
    target = np.asarray(env.target).ravel()

    logger.write("/robot/episode", t0_wall_ns, {
        "episode_id": episode_id,
        "target": [float(v) for v in target],
        "t0": float(t0_sim),
    })

    update_idx = 0   # ← initialize here

  
    if episode > ep_alpha and np.mod(episode,ep_alpha)==0:
        try:
            agent.alpha = float(alpha_rate[idx_alpha])
            idx_alpha += 1
        except IndexError:
            pass
    # Start the time trajectory loop.
    for step in range(traj_itr):
        if episode < config.data_collection_episodes:
            action = np.random.uniform(low=-1, high=1, size=2)
        else:
            # Call from the SAC agent. 
            action = agent.get_action(state.T)
            action = action.cpu().numpy()
        
        # Store action.
        store_act.append(action)

         
        # Environment step.        
        new_state, reward, done,_,info = env.step(action)
    
        # Store experience to replay buffer
        agent.replay_buffer.store(state, store_act[-1], reward, new_state, done)
                
        episode_reward.append(reward) # store reward
        state = new_state

        # Store the state,
        state_str[episode,step+1,:] = env.state

        # Convert sim time -> log timestamp
        t_sec = float(getattr(env, "current_time", step))   # fallback: step
        # t_ns  = t0_wall_ns + sec_to_ns(t_sec - t0_sim)
        t_ns += sec_to_ns(config.delta_t)

        s = np.asarray(env.state).ravel()
        w_d = np.asarray(getattr(env, "wheel_delta", np.zeros(4))).ravel()

        # State
        logger.write("/robot/state", t_ns, {
            "episode_id": episode_id,
            "t": t_sec,
            "x": float(s[0]),
            "y": float(s[1]),
            "yaw": float(s[2]),
            "v": float(s[3]) if s.size > 3 else 0.0,
            "state": [float(v) for v in s],
        })

        # Control
        u = np.asarray(action).ravel()
        msg_u = {
            "episode_id": episode_id,
            "t": t_sec,
            "u_sac": [float(v) for v in u],
            "delta_fl": float(w_d[0]) if w_d.size > 0 else 0.0,
            "delta_fr": float(w_d[1]) if w_d.size > 1 else 0.0,
            "delta_rl": float(w_d[2]) if w_d.size > 2 else 0.0,
            "delta_rr": float(w_d[3]) if w_d.size > 3 else 0.0,
        }
        logger.write("/robot/control", t_ns, msg_u)

        # Reward
        logger.write("/rl/reward", t_ns, {
            "episode_id": episode_id,
            "step": int(step),
            "t": t_sec,
            "reward": float(reward),
            "done": bool(done),
        })

        # Scene (Foxglove)
        scene_msg = mc.make_vehicle_scene_update(
            x=float(s[0]),
            y=float(s[1]),
            yaw=float(s[2]),
            wheel_delta=w_d,
            target=np.asarray(env.target).ravel(),
            L=float(config.L),
            W=float(config.W),
        )
        logger.write("/robot/scene", t_ns, scene_msg)
        
        
        if done:
            # pdb.set_trace()
            break

    # Update handling
    if episode >= config.data_collection_episodes:
        
    #and episode % update_every == 0:
    #and step % update_every == 0:
        
        for j in range(config.update_every):
            batch = agent.replay_buffer.sample_batch(batch_size) # Sample a batch of data from replay buffer
           
            batch['obs'] = batch['obs'].to(device) # Observations
            batch['obs2'] = batch['obs2'].to(device) # Next observations
            batch['act'] = batch['act'].to(device) # Actions
            batch['rew'] = batch['rew'].to(device) # Rewards
            batch['done'] = batch['done'].to(device) # Done flags
            
            metrics = agent.update(data=batch)  # Update the SAC agent

            # t_sec = float(getattr(env, "current_time", traj_itr))  # or step * env.dt
            # t_ns  = t0_wall_ns + sec_to_ns(t_sec - t0_sim)
            

            logger.write("/rl/train", t_ns, {
                "episode_id": int(episode_id),
                "update_idx": int(update_idx),
                "t": t_sec,
                "critic_loss": metrics["critic_loss"],
                "actor_loss": metrics["actor_loss"],
            })

            update_idx += 1
           
        # pdb.set_trace()
    traj_reward.append(np.sum(episode_reward))
    print('Episode Reward: {}'.format(np.sum(episode_reward)))

    ep_len.append(step)
    if np.mod(episode, config.data_collection_episodes) == 0 and episode >config.data_collection_episodes:
        
        torch.save(agent.ac.state_dict(), model_SAC) # Save the weights.

logger.close()
#%%
