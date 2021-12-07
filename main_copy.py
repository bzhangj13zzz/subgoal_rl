"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 12 Nov 2021
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
from parameters import ENV_NAME, LOAD_MODEL, LEARNING_RATE, EPISODES, HORIZON, AGENT_NAME, RENDER, GRID, BATCH_SIZE, SAVE_MODEL, SHOULD_LOG, SEED
import numpy as np
np.random.seed(SEED)
import sys
sys.dont_write_bytecode = True
import os
import gym
import time
from gym_minigrid.register import register
from utils import deleteDir, log, logRunTime, get_one_hot, subgoal_model, dumpDataStr
from agents.random import random_agent
from agents.vpg import vpg
from gym_minigrid.envs.fourrooms import FourRoomsEnv
from gym_minigrid.envs.rooms_james import TwoRoomsEnv

from ipdb import  set_trace
import matplotlib.pyplot as plt



# =============================== Variables ================================== #
seed = 1337

# ============================================================================ #

def init(pro_folder, dir_name):

    os.system("mkdir "+pro_folder+"/log/")
    os.system("mkdir "+pro_folder+"/log/" + dir_name)
    os.system("rm "+pro_folder+"/log/" + dir_name+"/*.png")

    if LOAD_MODEL is False:
        deleteDir(pro_folder+"/log/"+dir_name + "/plots/")
    os.system("cp "+pro_folder+"/"+"parameters.py "+pro_folder+"/log/"+dir_name+"/")
    os.system("mkdir "+pro_folder+"/log/"+dir_name+"/plots")
    os.system("mkdir "+pro_folder+"/log/" + dir_name + "/model")

def init_agent(agent_name, config):

    agent = None
    if agent_name == "random":
        agent = random_agent(config=config)
    elif agent_name == "vpg":
        agent = vpg(config=config)
    else:
        print("Agent not found !")
        exit()
    return agent

def get_env():
    return TwoRoomsEnv()
    # env = None
    # if "tworoom" in ENV_NAME:
    #     # env = gym.make('MiniGrid-TwoRooms-v0')
    #     env = TwoRoomsEnv()
    # elif "threeroom" in ENV_NAME:
    #     register(id='minigrid-threeroom-v0',
    # entry_point='gym_minigrid.envs:ThreeRoomEnvNxN',)
    #     env = gym.make('minigrid-threeroom-v0')
    # elif "fourroom" in ENV_NAME:
    #     # register(id='minigrid-fourroom-v0', entry_point='gym_minigrid.envs:FourRoomEnvNxN',)
    #     env = gym.make('MiniGrid-FourRooms-v0')
    # return env

def rollout(env, agent, render=False):
    obs = env.reset()

    for t in range(HORIZON):
        if render:
            env.render(mode='human', highlight=False)
            time.sleep(0.1)
        
        cur_agent_pos_x, cur_agent_pos_y = env.agent_pos
        cur_agent_dir = env.agent_dir
        obs = [cur_agent_pos_x, cur_agent_pos_y, cur_agent_dir]

        action = agent.get_action(t, obs)
        _, _, done, _ = env.step(action)

        cur_agent_pos_x, cur_agent_pos_y = env.agent_pos
        cur_agent_dir = env.agent_dir
        obs = [cur_agent_pos_x, cur_agent_pos_y, cur_agent_dir]

        if done:
            return t
    return HORIZON
    


def main():

    # --------- Init
    pro_folder = os.getcwd()
    dir_name_hash = {}
    dir_name_hash.update({"lr": str(LEARNING_RATE)})
    tstr = AGENT_NAME
    for k in dir_name_hash:
        tstr += "_"+k+"_"+dir_name_hash[k]
    dir_name = tstr
    init(pro_folder, dir_name)
    if LOAD_MODEL:
        lg = log(pro_folder + "/log/" + dir_name + "/log_inf"+".txt")
    else:
        lg = log(pro_folder + "/log/"+dir_name+"/log"+".txt")

    # ------ Subgoals
    # subgoal_list = [22, 20, 24, 40, 58, 56, 60, 70]
    # subgoal_list = [22, 20, 24, 40, 58, 56, 60]
    # sg = subgoal_model(sg_list=subgoal_list)



    # ------- Environment
    env = get_env()
    env.max_steps = HORIZON
    env_image = env.render('rgb_array')
    # plt.imsave(f"{pro_folder}/log/{dir_name}/env_image.png", env_image)
    plt.imsave('env_image.png', env_image)

    # --------- Agents
    config = {}
    config['num_actions'] = 3
    config['obs_dim'] = 3 
    config['pro_folder'] = pro_folder
    config['dir_name'] = dir_name
    config['agent_name'] = AGENT_NAME
    config['lg'] = lg
    agent = init_agent(AGENT_NAME, config)

    # ----- Memory Buffer
    x_mem = np.empty((0, 3))
    v_mem = np.empty((0, 1))

    # ------- Start Simulation
    for ep in range(1, EPISODES + 1):
        if ep % SHOULD_LOG == 0:
            lg.writeln("\n# --------------------- #")
            lg.writeln("Episode: "+str(ep))
        
        obs = env.reset()

        rt_sum = 0

        # ---- Buffer
        buff_r = np.empty((0, 1))
        buff_x = np.empty((0, 3))

        # ---- state traj
        s_traj = [obs]

        for t in range(HORIZON):
            if RENDER:
                env.render(mode='human', highlight=False)
                time.sleep(0.1)
            action = agent.get_action(t, obs)
            obs_new, reward, done, _ = env.step(action)
            rt_sum += reward

            buff_x = np.vstack((buff_x, obs))
            buff_r = np.vstack((buff_r, reward))

            # (cur_agent_pos_x, cur_agent_pos_y), cur_agent_dir = obs_new

            obs = obs_new

            s_traj.append(obs)
            if done:
                break

        # -------- Subgoal Update
        # if rt_sum > 0:
            # sg.update_sg_trans(s_traj)


        # -------- Compute Value Function
        val = agent.compute_val_fn(buff_rt=buff_r)

        # ---------- Add in memory buffer
        x_mem = np.vstack((x_mem, buff_x))
        v_mem = np.vstack((v_mem, val))

        # ----------- Training
        if LOAD_MODEL is False and x_mem.shape[0] > BATCH_SIZE:
            agent.train(x_mem = x_mem, v_mem = v_mem)

            # ----- Clear Memory
            x_mem = np.empty((0, 3))
            v_mem = np.empty((0, 1))
            if ep % SAVE_MODEL == 0:
                agent.save_model(ep, rt_sum)

        # -------- Log
        if ep % SHOULD_LOG == 0:
            agent.log(ep, rt_sum)
            agent.log_agent(ep)
            lg.writeln("\n Return: " + str(rt_sum))
            lg.writeln("\n State_traj: " + str(s_traj))

            cml_steps = 0
            for i in range(10):
                cml_steps += rollout(env, agent)
            print(f"\nOn average took {cml_steps / 10} steps to achieve goal")
    
    rollout(env, agent, render=True)

    ppath = pro_folder+"/log/"+dir_name

    # sg.sg_tran_plot(ppath)
    # dumpDataStr(ppath+"/sg_model", sg.sg_model)


# =============================================================================== #

if __name__ == '__main__':
    main()
