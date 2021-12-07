from gym_minigrid.envs.rooms_james import TwoRoomsEnv
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
from agents.networks import critic_nw

from ipdb import  set_trace
import matplotlib.pyplot as plt



# =============================== Variables ================================== #


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

    env = None
    if "tworoom" in ENV_NAME:
        # env = gym.make('MiniGrid-TwoRooms-v0')
        env = TwoRoomsEnv()
    elif "threeroom" in ENV_NAME:
        register(id='minigrid-threeroom-v0',
    entry_point='gym_minigrid.envs:ThreeRoomEnvNxN',)
        env = gym.make('minigrid-threeroom-v0')
    elif "fourroom" in ENV_NAME:
        # register(id='minigrid-fourroom-v0', entry_point='gym_minigrid.envs:FourRoomEnvNxN',)
        env = gym.make('MiniGrid-FourRooms-v0')
    return env

def rollout(env, agent, render=False):
    env.reset()

    for t in range(HORIZON):
        cur_agent_pos_x, cur_agent_pos_y = env.agent_pos
        cur_agent_dir = env.agent_dir
        if render:
            env.render(mode='human', highlight=False)
            time.sleep(0.1)
        action = action = np.argmax(agent[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir])
        _, _, done, _ = env.step(action)
        if done:
            return t
    return HORIZON
    



def main():
    import random
    from tqdm import tqdm

    epsilon = 0.2
    lr = 0.1
    discount_rate = 0.9

    def train(env):
        x_mem = np.empty((0, 3))
        v_mem = np.empty((0, 1))

        dqn = critic_nw(3, 3)
        # q_table = np.zeros((env.height, env.width, 4, env.action_space.n))
        # q_table = np.zeros((GRID, GRID, 4, 3))
        for eps in tqdm(range(EPISODES)):
            env.reset()
            cml_reward = 0

            buff_r = np.empty((0, 1))
            buff_x = np.empty((0, 3))

            cur_agent_pos_x, cur_agent_pos_y = env.agent_pos
            cur_agent_dir = env.agent_dir
            obs = [cur_agent_pos_x, cur_agent_pos_y, cur_agent_dir]

            s_traj = [obs]

            for step in range(HORIZON):
                q_vals = dqn(obs)

                if np.all(q_vals == q_vals[0]) or random.random() < epsilon:
                    # action = env.action_space.sample()
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(q_vals)

                _, reward, done, _ = env.step(action)

                cml_reward += reward

                new_agent_pos_x, new_agent_pos_y = env.agent_pos
                new_agent_dir = env.agent_dir
                new_obs = [new_agent_pos_x, new_agent_pos_y, new_agent_dir]

                buff_x = np.vstack((buff_x, new_obs))
                buff_r = np.vstack((buff_r, reward))

                if done:
                    break
            
            x_mem = np.vstack((x_mem, buff_x))
            v_mem = np.vstack((v_mem, val))

        return q_table

    # def rollout(env, q_table):
    #     env.reset()
    #     for step in range(HORIZON):
    #         cur_agent_pos_x, cur_agent_pos_y = env.agent_pos
    #         cur_agent_dir = env.agent_dir
    #         action = np.argmax(q_table[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir])
    #         _, reward, done, _ = env.step(action)
    #         if done:
    #             print(f'Achieved goal at step {step}')
    #             break

        
    # env = gen_wrapped_env('MiniGrid-FourRooms-v0', max_steps)
    env = get_env()
    # env = gym.make('MiniGrid-Empty-5x5-v0')
    env.max_steps = HORIZON
    # env = gym.make('MiniGrid-Empty-5x5-v0')
    init_img = env.render('rgb_array')
    plt.imshow(init_img)

    q_table = train(env)
    rollout(env, q_table, True)
if __name__ == '__main__':
    main()
