from parameters import ENV_NAME, LOAD_MODEL, LEARNING_RATE, EPISODES, HORIZON, AGENT_NAME, RENDER, BATCH_SIZE, SAVE_MODEL, SHOULD_LOG, SEED
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
from gym_minigrid.envs.rooms_james import TwoRoomsEnv
from gym_minigrid.envs.fourrooms import FourRoomsEnv
from agents.vpg import vpg

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
    env.seed(SEED)
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
    lr = 0.001
    discount_rate = 0.9
    def train(env):
        # q_table = np.zeros((env.height, env.width, 4, env.action_space.n))
        q_table = np.zeros((env.height, env.width, 4, 3))
        for eps in tqdm(range(EPISODES)):
            env.seed(SEED)
            obs = env.reset()
            for step in range(HORIZON):
                cur_agent_pos_x, cur_agent_pos_y, cur_agent_dir = obs
                if np.all(q_table[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir] == q_table[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir][0]) or random.random() < epsilon:
                    # action = env.action_space.sample()
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(q_table[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir])
                new_obs, reward, done, _ = env.step(action)
                new_agent_pos_x, new_agent_pos_y, new_agent_dir = new_obs
                obs = new_obs
                q_table[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir][action] += lr * (reward + discount_rate * np.max(q_table[new_agent_pos_x][new_agent_pos_y][new_agent_dir]) - q_table[cur_agent_pos_x][cur_agent_pos_y][cur_agent_dir][action])

                if done:
                    break

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
    t = rollout(env, q_table, True)
    print(f"Took {t} to achieve goal.")
if __name__ == '__main__':
    main()
