"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 18 Nov 2021
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys

sys.dont_write_bytecode = True
import os
import time
from ipdb import set_trace
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.pg import PGTrainer
from gym_minigrid.register import register
import gym
from gym_minigrid.envs.rooms_james import TwoRoomsEnv
from rllib_agents.myagent import MyTrainer
# =============================== Variables ================================== #

import gym, ray
from ray.rllib.agents import ppo

# ============================================================================ #



def main():

    # Configure the algorithm.
    config = {
        "env": 'MiniGrid-TwoRooms-v0',
        "env_config": {},
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 1,
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        # `trainer.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": False,
        }
    }

    trainer = ppo.PPOTrainer(env=TwoRoomsEnv, config=config)
    # trainer = PGTrainer(env=TwoRoomsEnv, config=config)
    # trainer = MyTrainer(env=TwoRoomsEnv, config=config)

    while True:
        print(trainer.train())


    # Create our RLlib Trainer.
    # trainer = PPOTrainer(config=config)

# =============================================================================== #

if __name__ == '__main__':
    main()
