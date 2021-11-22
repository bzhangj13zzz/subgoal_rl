"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 03 Jun 2021
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys

sys.dont_write_bytecode = True
import os
from pprint import pprint
import time
from ipdb import set_trace
import pdb
import rlcompleter
from tensorboardX import SummaryWriter
import torch as tc
import numpy as np
from parameters import LARGE, TOTAL_AGENTS
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

class baseAgent(object):

    def __init__(self,  config=None):

        self.num_actions = config['num_actions']
        self.obs_dim = config['obs_dim']
        self.agent_name = config['agent_name']
        self.pro_folder = config['pro_folder']
        self.dir_name = config['dir_name']
        self.lg = config['lg']
        self.action_space = [ _ for _ in range(self.num_actions)]
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.max_return = -1 * LARGE
        self.entropy = -1
        self.actor = None

    def get_action(self, t, x):

        pass

    def log(self, ep, rt_sum):

        self.writer.add_scalar('Return/', rt_sum, ep)
        if self.entropy != -1:
            self.writer.add_scalar('Avg_Entropy/', self.entropy, ep)

    def save_model(self, ep, rt_sum):

        if rt_sum >= self.max_return:
            self.max_return = rt_sum
            tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor' + "_" + self.agent_name + ".pt")

    def train(self, x_mem = None, v_mem = None, n_mem = None, g_mem = None):

        pass



def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
