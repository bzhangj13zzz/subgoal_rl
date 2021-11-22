"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 25 Feb 2021
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
import numpy as np

from agents.base_agent import baseAgent
from utils import discounted_return

# pdb.Pdb.complete = rlcompleter.Completer(locals())
# pdb.set_trace() 


# =============================== Variables ================================== #


# ============================================================================ #


class random_agent(baseAgent):

    def __init__(self, config=None):
        super(random_agent, self).__init__(config=config)
        self.entropy = -1

    def get_action(self, t, x):

        a = np.random.choice([_ for _ in range(self.num_actions)])

        return a

    def compute_val_fn(self, buff_rt=None):

        # ------- Emperical Return ------- #
        tot_time = buff_rt.shape[0]
        empRet = discounted_return(buff_rt)
        empRet = empRet.reshape(tot_time, 1)

        # --- Compute value
        return  empRet

    def log_agent(self, ep):
        pass

    def train(self, x_mem=None, v_mem=None):
        pass

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
