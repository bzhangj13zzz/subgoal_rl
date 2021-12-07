import sys

sys.dont_write_bytecode = True
import os
from pprint import pprint
import time
from ipdb import set_trace
import pdb
import rlcompleter
import numpy as np
from agents.base_agent import baseAgent
from agents.networks import actor_nw
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, HORIZON, ENTROPY_WEIGHT, VERBOSE, SEED, TOTAL_AGENTS
import torch as tc
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import discounted_return

class subgoal_conditioned_low_policy(baseAgent):

    def __init__(self, config=None):
        super().__init__(config=config)

        self.actor = None
        if LOAD_MODEL:
            self.lg.writeln("-----------------------")
            self.lg.writeln("Loading Old Model")
            self.lg.writeln("-----------------------")
            self.actor = tc.load(self.pro_folder + '/load_model'+ '/model_actor_'  + self.agent_name +".pt")
            self.actor.eval()
        else:
            # Policy Network
            ip_dim = self.obs_dim * 2 # Concatenate the (sub)goal state
            self.actor = actor_nw(ip_dim, self.num_actions)
            self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        
    def get_action(self, t, x):
        pi_logit, _ = self.actor(x)
        probs = F.softmax(pi_logit)
        sampler = Categorical(probs)
        action = int(sampler.sample())
        return action
    
    def train(self, x_mem=None, v_mem=None):

        # ----- Policy Train
        input = x_mem

        # ---- val fn
        val = tc.tensor(v_mem).float()
        pi_logit, log_pi = self.actor(input)
        action_probs = F.softmax(pi_logit)
        dist = Categorical(action_probs)
        entropy = dist.entropy().reshape(input.shape[0], 1)
        op1 = tc.mul(log_pi, val)
        op2 = tc.add(op1, ENTROPY_WEIGHT * entropy)
        pi_loss = -(tc.mean(op2))
        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()
        self.entropy = entropy.mean().data