"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 01 Jul 2021
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
from parameters import PY_ENV, AGENT_NAME


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

def main():

    # os.system(PY_ENV + " -m scripts.train --env MiniGrid-TwoRooms-v0 --algo ppo")
    os.system(PY_ENV + " -m scripts.train --env MiniGrid-TwoRooms-v0 --algo ppo")

# =============================================================================== #

if __name__ == '__main__':
    main()
