# ======= Main
PY_ENV = "/home/james/miniconda3/envs/sg_minigrid/bin/python"
SEED = 1
EPISODES = 10000
HORIZON = 100
LOAD_MODEL = False
BATCH_SIZE = 200
SAVE_MODEL = 100
AGENT_NAME = "random"
# AGENT_NAME = "vpg"
SHOULD_LOG = 100

# ===== Enviroment
RENDER = False
ENV_NAME = "tworoom"
IMAGE_OBS = False
GRID = 9
MID_WALL = True
TOTAL_AGENTS = 1

# ===== NN
DISCOUNT = 0.99
HIDDEN_DIM = 32
LEARNING_RATE = 1e-3
ENTROPY_WEIGHT = 0

# ===== Others
LARGE = 1e10
TINY = 1e-8
VERBOSE = True

# ========== System Environment Variables
ENV_VAR = True
OPENBLAS = 1
OMP = 1
MKL = 1
NUMEXPR = 1
NUM_CORES = 1
