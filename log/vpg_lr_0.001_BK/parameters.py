# ======= Main
SEED = 1
EPISODES = 100000
HORIZON = 100
LOAD_MODEL = False
BATCH_SIZE = 400
SAVE_MODEL = 100
# AGENT_NAME = "random"
AGENT_NAME = "vpg"
SHOULD_LOG = 1

# ===== Enviroment
ENV_NAME = "tworoom"
IMAGE_OBS = False
RENDER = False
GRID = 11
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
VERBOSE = False

# ========== System Environment Variables
ENV_VAR = True
OPENBLAS = 1
OMP = 1
MKL = 1
NUMEXPR = 1
NUM_CORES = 1
