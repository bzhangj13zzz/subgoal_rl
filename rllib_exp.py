# Import the RL algorithm (Trainer) we would like to use.
from collections import deque
from ray.rllib.agents.ppo import PPOTrainer
from gym_minigrid.envs.empty import EmptyEnv5x5
from gym_minigrid.envs.fourrooms import FourRoomsEnv
from ray.tune import register_env
import gym
import gym_minigrid
import numpy as np
from ray.rllib.utils.numpy import one_hot

class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim, )))

        self.observation_space = gym.spaces.Box(
            0.0,
            1.0,
            shape=(self.single_frame_dim * self.framestack, ),
            dtype=np.float32)

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim, )))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt((np.array(self.x_positions) - self.init_x)**2 +
                                (np.array(self.y_positions) - self.init_y)**2))
                    self.x_y_delta_buffer.append(max_diff)
                    print("100-average dist travelled={}".format(
                        np.mean(self.x_y_delta_buffer)))
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1, ))
        direction = one_hot(
            np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)

def env_maker(env_config):
    return FourRoomsEnv()
    # name = config.get("name", "MiniGrid-Empty-5x5-v0")
    # env = gym.make(name)
    # # Only use image portion of observation (discard goal and direction).
    # env = gym_minigrid.wrappers.ImgObsWrapper(env)
    # env = OneHotWrapper(
    #     env,
    #     config.vector_index if hasattr(config, "vector_index") else 0,
    #     framestack=framestack)
    # return env


register_env("mini-grid", env_maker)

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "mini-grid",
    "env_config": {},
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
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
        "render_env": True,
    }
}

trainer = PPOTrainer(env="mini-grid")

# Create our RLlib Trainer.
# trainer = PPOTrainer(env=EmptyEnv5x5, config={ 
#     "env_config": {}, "framework": "torch"  # config to pass to env class
# })

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    print(trainer.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()