import numpy as np
import gym
from gym import spaces

class DigitalTwin5GEnv(gym.Env):
    def __init__(self):
        super(DigitalTwin5GEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(60, 11), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # emBB, URLLC, mMTC
        self.current_step = 0
        self.state = np.random.rand(60, 11)

    def reset(self):
        self.current_step = 0
        self.state = np.random.rand(60, 11)
        return self.state

    def step(self, action):
        reward = np.random.uniform(0, 1)  # Simulate reward (can customize this)
        done = self.current_step >= 200
        self.current_step += 1
        self.state = np.random.rand(60, 11)
        return self.state, reward, done, {}
