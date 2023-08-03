from ThrowEnv import ThrowEnv
import numpy as np
import time


env = ThrowEnv()
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
env.close()