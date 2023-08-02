import time
from ThrowEnv import ThrowEnv


env = ThrowEnv()
for _ in range(10):
    env.reset()
    time.sleep(1)
env.close()