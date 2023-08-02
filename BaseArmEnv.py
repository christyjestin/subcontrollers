import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Tuple, Box, Discrete

XML_FILE = "./arm.xml"

PURPLE = [1, 0, 1]
BLUE = [0, 0, 1]

def rgba(color, alpha = 1):
    assert len(color) == 3 and all(0 <= i <= 1 for i in color), f"Invalid color: {color}"
    assert 0 <= alpha <= 1, f"Invalid alpha: {alpha}"
    return " ".join([str(i) for i in color]) + " " + str(alpha)

class BaseArmEnv(MujocoEnv):
    metadata = {'render_modes': ["human", "rgb_array", "depth_array"], 'render_fps': 25}

    def __init__(self, frame_skip: int = 20):
        MujocoEnv.__init__(self, XML_FILE, frame_skip, observation_space = None, camera_id = 0, render_mode = "human")

        # motors on shoulder and elbow joints plus a discrete action for opening and closing fist
        self.action_space = Tuple((Box(-np.inf, np.inf, shape = (self.model.nu,)), Discrete(2)))

        # hinge joint angular position and velocity for shoulder and elbow plus free joint position and velocity for ball
        # and finally a discrete state for the fist being open or closed
        obs_size = self.model.nv + self.model.nq
        self.observation_space = Tuple((Box(-np.inf, np.inf, shape = (obs_size,)), Discrete(2)))

        self.closed_fist = True
        self.terminated = False
        # TODO: update weights
        self._ctrl_cost_weight = 1
        self._change_fist_weight = 1
        # TODO: set camera attributes self.camera_id, self.camera_name
        # TODO: figure out reward range and spec
        self.render_mode = 'human'

    def _get_obs(self):
        return (np.concatenate((self.data.qpos, self.data.qvel)), self.closed_fist)

    def reset_model(self):
        raise NotImplementedError # random initialization depends on the task

    def reset(self, seed = None):
        super().reset(seed = seed)
        observation = self.reset_model()
        self._reset_simulation()
        info = {}

        if self.render_mode == "human":
            self.render()
        return observation, info

    def control_cost(self, control, changed_fist):
        return self._ctrl_cost_weight * np.sum(np.square(control)) + self._change_fist_weight * changed_fist

    def reward(self, changed_fist):
        # reward function depends on the task; this function must also set self.terminated
        raise NotImplementedError

    # returns a boolean indicating whether the fist has changed
    def handle_fist(self, close_fist):
        if self.closed_fist == close_fist: # do nothing if fist already matches desired state
            return False
        self.closed_fist = close_fist
        self.model.geom('fist_geom').rgba = rgba(PURPLE) if self.closed_fist else rgba(BLUE)
        self.model.geom('fist_geom').size = "0.02" if self.closed_fist else "0.04"
        return True

    def step(self, action):
        control, fist_action = action
        changed_fist = self.handle_fist(close_fist = bool(fist_action))
        self.do_simulation(control, self.frame_skip)

        rewards = self.reward(changed_fist) 
        costs = self.control_cost(control, changed_fist)
        net_reward = rewards - costs
        truncated = False
        info = {'rewards': rewards, 'costs': costs}

        observation = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return (observation, net_reward, self.terminated, truncated, info)