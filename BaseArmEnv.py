import numpy as np

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Tuple, Box, Discrete

XML_FILE = "./arm.xml"

PURPLE = np.array([1, 0, 1, 1])
HALF_TRANSPARENT_PURPLE = np.array([1, 0, 1, 0.4])
TRANSPARENT_PURPLE = np.array([1, 0, 1, 0.25])

CLOSED_FIST_RADIUS = 0.02 # empty closed fist
BALL_IN_HAND_RADIUS = 0.03 # closed fist with ball in hand
OPEN_FIST_RADIUS = 0.04

DEFAULT_CAMERA_CONFIG = {
  "azimuth": 90.0,
  "distance": 2.0,
  "elevation": -15,
  "lookat": np.array([0.  , 0.  , 0.15]),
}

class BaseArmEnv(MujocoEnv):
    metadata = {'render_modes': ["human", "rgb_array", "depth_array"], 'render_fps': 25}

    def __init__(self, frame_skip: int = 20):
        MujocoEnv.__init__(self, XML_FILE, frame_skip, observation_space = None, render_mode = "human", 
                           default_camera_config = DEFAULT_CAMERA_CONFIG)

        # motors on shoulder and elbow joints plus a discrete action for opening and closing fist
        self.action_space = Tuple((Box(-1.5, 1.5, shape = (self.model.nu,)), Discrete(2)))

        # hinge joint angular position and velocity for shoulder and elbow plus free joint position and velocity for ball
        # and finally a discrete state for the fist being open or closed
        obs_size = self.model.nv + self.model.nq
        self.observation_space = Tuple((Box(-np.inf, np.inf, shape = (obs_size,)), Discrete(2)))

        self.closed_fist = None
        self._ball_in_hand = None
        self.terminated = None
        # TODO: update weights
        self._ctrl_cost_weight = 1
        self._change_fist_weight = 1
        # TODO: figure out reward range and spec

    def _get_obs(self):
        return (np.concatenate((self.data.qpos, self.data.qvel)), self.closed_fist)

    def control_cost(self, control, changed_fist):
        return self._ctrl_cost_weight * np.sum(np.square(control)) + self._change_fist_weight * changed_fist

    def reward(self, changed_fist):
        # reward function depends on the task; this function must also set self.terminated
        raise NotImplementedError

    @property # getter and setter to ensure the side effect of turning the weld constraint on and off
    def ball_in_hand(self):
        return self._ball_in_hand

    @ball_in_hand.setter
    def ball_in_hand(self, val):
        assert isinstance(val, bool)
        self._ball_in_hand = val
        self.model.eq_active[0] = self._ball_in_hand

    @property
    def ball_within_reach(self):
        fist_pos = self.data.geom('fist_geom').xpos
        ball_pos = self.data.geom('ball_geom').xpos
        dist = np.linalg.norm(fist_pos - ball_pos)
        sum_of_radii = self.model.geom('fist_geom').size[0] + self.model.geom('ball_geom').size[0]
        # d + o = r1 + r2 -> o = (r1 + r2) - d
        overlap = sum_of_radii - dist # signed overlap i.e. negative values indicate no collision
        return overlap > 0.025 # minimum overlap for a catch is 0.025

    # returns a boolean indicating whether the fist has changed
    def handle_fist(self, close_fist):
        if self.closed_fist == close_fist: # do nothing if fist already matches desired state
            return False

        # handle grasping or letting go
        if close_fist and self.ball_within_reach:
            self.ball_in_hand = True # grabbing the ball
        elif (not close_fist) and self.ball_in_hand:
            self.ball_in_hand = False # letting go off the ball

        # handle fist appearance
        self.closed_fist = close_fist
        if self.ball_in_hand:
            self.model.geom('fist_geom').size = BALL_IN_HAND_RADIUS
            self.model.geom('fist_geom').rgba = TRANSPARENT_PURPLE
        else:
            self.model.geom('fist_geom').size = CLOSED_FIST_RADIUS if self.closed_fist else OPEN_FIST_RADIUS
            self.model.geom('fist_geom').rgba = PURPLE if self.closed_fist else HALF_TRANSPARENT_PURPLE
        return True

    def step(self, action):
        bools = [self.ball_in_hand, self.closed_fist, self.terminated]
        assert None not in bools, f"Uninitialized variable: ball_in_hand, closed_fist, terminated = {bools}"
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