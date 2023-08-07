import numpy as np
import mujoco

from constants import *
from BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class SetEnv(BaseArmEnv):
    def __init__(self):
        super().__init__()
        self.has_made_contact = False
        self.set = False

    # grabbing is illegal in the set environment, and this will automatically terminate an episode, 
    # so it's impossible to let go in SetEnv
    def handle_fist(self, close_fist):
        self.check_for_grab(close_fist)
        self.closed_fist = close_fist

        # figure out when the ball has been set by tracking the collision
        if not self.has_made_contact: # pre contact
            self.has_made_contact = self.ball_and_fist_colliding
        elif not self.set: # mid collision
            # the ball has been set once the fist is no longer in contact
            self.set = not self.ball_and_fist_colliding

    # terminate immediately on catch since grabbing is illegal in SetEnv
    # otherwise terminate at the point after the set where the ball is closest to the target
    def should_terminate(self):
        return self.ball_in_hand or (self.set and self.at_perigee)

    def reset_model(self):
        elbow_angle, shoulder_angle, _ = self.arm_random_init()
        self.target_pos = self.target_random_init()
        self.launch_point_pos = self.launch_random_init()
        catch_candidate_pos = self.catch_candidate_pos_random_init()
        init_xdot, init_zdot = self.calc_launch_velocity(self.launch_point_pos, catch_candidate_pos)

        # set and propagate state via data variable (had bugs when I tried modifying model.qpos0 instead)
        self.data.qpos = np.concatenate((np.array([shoulder_angle, elbow_angle]), self.launch_point_pos, IDENTITY_QUAT))
        # no rotation or velocity in the y direction for the ball
        self.data.qvel = np.array([0., 0., init_xdot, 0., init_zdot, 0., 0., 0.])
        mujoco.mj_kinematics(self.model, self.data)

        self.closed_fist = False
        self.ball_in_hand = False
        self.set = False
        self.has_made_contact = False
        self.handle_fist_appearance()
        return self._get_obs()

    # reward is only received at the perigee (i.e. right before the episode terminates)
    def reward(self, close_fist):
        if self.set and self.at_perigee:
            return 1 / self.ball_to_target_distance
        else:
            return 0