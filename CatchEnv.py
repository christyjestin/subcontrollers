import numpy as np
import mujoco

from constants import *
from BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class CatchEnv(BaseArmEnv):
    # because of the large bonus, all successful catches are more or less equal
    # but we still want to use distance in the reward function to make the reward landscape dense
    BALL_IN_HAND_BONUS = 300

    def __init__(self):
        # lower reward weight to disincentivize model from exploiting dense rewards: the model could try
        # to exploit the system by frequently making futile catch attempts that still get small distance rewards
        # hopefully the smaller reward_weight prevents this exploit
        super().__init__(reward_weight = 0.3)
        self.hide('target') # target is irrelevant for the catching task

    # note the episode terminates immediately after a catch so you can't let go in CatchEnv
    def handle_fist(self, close_fist):
        self.check_for_grab(close_fist)
        self.closed_fist = close_fist

    # terminate immediately on catch
    def should_terminate(self):
        return self.ball_in_hand

    def reset_model(self):
        elbow_angle, shoulder_angle, _ = self.arm_random_init()
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
        self.handle_fist_appearance()
        return self._get_obs()

    # reward grasp attempts that are closer to the ball
    def reward(self, close_fist):
        bonus = self.BALL_IN_HAND_BONUS * self.ball_in_hand
        return (bonus + (1 / self.ball_to_fist_distance)) if close_fist else 0