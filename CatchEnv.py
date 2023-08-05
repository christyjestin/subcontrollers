import numpy as np
import mujoco

from BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

g = 9.81

class CatchEnv(BaseArmEnv):
    def __init__(self):
        super().__init__()

    # choose a candidate position for the arm to make a catch i.e. a position within range for the arm
    # this position will be used to compute the launch velocity to ensure that the ball is catchable
    def catch_candidate_pos_random_init(self):
        r = np.random.uniform(0.12, 0.25)
        th = np.random.uniform(np.pi / 12, np.pi * 11 / 12) # theta is between 15 and 165 degrees
        return r * np.array([np.cos(th), 0., np.sin(th)])

    # compute an initial velocity such that the ball will go from target_pos to catch_candidate_pos
    def calc_launch_velocity(self, target_pos, catch_candidate_pos):
        assert target_pos[1] == 0. and catch_candidate_pos[1] == 0.
        delta_x, _, delta_z = catch_candidate_pos - target_pos
        assert delta_x < 0, "ball should be moving to the left"

        # first choose the duration of the throw and then back out the launch velocity
        t = np.random.uniform(0.1, 0.8)
        v_avg = delta_z / t
        v0 = v_avg + 0.5 * g * t # v_avg = v0 - 0.5 * g * t -> v0 = v_avg + 0.5 * g * t
        return delta_x / t, v0

    def reset_model(self):
        elbow_angle, shoulder_angle, _ = self.arm_random_init()
        target_pos = self.target_random_init()
        catch_candidate_pos = self.catch_candidate_pos_random_init()
        init_xdot, init_zdot = self.calc_launch_velocity(target_pos, catch_candidate_pos)

        # directly change the model for the fixed body
        self.model.body('target').pos = target_pos
        # set and propagate state via data variable (had bugs when I tried modifying model.qpos0 instead)
        self.data.qpos = np.concatenate((np.array([shoulder_angle, elbow_angle]), target_pos, IDENTITY_QUAT))
        # no rotation or velocity in the y direction for the ball
        self.data.qvel = np.array([0., 0., init_xdot, 0., init_zdot, 0., 0., 0.])
        mujoco.mj_kinematics(self.model, self.data)

        self.handle_fist(close_fist = False)
        self.ball_in_hand = False
        self.terminated = False
        return self._get_obs()

    def reward(self, changed_fist):
        return 0