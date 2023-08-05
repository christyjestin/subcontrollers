import numpy as np
import mujoco

from BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class SetEnv(BaseArmEnv):
    def __init__(self):
        super().__init__()

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

        self.handle_fist(close_fist = False)
        self.ball_in_hand = False
        self.terminated = False
        return self._get_obs()

    def reward(self, changed_fist):
        return 0