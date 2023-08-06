import numpy as np
import mujoco

from BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class ThrowEnv(BaseArmEnv):
    def __init__(self):
        super().__init__()
        self.hide('launch_point')
        self.released = False

    def handle_fist(self, close_fist):
        if self.released: # no action is possible if you've already let go of the ball
            return
        if not close_fist: # letting go off the ball
            self.ball_in_hand = False
            self.released = True
            self.closed_fist = False

    # TODO: for now, the termination condition will just be invalid_position; this decision depends on reward design
    def should_terminate(self):
        return False

    def reset_model(self):
        elbow_angle, shoulder_angle, fist_pos = self.arm_random_init()
        ball_pos = fist_pos # ball starts in hand
        self.target_pos = self.target_random_init()
        # set and propagate state via data variable (had bugs when I tried modifying model.qpos0 instead)
        self.data.qpos = np.concatenate((np.array([shoulder_angle, elbow_angle]), ball_pos, IDENTITY_QUAT))
        self.data.qvel = 0
        mujoco.mj_kinematics(self.model, self.data)

        self.closed_fist = True
        self.ball_in_hand = True
        self.released = False
        self.handle_fist_appearance()
        return self._get_obs()

    def reward(self, changed_fist):
        return 0