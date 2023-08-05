import numpy as np
import mujoco

from BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class ThrowEnv(BaseArmEnv):
    def __init__(self):
        super().__init__()

    def reset_model(self):
        elbow_angle, shoulder_angle, fist_pos = self.arm_random_init()
        ball_pos = fist_pos # ball starts in hand

        # directly change the model for the fixed body
        self.model.body('target').pos = self.target_random_init()
        # set and propagate state via data variable (had bugs when I tried modifying model.qpos0 instead)
        self.data.qpos = np.concatenate((np.array([shoulder_angle, elbow_angle]), ball_pos, IDENTITY_QUAT))
        self.data.qvel = 0
        mujoco.mj_kinematics(self.model, self.data)

        self.handle_fist(close_fist = True)
        self.terminated = False
        return self._get_obs()

    def reward(self, changed_fist):
        return 0