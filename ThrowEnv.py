import numpy as np
import mujoco

from BaseArmEnv import BaseArmEnv

ALT_CONTACT_TYPE = 2
assert ALT_CONTACT_TYPE & 1 == 0, "The alternate contact type must not be compatible with the default type: 1"

IDENTITY_QUAT = np.array([1., 0., 0., 0.])

class ThrowEnv(BaseArmEnv):
    def __init__(self):
        super().__init__()

    def reset_model(self):
        shoulder_angle = np.random.uniform(0, np.pi)
        elbow_lower_limit = -0.75 * np.pi # 135 degrees on either side for elbow joint
        elbow_upper_limit = 0.75 * np.pi
        # set elbow limits to avoid ground penetration
        # set upper limit if upper arm is to the left and lower limit if upper arm is to the right
        if shoulder_angle > np.pi / 2:
            elbow_upper_limit = min(elbow_upper_limit, 2 * np.pi - 2 * shoulder_angle)
        else:
            elbow_lower_limit = max(elbow_lower_limit, -2 * shoulder_angle)

        # randomly initialize arm configuration
        elbow_angle = np.random.uniform(elbow_lower_limit, elbow_upper_limit)
        total_angle = shoulder_angle + elbow_angle
        shoulder_pos = np.array([0, 0, 0.02])
        elbow_pos = shoulder_pos + 0.10 * np.array([np.cos(shoulder_angle), 0, np.sin(shoulder_angle)])
        # note that upper arm and forearm are same length, but the fist is positioned
        # at 0.15 i.e. further than the end of the forearm
        fist_pos = elbow_pos + 0.15 * np.array([np.cos(total_angle), 0, np.sin(total_angle)])
        ball_pos = fist_pos # ball starts in hand

        # randomly initialize target
        target_x = np.random.uniform(0.3, 1)
        target_z = np.random.uniform(0.05, 0.5)
        target_pos = np.array([target_x, 0, target_z])
        # directly change the model for the fixed body
        self.model.body('target').pos = target_pos

        # set and propagate state via data variable (had bugs when I tried modifying model.qpos0 instead)
        self.data.qpos = np.concatenate((np.array([shoulder_angle, elbow_angle]), ball_pos, IDENTITY_QUAT))
        self.data.qvel = 0
        mujoco.mj_kinematics(self.model, self.data)

        self.handle_fist(close_fist = True)
        self.terminated = False
        return self._get_obs()

    def reward(self, changed_fist):
        return 0