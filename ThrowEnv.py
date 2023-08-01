import numpy as np

from BaseArmEnv import BaseArmEnv

ALT_CONTACT_TYPE = 2
assert ALT_CONTACT_TYPE & 1 == 0, "The alternate contact type must not be compatible with the default type: 1"

class ThrowEnv(BaseArmEnv):
    def __init__(self, frame_skip: int = 1, **kwargs):
        super().__init__(frame_skip, **kwargs)

    def reset_model(self):
        shoulder_angle = np.random.rand(0, np.pi)
        elbow_lower_limit = -0.75 * np.pi # 135 degrees on either side for elbow joint
        elbow_upper_limit = 0.75 * np.pi
        # set elbow limits to avoid ground penetration
        # set upper limit if upper arm is to the left and lower limit if upper arm is to the right
        if shoulder_angle > np.pi / 2:
            elbow_upper_limit = min(elbow_upper_limit, 2 * np.pi - 2 * shoulder_angle)
        else:
            elbow_lower_limit = max(elbow_lower_limit, -2 * shoulder_angle)

        elbow_angle = np.random.uniform(elbow_lower_limit, elbow_upper_limit)
        shoulder_pos = np.array([0, 0, 0.02])
        elbow_pos = shoulder_pos + 0.10 * np.array([np.cos(shoulder_angle), 0, np.sin(shoulder_angle)])
        fist_pos = elbow_pos + 0.15 * np.array([np.cos(elbow_angle), 0, np.sin(elbow_angle)])
        ball_pos = fist_pos # ball starts in hand

        target_x = np.random.uniform(0.3, 1)
        target_z = np.random.uniform(0.05, 0.5)
        target_pos = np.array([target_x, 0, target_z])

        self.data.qpos = np.concatenate((np.array([shoulder_angle, elbow_angle]), ball_pos, np.zeros(4)))
        self.model.body('target').pos = target_pos
        self.data.qvel = 0

    def reward(self, changed_fist):
        pass