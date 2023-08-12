import numpy as np
import mujoco

from envs.constants import *
from envs.BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class CatchEnv(BaseArmEnv):
    '''
    This environment implements the catching task. It includes the launch point but not the target. The ball starts at
    the launch point and its initial velocity is chosen such that the resulting pass is within reach for the arm.

    The robot is rewarded for grasp attempts where the fist is close to the ball. It has an additional large bonus for
    successful grasp attempts i.e. ones that are close enough to result in a catch. This scheme produces a more dense
    reward but requires some tweaks to reduce exploits (see `BALL_IN_HAND_BONUS` and `reward_weight` for details). The
    reward function is of the form `bonus + 1 / dist` where the bonus is a fixed constant if the catch is successful
    and 0 otherwise.

    The task terminates when the ball is caught or when it becomes unviable i.e. hits the ground or goes out of bounds.
    '''

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
        self.previous_obs = None
        return self._get_obs()

    # reward grasp attempts that are closer to the ball
    def reward(self, close_fist):
        bonus = self.BALL_IN_HAND_BONUS * self.ball_in_hand
        return (bonus + (1 / self.ball_to_fist_distance)) if close_fist else 0