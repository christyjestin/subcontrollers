import numpy as np
import mujoco

from envs.constants import *
from envs.BaseArmEnv import BaseArmEnv, IDENTITY_QUAT

class ThrowEnv(BaseArmEnv):
    '''
    This environment implements the throwing task. It includes the target but not the launch point. The ball starts in
    the robot's hand.

    Like in the set task, the robot is rewarded for throw attempts where the ball's trajectory goes near the target. See
    `SetEnv` for an explanation about how this calculation works.

    The task terminates when the ball reaches the perigee or becomes unviable i.e. hits the ground or goes out of bounds.
    Note that after the ball is released, a control input that tries to change the fist will have no effect on the
    robot's fist configuration. This is to prevent the robot from accidentally regrabbing the ball. Letting go of the
    ball without regrabbing is likely a very difficult behavior to produce with random exploration. If the robot is 
    choosing actions randomly, then it has probability `0.5 ** n` of not regrabbing where n is the number of timesteps
    that it must refrain from grabbing. The reasoning behind this design choice is that the robot should learn to not
    regrab on its own because this action will have no effect on the state despite still having a control cost.
    '''

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

    # terminate at the point after the release where the ball is closest to the target
    def should_terminate(self):
        return self.released and self.at_perigee

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

    # reward is only received at the perigee (i.e. right before the episode terminates)
    def reward(self, close_fist):
        return (1 / self.ball_to_target_distance) if (self.released and self.at_perigee) else 0