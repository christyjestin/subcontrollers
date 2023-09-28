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

    def __init__(self, render_mode = None):
        super().__init__(render_mode = render_mode)
        self.hide('launch_point')
        self.released = False
        self.just_released = False

    def handle_fist(self, close_fist):
        if self.released: # no action is possible if you've already let go of the ball
            return
        if not close_fist: # letting go off the ball
            self.ball_in_hand = False
            self.released = True
            self.just_released = True
            self.closed_fist = False

    # terminate at the point after the release where the ball is closest to the target
    def should_terminate(self):
        return (self.released and self.at_perigee) or self.ball_in_target

    # custom sample function that makes the robot much more likely to hold onto the ball
    def sample_random_action(self):
        continuous, _ = self.action_space.sample()
        discrete = bool(np.random.rand() < 0.8) # close fist with prob 0.8
        return (continuous, discrete)

    def run_custom_step_logic(self):
        self.check_ball_in_target()

    def reset_model(self):
        self.reset_ball_in_target()
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
        self.just_released = False
        self.t = 0
        self.handle_fist_appearance()
        self.previous_obs = None
        return self._get_obs()

    # wrapper function adds a small penalty to every timestep to discourage longer episodes
    # hopefully this encourages more intuitive throws
    def reward(self):
        return self.true_reward() - 20 * np.sqrt(self.t)

    def true_reward(self):
        # COMPONENT 0: max reward override to avoid issues with edge case where ball gets inside the target before it reaches perigee
        if self.ball_in_target:
            return 20000
        # COMPONENT 1: primary reward that is based on the closest the ball gets to the target i.e. how far it is at the perigee
        # we use a piecewise function that rewards throws that are close but is also very harsh on throws that are way off
        if self.released and (self.at_perigee or self.invalid_position()):
            threshold = 0.15
            # more intuitive scale where the ball is closer than the threshold iff the distance is less than 1
            scaled_distance = self.ball_to_target_distance / threshold
            inverse_scaled_distance = 1 / scaled_distance
            # the functional form is x^(a-bx): we use a (relatively) high initial power for a and
            # a small factor for b to reduce this power over time and slow down reward growth
            # x is the inverse scaled distance for rewards and scaled distance for punishments
            a, b = 8, 0.65
            c, d = 5, 0.1 # punishment has higher peak values of x, so we adjust the parameters to fit the range
            if scaled_distance < 1: # reward
                val = 10 * inverse_scaled_distance ** (a - b * inverse_scaled_distance)
            else: # punishment
                val = -10 * scaled_distance ** (c - d * scaled_distance)
            return np.clip(val, -10000, 20000) # clip to avoid precision or backprop issues
        # COMPONENT 2: smaller, auxiliary reward for throws that "have the right idea" even if they're way off target
        if self.just_released:
            self.just_released = False
            # max velocity is chosen based on the velocity of a quick pass (t = 0.3 seconds) to a hypothetical
            # target at the rightmost edge of the plane at the same level as the release point
            # t = 2v_z / g -> v_z = 0.5gt; t = d / v_x -> v_x = d / t
            t = 0.3
            MAX_VEL = np.linalg.norm(np.array([0.5 * 10 * t, 0, 1. / t]))
            # only reward throws that point up and to the right (this is typically the right direction)
            up_and_right = (self.ball_vel[0] > 0) and (self.ball_vel[2] > 0)
            # cap velocity reward because there's diminishing returns, and this is an auxiliary reward
            # to help the arm find the true, primary reward (which is based on distance to target)
            return 100 * np.clip(np.linalg.norm(self.ball_vel), 0, MAX_VEL) if up_and_right else 0
        # COMPONENT 3: penalty to prevent the robot from ending the episode by just hitting the floor or holding onto the ball
        if (self.invalid_position() and not self.released) or self.t == self.MAX_EPISODE_LENGTH:
            return -15000
        return 0 # default is no reward