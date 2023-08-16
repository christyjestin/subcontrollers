import numpy as np
import os

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Tuple, Box, Discrete

from envs.utils import cosine_similarity
from envs.constants import *


XML_FILE = os.path.realpath("envs\\arm.xml")

DEFAULT_CAMERA_CONFIG = {
  "azimuth": 90.0,
  "distance": 2.0,
  "elevation": -15,
  "lookat": np.array([0.  , 0.  , 0.15]),
}

class BaseArmEnv(MujocoEnv):
    '''
    This super class contains all logic that is common across tasks: each sub class has its own logic for resetting, 
    terminating, and changing the fist as well as a tailored reward function.

    ### This Class
    The primary methods are from the standard `gym.Env` API: `reset()`, `step()`, `render()`, and `close()`. This class
    has a custom implementation of `step()` that is heavily inspired by `MujocoEnv`'s implementation while all other
    methods are inherited from `MujocoEnv`. Each subclass also implements the `reset_model()` method which is used in
    `reset()`. All other logic — transitions, rewards, and terminating — is part of the `step()` function. `BaseArmEnv`
    also contains a number of helper functions that are useful for interacting with the underlying MuJoCo model, 
    computing various quantities, and keeping track of state.

    ### The Robot
    The robot is a 2 link arm with hinge joints in the shoulder and elbow. The arm also has a hand with an abstract
    actuator that switches between the open palm and closed fist configurations. The robot's movement is entirely within
    the xz plane, which also bisects the body of the robot. The state of the robot consists of the two joint angles
    and a discrete binary state for the fist configuration where 1 denotes a closed fist. Similarly the action space
    of the robot is two continuous control inputs and a boolean input for the fist configuration.

    ### The Environment
    The environment always contains the robot and the ball and a ground plane. Depending on the task, it will also
    contain at least one of the target and the launch point. The robot, launch point, and target are all independently
    randomly initialized while the ball's position is chosen based on the task and the position of the other objects
    in the environment. The base of the robot is always at the origin.

    ### The Observation Space
    The observation space consists of a 23 dimensional continuous vector and a single binary variable. The first 9
    indices denote the position of the system while the next 8 denote the velocity. The final 6 values contain the
    stationary 3D coordinates of the launch point and target (in that order). Depending on the task, the launch point
    and target are not always used, but they are always included so that all tasks can share an observation and action
    space.

    The 9 values for the position of the system consist of 2 joint angles, the ball's 3D position, and a quaternion for
    the ball's orientation. Similarly, the 8 velocity values hold the velocity of the 2 joint angles and the 6D velocity
    of the ball. There are no limits for the continuous part of the observation space.

    The lone binary variable denotes the fist configuration: 1 for a closed fist and 0 for an open palm.

    ### The Action Space
    The action space is a pair of continuous inputs and a single discrete binary input. The continuous inputs are
    torques for the shoulder and elbow hinge joints while the binary variable opens and closes the fist: 1 closes the
    fist while 0 opens the fist. Note that the fist does not simply maintain its configuration i.e. a controller must
    still output a 0 or 1 even if the fist is already in the desired shape. The two control inputs must be in the range
    [-1.5, 1.5].

    ### The Costs
    All subclasses share the same costs. There is a quadratic control cost and a fixed cost for changing the fist.

    ### Miscellaneous
    None of the environments truncate episodes. Instead there is logic to check for unviable positions of the ball, and
    the episode will terminate if the ball reaches one of these positions.
    '''

    metadata = {'render_modes': ["human", "rgb_array", "depth_array"], 'render_fps': 25}

    def __init__(self, frame_skip: int = 20, reward_weight: float = 10, render_mode = None):
        MujocoEnv.__init__(self, XML_FILE, frame_skip, observation_space = None, render_mode = render_mode, 
                           default_camera_config = DEFAULT_CAMERA_CONFIG)

        assert NUM_MOTORS == self.model.nu, f"Please update the constant NUM_MOTORS to {self.model.nu}"
        assert PLANE_HALF_SIZE == self.model.geom('plane_geom').size[0], \
            f"Please update the constant PLANE_HALF_SIZE to {self.model.geom('plane_geom').size[0]}"

        # motors on shoulder and elbow joints plus a discrete action for opening and closing fist
        self.action_space = Tuple((Box(-1.5, 1.5, shape = (self.model.nu,)), Discrete(2)))

        # hinge joint angular position and velocity for shoulder and elbow plus free joint position and velocity for ball
        # another 6 dimensions for the fixed 3d positions of the launch point and target
        # and finally a discrete state for the fist being open or closed
        continuous_obs_size = self.model.nv + self.model.nq + 3 + 3
        self.observation_space = Tuple((Box(-np.inf, np.inf, shape = (continuous_obs_size,)), Discrete(2)))

        self.closed_fist = None
        self._ball_in_hand = None

        self._ctrl_cost_weight = 1
        self._change_fist_weight = 1
        self._reward_weight = reward_weight
        # TODO: figure out reward range and spec

        self._launch_point_pos = self.model.body('launch_point').pos
        self._target_pos = self.model.body('target').pos
        self.storage_point = np.array([1.2, 0., 0.])
        self.ball_radius = self.model.geom('ball_geom').size[0]

        self.previous_obs = None

    def _get_obs(self):
        continuous_obs = np.concatenate((self.data.qpos, self.data.qvel, self.launch_point_pos, self.target_pos))
        return (continuous_obs, self.closed_fist)

    def control_cost(self, control, changed_fist):
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(control))
        changed_fist_cost = self._change_fist_weight * changed_fist
        return (ctrl_cost, changed_fist_cost, ctrl_cost + changed_fist_cost)

    def reward(self, close_fist):
        '''Evaluates a reward function that is dependent on the environment'''
        raise NotImplementedError

    def should_terminate(self):
        '''Evaluates a termination condition that is dependent on the environment'''
        raise NotImplementedError

    def stuck(self, obs):
        stuck = self.previous_obs and np.allclose(obs[0], self.previous_obs[0]) and (obs[1] == self.previous_obs[1])
        self.previous_obs = obs
        return bool(stuck)

    def check_for_collision(self, geom1, geom2):
        '''Checks for contact between the two geometries; both inputs should be the names of the geometries'''
        index1 = self.data.geom(geom1).id
        index2 = self.data.geom(geom2).id
        contacts = set(zip(self.data.contact.geom1, self.data.contact.geom2))
        return (index1, index2) in contacts or (index2, index1) in contacts

    @property
    def ball_hit_the_floor(self):
        return self.check_for_collision('plane_geom', 'ball_geom')

    # the bounds are defined as the edges of the world plane (z coordinate is irrelevant for this condition)
    @property
    def out_of_bounds(self):
        xy = self.data.qpos[NUM_MOTORS:NUM_MOTORS + Z_INDEX]
        return bool(np.any(xy > PLANE_HALF_SIZE) or np.any(xy < -PLANE_HALF_SIZE))

    # in both cases, the position is invalid because the arm won't be able to complete the task
    # this is a generic termination condition
    def invalid_position(self):
        return self.ball_hit_the_floor or self.out_of_bounds

    @property
    def at_perigee(self):
        '''
        This function computes whether the ball is at the perigee (the point along its trajectory that is closest to the
        target). It should only be called in certain contexts e.g. only after the ball has been released in the throwing
        task.

        ### Cosine Similarity Explanation
        We're flipping signs at the perigee: before the perigee, the velocity is pointing toward the target, and after
        the perigee, it's pointing away from the target. Thus the vector that points to the target will be perpendicular
        to the velocity at the perigee, and we want the cosine similarity to be 0.

        ### Tolerance Explanation
        The lower the tolerance, the closer we'll push it to 0 before saying we're at the perigee. Note that we check
        for being less than the tolerance rather than being close to 0 because we might slightly overshoot the perigee
        and have a small negative value for cos_sim. If this is the case, then we'll also want to immediately terminate.
        '''
        ball_vel = self.data.qvel[NUM_MOTORS : NUM_MOTORS + 3]
        vec_to_target = self.target_pos - self.ball_pos
        cos_sim = cosine_similarity(ball_vel, vec_to_target)
        return bool(cos_sim < PERIGEE_COSINE_SIMILARITY_TOLERANCE)

    @property # getter and setter to ensure the side effect of turning the weld constraint on and off
    def ball_in_hand(self):
        return self._ball_in_hand

    @ball_in_hand.setter
    def ball_in_hand(self, val):
        assert isinstance(val, bool)
        self._ball_in_hand = val
        self.model.eq_active[0] = self._ball_in_hand

    @property # getter and setter to ensure the side effect of repositioning the target
    def target_pos(self):
        return self._target_pos

    @target_pos.setter
    def target_pos(self, val):
        assert isinstance(val, np.ndarray) and val.shape == (3,)
        self._target_pos = val
        self.model.body('target').pos = self._target_pos

    @property # getter and setter to ensure the side effect of repositioning the launch point
    def launch_point_pos(self):
        return self._launch_point_pos

    @launch_point_pos.setter
    def launch_point_pos(self, val):
        assert isinstance(val, np.ndarray) and val.shape == (3,)
        self._launch_point_pos = val
        self.model.body('launch_point').pos = self._launch_point_pos

    @property
    def ball_pos(self):
        return self.data.geom('ball_geom').xpos

    @property
    def fist_pos(self):
        return self.data.geom('fist_geom').xpos

    @property
    def ball_to_fist_distance(self):
        return np.linalg.norm(self.fist_pos - self.ball_pos)

    @property
    def ball_to_target_distance(self):
        return np.linalg.norm(self.target_pos - self.ball_pos)

    @property
    def ball_within_reach(self):
        # either in contact with hand or within a small distance
        return self.ball_and_fist_colliding or (self.ball_to_fist_distance < BALL_IN_HAND_RADIUS + self.ball_radius)

    @property
    def ball_and_fist_colliding(self):
        return self.check_for_collision('fist_geom', 'ball_geom')

    @property
    def name(self):
        return self.__class__.__name__

    # logic is dependent on the environment
    def handle_fist(self, close_fist):
        raise NotImplementedError

    def check_for_grab(self, close_fist):
        if (not self.closed_fist and close_fist) and self.ball_within_reach: # trying to grab and within reach
            self.ball_in_hand = True # grab i.e. turn on weld constraint

    # handle fist appearance (this function is only updates visuals so it's common to all environments)
    def handle_fist_appearance(self):
        if self.ball_in_hand:
            size, rgba, geom_type = BALL_IN_HAND_RADIUS, TRANSPARENT_PURPLE, SPHERE_GEOM_INDEX
        elif self.closed_fist: # empty, closed fist
            size, rgba, geom_type = CLOSED_FIST_RADIUS, PURPLE, SPHERE_GEOM_INDEX
        else:
            size, rgba, geom_type = OPEN_PALM_SIZE, PURPLE, ELLIPSOID_GEOM_INDEX
        self.model.geom('fist_geom').size = size
        self.model.geom('fist_geom').rgba = rgba
        self.model.geom('fist_geom').type = geom_type

    # randomly choose an initial configuration for the arm (has no side effects i.e. state changes)
    def arm_random_init(self):
        shoulder_angle = np.random.uniform(0, np.pi)
        elbow_lower_limit = -0.75 * np.pi # 135 degrees on either side for elbow joint
        elbow_upper_limit = 0.75 * np.pi
        # set elbow limits to avoid ground penetration (the limits aren't perfect because the elbow to fist distance is
        # longer than the upper arm length, but the simulator seems to prevent ground penetration in this edge case)
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
        return elbow_angle, shoulder_angle, fist_pos

    # randomly choose a position for the target (again has no side effects i.e. state changes)
    def target_random_init(self):
        target_x = np.random.uniform(0.3, 1)
        target_z = np.random.uniform(0.05, 0.5)
        return np.array([target_x, 0, target_z])

    # we initialize the launch point like the target point because the requirements for the two are the same
    def launch_random_init(self):
        return self.target_random_init()

    # choose a candidate position for the arm to make a catch i.e. a position within range for the arm
    # this position will be used to compute the launch velocity to ensure that the pass is catchable
    def catch_candidate_pos_random_init(self):
        r = np.random.uniform(0.12, 0.25)
        th = np.random.uniform(np.pi / 12, np.pi * 11 / 12) # theta is between 15 and 165 degrees
        return r * np.array([np.cos(th), 0., np.sin(th)])

    # compute an initial velocity such that the ball will go from launch_pos to catch_candidate_pos
    def calc_launch_velocity(self, launch_pos, final_pos):
        assert launch_pos[1] == 0. and final_pos[1] == 0.
        delta_x, _, delta_z = final_pos - launch_pos
        assert delta_x < 0, "ball should be moving to the left"

        # first choose the duration of the throw and then back out the launch velocity
        t = np.random.uniform(0.1, 0.8)
        v_avg = delta_z / t
        v0 = v_avg + 0.5 * g * t # v_avg = v0 - 0.5 * g * t -> v0 = v_avg + 0.5 * g * t
        return delta_x / t, v0

    def hide(self, body_name):
        assert body_name in ['target', 'launch_point']
        self.model.body(body_name).pos = self.storage_point
        self.model.geom(f"{body_name}_geom").rgba = INVISIBLE

    def step(self, action):
        bools = [self.ball_in_hand, self.closed_fist]
        assert None not in bools, f"Uninitialized variable: ball_in_hand, closed_fist = {bools}"
        control, close_fist = action
        changed_fist = (self.closed_fist == close_fist)
        self.do_simulation(control, self.frame_skip)

        self.handle_fist(close_fist)
        self.handle_fist_appearance()
        observation = self._get_obs()
        rewards = self._reward_weight * self.reward(close_fist)
        ctrl_cost, changed_fist_cost, total_cost = self.control_cost(control, changed_fist)
        net_reward = rewards - total_cost
        terminated = self.should_terminate() or self.invalid_position() or self.stuck(observation)
        truncated = False
        info = {'rewards': rewards, 'control': ctrl_cost, 'changing_fist': changed_fist_cost, 'total': total_cost}

        if self.render_mode == "human":
            self.render()

        return (observation, net_reward, terminated, truncated, info)

    # step the simulation without applying control
    # this function is meant to help visualize a handful of timesteps after the episode terminates
    def passive_step(self):
        self.step((np.zeros(NUM_MOTORS), self.closed_fist))