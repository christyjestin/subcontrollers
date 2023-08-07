import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Tuple, Box, Discrete

from utils import cosine_similarity
from constants import *

XML_FILE = "./arm.xml"

DEFAULT_CAMERA_CONFIG = {
  "azimuth": 90.0,
  "distance": 2.0,
  "elevation": -15,
  "lookat": np.array([0.  , 0.  , 0.15]),
}

class BaseArmEnv(MujocoEnv):
    metadata = {'render_modes': ["human", "rgb_array", "depth_array"], 'render_fps': 25}

    def __init__(self, frame_skip: int = 20):
        MujocoEnv.__init__(self, XML_FILE, frame_skip, observation_space = None, render_mode = "human", 
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
        # TODO: update weights
        self._ctrl_cost_weight = 1
        self._change_fist_weight = 1
        # TODO: figure out reward range and spec

        self._launch_point_pos = self.model.body('launch_point').pos
        self._target_pos = self.model.body('target').pos
        self.storage_point = np.array([1.2, 0., 0.])
        self.ball_radius = self.model.geom('ball_geom').size[0]

    def _get_obs(self):
        continuous_obs = np.concatenate((self.data.qpos, self.data.qvel, self.launch_point_pos, self.target_pos))
        return (continuous_obs, self.closed_fist)

    def control_cost(self, control, changed_fist):
        return self._ctrl_cost_weight * np.sum(np.square(control)) + self._change_fist_weight * changed_fist

    # reward function is dependent on the environment
    def reward(self, close_fist):
        raise NotImplementedError

    # termination condition is also dependent on the environment
    def should_terminate(self):
        raise NotImplementedError

    # checks whether the z acceleration is 0 since this means the ball is simply rolling on the floor
    def on_the_floor(self):
        return np.isclose(self.data.qacc[NUM_MOTORS + Z_INDEX], 0, atol=1e-4)

    # the bounds are defined as the edges of the world plane (z coordinate is irrelevant for this condition)
    def out_of_bounds(self):
        xy = self.data.qpos[NUM_MOTORS:NUM_MOTORS + Z_INDEX]
        return np.any(xy > PLANE_HALF_SIZE) or np.any(xy < -PLANE_HALF_SIZE)

    # in both cases, the position is invalid because the arm won't be able to complete the task
    # this is a generic termination condition
    def invalid_position(self):
        return self.on_the_floor() or self.out_of_bounds()

    # computes whether the ball is at the perigee (the point along its trajectory that is closest to the target)
    # should only be called in certain contexts e.g. only after the ball has been released in the throwing task
    @property
    def at_perigee(self):
        # Cosine Similarity Explanation
        # We're flipping signs at the perigee: before the perigee, the velocity is pointing toward the target and after the
        # perigee, it's pointing away from the target; thus the vector that points to the target will be perpendicular
        # to the velocity at the perigee, and we want the cosine similarity to be 0

        # Tolerance Explanation
        # The lower the tolerance, the closer we'll push it to 0 before saying we're at the perigee; note that we check
        # for being less than the tolerance rather than being close to 0 because we might slightly overshoot the perigee
        # and have a small negative value for cos_sim: if this is the case, then we'll also want to immediately terminate
        ball_vel = self.data.qvel[NUM_MOTORS : NUM_MOTORS + 3]
        vec_to_target = self.target_pos - self.ball_pos
        cos_sim = cosine_similarity(ball_vel, vec_to_target)
        return cos_sim < PERIGEE_COSINE_SIMILARITY_TOLERANCE

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
        fist_index = self.data.geom('fist_geom').id
        ball_index = self.data.geom('ball_geom').id
        contacts = set(zip(self.data.contact.geom1, self.data.contact.geom2))
        return (fist_index, ball_index) in contacts or (ball_index, fist_index) in contacts

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
            size, rgba, geom_type = OPEN_FIST_SIZE, PURPLE, ELLIPSOID_GEOM_INDEX
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
        rewards = self.reward(close_fist)
        costs = self.control_cost(control, changed_fist)
        net_reward = rewards - costs
        terminated = self.should_terminate() or self.invalid_position()
        truncated = False
        info = {'rewards': rewards, 'costs': costs}

        observation = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return (observation, net_reward, terminated, truncated, info)

    # step the simulation without applying control
    # this function is meant to help visualize a handful of timesteps after the episode terminates
    def passive_step(self):
        self.step((np.zeros(NUM_MOTORS), self.closed_fist))