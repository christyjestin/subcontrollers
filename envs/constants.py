import numpy as np

PURPLE = np.array([1, 0, 1, 1])
TRANSPARENT_PURPLE = np.array([1, 0, 1, 0.25])
INVISIBLE = np.zeros(4)
ALPHA_CHANNEL = 3
TRANSPARENT_ALPHA = 0.25
SEMI_TRANSPARENT_ALPHA = 0.5

CLOSED_FIST_RADIUS = 0.02 # empty closed fist
BALL_IN_HAND_RADIUS = 0.03 # closed fist with ball in hand
OPEN_PALM_SIZE = np.array([0.75, 2, 2]) * CLOSED_FIST_RADIUS # ellipsoid shape for open palm

BALL_IN_HAND_CONSTRAINT = 0
BALL_IN_TARGET_CONSTRAINT = 1

SPHERE_GEOM_INDEX = 2
ELLIPSOID_GEOM_INDEX = 4

IDENTITY_QUAT = np.array([1., 0., 0., 0.])

g = 9.81

NUM_MOTORS = 2
Z_INDEX = 2

PLANE_HALF_SIZE = 1.5

PERIGEE_COSINE_SIMILARITY_TOLERANCE = 1e-6 # see at_perigee function in BaseArmEnv for explanation