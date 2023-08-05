import numpy as np

PURPLE = np.array([1, 0, 1, 1])
HALF_TRANSPARENT_PURPLE = np.array([1, 0, 1, 0.4])
TRANSPARENT_PURPLE = np.array([1, 0, 1, 0.25])
INVISIBLE = np.zeros(4)

CLOSED_FIST_RADIUS = 0.02 # empty closed fist
BALL_IN_HAND_RADIUS = 0.03 # closed fist with ball in hand
OPEN_FIST_RADIUS = 0.04

IDENTITY_QUAT = np.array([1., 0., 0., 0.])

g = 9.81