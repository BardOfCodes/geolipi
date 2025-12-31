"""
Mathematical and numerical constants used throughout the torch_compute module.

This module centralizes all constants to ensure consistency and make them
easily configurable if needed.
"""

import numpy as np

# Numerical precision constants
EPSILON = 1e-9
ACOS_EPSILON = 1e-7

# Basic mathematical constants
SQRT_2 = np.sqrt(2, dtype=np.float32)
SQRT_3 = np.sqrt(3, dtype=np.float32)

# Trigonometric constants
COS_30 = np.cos(np.pi / 6)
TAN_30 = np.tan(np.pi / 6)

# Geometric constants for 2D shapes
PENT_VEC = np.array([0.809016994, 0.587785252, 0.726542528], dtype=np.float32)
HEX_VEC = np.array([-0.866025404, 0.5, 0.577350269], dtype=np.float32)
OCT_VEC = np.array([-0.9238795325, 0.3826834323, 0.4142135623], dtype=np.float32)
HEXAGRAM_VEC = np.array(
    [-0.5, 0.8660254038, 0.5773502692, 1.7320508076], dtype=np.float32
)
STAR_VEC = np.array([0.809016994375, -0.587785252292], dtype=np.float32)
# v1 = ( k1x, -k1y), v2 = (-k1x, -k1y), v3 = ( k2x, -k2y)


PENTAGRAM_V1 = np.array([ 0.809016994, -0.587785252], dtype=np.float32).reshape(1, 1, 2)
PENTAGRAM_V2 = np.array([-0.809016994, -0.587785252], dtype=np.float32).reshape(1, 1, 2)
PENTAGRAM_V3 = np.array([ 0.309016994, -0.951056516], dtype=np.float32).reshape(1, 1, 2)
PENTAGRAM_K1Z = 0.726542528  # tan(pi/5)
