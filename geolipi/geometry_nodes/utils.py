import numpy as np
from geolipi.symbolic.combinators import Union, Intersection, Difference, JoinUnion
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D
from geolipi.symbolic.primitives_3d import (
    NoParamCuboid3D,
    NoParamCylinder3D,
    NoParamSphere3D,
    PreBakedPrimitive3D,
)

from .geonodes import (
    create_boolean_union_node_seq,
    create_boolean_intersection_node_seq,
    create_boolean_difference_node_seq,
    create_boolean_join_node_seq,
    create_transform_node_seq,
    create_cuboid_node_seq,
    create_sphere_node_seq,
    create_cylinder_node_seq,
    create_prebaked_primitive_node_seq,
)

COMBINATOR_MAP = {
    Union: create_boolean_union_node_seq,
    Intersection: create_boolean_intersection_node_seq,
    Difference: create_boolean_difference_node_seq,
    JoinUnion: create_boolean_join_node_seq,
}
PRIMITIVE_MAP = {
    NoParamCuboid3D: create_cuboid_node_seq,
    NoParamSphere3D: create_sphere_node_seq,
    NoParamCylinder3D: create_cylinder_node_seq,
    PreBakedPrimitive3D: create_prebaked_primitive_node_seq,
}

MODIFIER_MAP = {
    Translate3D: (create_transform_node_seq, "Translation"),
    Scale3D: (create_transform_node_seq, "Scale"),
    EulerRotate3D: (create_transform_node_seq, "Rotation"),
}


BASE_COLORS = np.array(
    [
        [0.0, 0.0, 0.06666667],
        [0.0, 0.01568627, 1.0],
        [0.0, 0.94901961, 0.0],
        [0.0, 1.0, 0.87058824],
        [1.0, 0.0, 0.05882353],
        [1.0, 0.0, 1.0],
        [0.95294118, 1.0, 0.0],
        [0.9372549, 1.0, 1.0],
        [0.0, 0.61176471, 0.20392157],
        [0.0, 0.63921569, 1.0],
        [0.75686275, 0.0, 0.56470588],
        [0.56078431, 1.0, 0.51764706],
        [1.0, 0.62745098, 0.10588235],
        [1.0, 0.61960784, 1.0],
        [0.2627451, 0.0, 0.60392157],
        [0.0, 0.2627451, 0.38039216],
        [0.0, 0.97647059, 0.43529412],
        [0.50980392, 0.0, 0.10588235],
        [0.5254902, 0.0, 1.0],
        [0.25882353, 0.32156863, 0.0],
        [0.20392157, 0.33333333, 0.80784314],
        [0.49803922, 1.0, 0.03529412],
        [0.48235294, 0.96862745, 0.99215686],
        [1.0, 0.29803922, 0.41960784],
        [0.74117647, 0.32156863, 0.0],
        [1.0, 0.89019608, 0.50196078],
        [0.76862745, 0.30588235, 0.8627451],
        [0.50196078, 0.63137255, 0.13333333],
        [0.49803922, 0.29411765, 0.41568627],
        [0.49803922, 0.59215686, 1.0],
        [0.23529412, 0.67058824, 0.60392157],
        [0.70980392, 0.60784314, 0.56862745],
    ]
)
BASE_COLORS = BASE_COLORS[::-1, :]
