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
