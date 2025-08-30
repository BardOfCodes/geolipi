
from .base import GLFunction
from .registry import register_symbol

class Combinator(GLFunction):
    """
    This class is a base for combinator functions in torch_compute.
    Read evaluator specific documentation for more.
    """

@register_symbol
class Union(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_union
    - geometry_nodes.geonodes.create_boolean_union_node_seq
    Read evaluator specific documentation for more.
    """

@register_symbol
class JoinUnion(Union):
    """
    This class is specifically for Blender evaluator. In torch_compute, it is the same as Union.
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_union
    - geometry_nodes.geonodes.create_boolean_join_node_seq
    Read Blender evaluator specific documentation for more.
    """

@register_symbol
class Intersection(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_intersection
    - geometry_nodes.geonodes.create_boolean_intersection_node_seq
    Read evaluator specific documentation for more.
    """

@register_symbol
class Complement(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_complement
    Read evaluator specific documentation for more.
    """

@register_symbol
class Difference(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_difference
    - torch_compute.sdf_operators.create_boolean_difference_node_seq
    Read evaluator specific documentation for more.
    """

@register_symbol
class SwitchedDifference(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_switched_difference
    Read evaluator specific documentation for more.
    """

@register_symbol
class SmoothUnion(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_smooth_union
    Read evaluator specific documentation for more.
    """

@register_symbol
class SmoothIntersection(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_smooth_intersection
    Read evaluator specific documentation for more.
    """

@register_symbol
class SmoothDifference(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_smooth_difference
    Read evaluator specific documentation for more.
    """

@register_symbol
class NarySmoothUnion(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_smooth_union
    Read evaluator specific documentation for more.
    """

@register_symbol
class NarySmoothIntersection(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_smooth_intersection
    Read evaluator specific documentation for more.
    """

@register_symbol
class XOR(Combinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_xor
    Read evaluator specific documentation for more.
    """