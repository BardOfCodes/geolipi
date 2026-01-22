from .base import GLExpr, GLFunction
from sympy import Tuple as SympyTuple
import sympy as sp
from .registry import register_symbol

class Primitive3D(GLFunction):
    """Functions for declaring 3D primitives."""

    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = []
        for arg in args:
            if isinstance(arg, sp.Symbol):
                replaced_args.append(self.lookup_table.get(arg, arg))
            else:
                replaced_args.append(arg)
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.pretty_print(tabs=tabs + 1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    item = [f"{x:.3f}" for x in arg]
                    item = ", ".join(item)
                    str_args.append(f"({item})")
                else:
                    str_args.append(str(arg))
        if str_args:
            n_tabs_1 = tab_str * (tabs + 1)
            # str_args = [""] + str_args
            str_args = f", ".join(str_args)
            final = f"{self.func.__name__}({str_args})"
        else:
            final = f"{self.func.__name__}()"
        return final

@register_symbol
class Sphere3D(Primitive3D):
    @classmethod
    def default_spec(cls):
        return {"radius": {"type": "float"}}


@register_symbol
class Box3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"size": {"type": "Vector[3]"}}


@register_symbol
class Cuboid3D(Box3D):

    @classmethod
    def default_spec(cls):
        return {"size": {"type": "Vector[3]"}}


@register_symbol
class RoundedBox3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"size": {"type": "Vector[3]"}, "radius": {"type": "float"}}


@register_symbol
class BoxFrame3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"b": {"type": "Vector[3]"}, "e": {"type": "float"}}


@register_symbol
class Torus3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"t": {"type": "Vector[2]"}}


@register_symbol
class CappedTorus3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"angle": {"type": "float"}, "ra": {"type": "float"}, "rb": {"type": "float"}}


@register_symbol
class Link3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"le": {"type": "float"}, "r1": {"type": "float"}, "r2": {"type": "float"}}


@register_symbol
class InfiniteCylinder3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"c": {"type": "Vector[3]"}}


@register_symbol
class Cone3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"angle": {"type": "float"}, "h": {"type": "float", "min": 0.0}}


@register_symbol
class InexactCone3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"angle": {"type": "float"}, "h": {"type": "float", "min": 0.0}}


@register_symbol
class InfiniteCone3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"angle": {"type": "float"}}


@register_symbol
class Plane3D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"n": {"type": "Vector[3]"}, "h": {"type": "float"}}

@register_symbol
class PlaneV23D(Primitive3D):

    @classmethod
    def default_spec(cls):
        return {"origin": {"type": "Vector[3]"}, "normal": {"type": "Vector[3]"}}


@register_symbol
class HexPrism3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_hex_prism
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "Vector[2]"}}


@register_symbol
class TriPrism3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_tri_prism
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "Vector[2]"}}


@register_symbol
class Capsule3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capsule
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "r": {"type": "float", "min": 0.0}}


@register_symbol
class VerticalCapsule3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_vertical_capsule
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "float", "min": 0.0}, "r": {"type": "float", "min": 0.0}}


@register_symbol
class VerticalCappedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_cylinder
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "float", "min": 0.0}, "r": {"type": "float", "min": 0.0}}


@register_symbol
class CappedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_cylinder
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "float", "min": 0.0}, "r": {"type": "float", "min": 0.0}}


@register_symbol
class Cylinder3D(CappedCylinder3D):

    """
    This class is mapped to the following evaluator function(s):
    # TODO: No Mapping yet.
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "float", "min": 0.0}, "r": {"type": "float", "min": 0.0}}


@register_symbol
class ArbitraryCappedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_arbitrary_capped_cylinder
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "r": {"type": "float", "min": 0.0}}


@register_symbol
class RoundedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_rounded_cylinder
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"ra": {"type": "float", "min": 0.0}, "rb": {"type": "float", "min": 0.0}, "h": {"type": "float", "min": 0.0}}


@register_symbol
class CappedCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_cone
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"r1": {"type": "float", "min": 0.0}, "r2": {"type": "float", "min": 0.0}, "h": {"type": "float", "min": 0.0}}


@register_symbol
class ArbitraryCappedCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_arbitrary_capped_cone
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "ra": {"type": "float", "min": 0.0}, "rb": {"type": "float", "min": 0.0}}


@register_symbol
class SolidAngle3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_solid_angle
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"angle": {"type": "float"}, "ra": {"type": "float", "min": 0.0}}


@register_symbol
class CutSphere3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_cut_sphere
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"r": {"type": "float", "min": 0.0}, "h": {"type": "float"}}


@register_symbol
class CutHollowSphere(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_cut_hollow_sphere
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"r": {"type": "float", "min": 0.0}, "h": {"type": "float"}, "t": {"type": "float", "min": 0.0}}


@register_symbol
class DeathStar3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_death_star
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"ra": {"type": "float", "min": 0.0}, "rb": {"type": "float", "min": 0.0}, "d": {"type": "float", "min": 0.0}}


@register_symbol
class RoundCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_round_cone
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"r1": {"type": "float", "min": 0.0}, "r2": {"type": "float", "min": 0.0}, "h": {"type": "float", "min": 0.0}}


@register_symbol
class ArbitraryRoundCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_arbitrary_round_cone
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "r1": {"type": "float", "min": 0.0}, "r2": {"type": "float", "min": 0.0}}


@register_symbol
class InexactEllipsoid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_ellipsoid
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"r": {"type": "Vector[3]"}}


@register_symbol
class RevolvedVesica3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_revolved_vesica
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "w": {"type": "float", "min": 0.0}}


@register_symbol
class Rhombus3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_rhombus
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"la": {"type": "float", "min": 0.0}, "lb": {"type": "float", "min": 0.0}, "h": {"type": "float", "min": 0.0}, "ra": {"type": "float", "min": 0.0}}


@register_symbol
class Octahedron3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_octahedron
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"s": {"type": "float", "min": 0.0}}


@register_symbol
class InexactOctahedron3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_octahedron
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"s": {"type": "float", "min": 0.0}}


@register_symbol
class Pyramid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_pyramid
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"h": {"type": "float", "min": 0.0}}


@register_symbol
class Triangle3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_triangle
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "c": {"type": "Vector[3]"}}


@register_symbol
class Quadrilateral3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_quadrilateral
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"a": {"type": "Vector[3]"}, "b": {"type": "Vector[3]"}, "c": {"type": "Vector[3]"}, "d": {"type": "Vector[3]"}}


@register_symbol
class NoParamCuboid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_no_param_cuboid
    - geometry_nodes.geonodes.create_cuboid_node_seq
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {}


@register_symbol
class NoParamSphere3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_no_param_sphere
    - geometry_nodes.geonodes.create_sphere_node_seq
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {}


@register_symbol
class NoParamCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_no_param_cylinder
    - geometry_nodes.geonodes.create_cylinder_node_seq
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {}


@register_symbol
class InexactSuperQuadric3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_super_quadrics
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"skew_vec": {"type": "Vector[3]"}, "epsilon_1": {"type": "float", "min": 0.0}, "epsilon_2": {"type": "float", "min": 0.0}}


@register_symbol
class InexactAnisotropicGaussian3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_anisotropic_gaussian
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"center": {"type": "Vector[3]"}, "axial_radii": {"type": "Vector[3]"}, "scale_constant": {"type": "float"}}


@register_symbol
class SDFGrid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_sdf_grid
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"sdf_grid": {"type": "Tensor[float, (D,H,W)]"},
                 "name": {"type": "string"}, 
                 "bound_threshold": {"type": "float", "optional": True}
                 }


@register_symbol
class NullExpression3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf_null_op
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {}
