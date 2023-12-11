from .base_symbolic import GLExpr, GLFunction
from sympy import Tuple as SympyTuple


class Primitive3D(GLFunction):
    """Functions for declaring 3D primitives."""

    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
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


class Sphere3D(Primitive3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_sphere
    Read evaluator specific documentation for more.
    """


class Box3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_box
    Read evaluator specific documentation for more.
    """


class Cuboid3D(Box3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_box
    Read evaluator specific documentation for more.
    """


class RoundedBox3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_rounded_box
    Read evaluator specific documentation for more.
    """


class BoxFrame3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_box_frame
    Read evaluator specific documentation for more.
    """


class Torus3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_torus
    Read evaluator specific documentation for more.
    """


class CappedTorus3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_torus
    Read evaluator specific documentation for more.
    """


class Link3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_link
    Read evaluator specific documentation for more.
    """


class InfiniteCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_infinite_cylinder
    Read evaluator specific documentation for more.
    """


class Cone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_cone
    Read evaluator specific documentation for more.
    """


class InexactCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_cone
    Read evaluator specific documentation for more.
    """


class InfiniteCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_plane
    # TODO: Add support (Used in CSGStump)
    Read evaluator specific documentation for more.
    """


class Plane3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_plane
    Read evaluator specific documentation for more.
    """


class HexPrism3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_hex_prism
    Read evaluator specific documentation for more.
    """


class TriPrism3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_tri_prism
    Read evaluator specific documentation for more.
    """


class Capsule3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capsule
    Read evaluator specific documentation for more.
    """


class VerticalCapsule3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_vertical_capsule
    Read evaluator specific documentation for more.
    """


class VerticalCappedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_cylinder
    Read evaluator specific documentation for more.
    """


class CappedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_cylinder
    Read evaluator specific documentation for more.
    """


class Cylinder3D(CappedCylinder3D):

    """
    This class is mapped to the following evaluator function(s):
    # TODO: No Mapping yet.
    Read evaluator specific documentation for more.
    """


class ArbitraryCappedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_arbitrary_capped_cylinder
    Read evaluator specific documentation for more.
    """


class RoundedCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_rounded_cylinder
    Read evaluator specific documentation for more.
    """


class CappedCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_capped_cone
    Read evaluator specific documentation for more.
    """


class ArbitraryCappedCone(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_arbitrary_capped_cone
    Read evaluator specific documentation for more.
    """


class SolidAngle3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_solid_angle
    Read evaluator specific documentation for more.
    """


class CutSphere3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_cut_sphere
    Read evaluator specific documentation for more.
    """


class CutHollowSphere(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_cut_hollow_sphere
    Read evaluator specific documentation for more.
    """


class DeathStar3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_death_star
    Read evaluator specific documentation for more.
    """


class RoundCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_round_cone
    Read evaluator specific documentation for more.
    """


class ArbitraryRoundCone3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_arbitrary_round_cone
    Read evaluator specific documentation for more.
    """


class InexactEllipsoid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_ellipsoid
    Read evaluator specific documentation for more.
    """


class RevolvedVesica3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_revolved_vesica
    Read evaluator specific documentation for more.
    """


class Rhombus3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_rhombus
    Read evaluator specific documentation for more.
    """


class Octahedron3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_octahedron
    Read evaluator specific documentation for more.
    """


class InexactOctahedron3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_octahedron
    Read evaluator specific documentation for more.
    """


class Pyramid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_pyramid
    Read evaluator specific documentation for more.
    """


class Triangle3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_triangle
    Read evaluator specific documentation for more.
    """


class Quadrilateral3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_quadrilateral
    Read evaluator specific documentation for more.
    """


class NoParamCuboid3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_no_param_cuboid
    - geometry_nodes.geonodes.create_cuboid_node_seq
    Read evaluator specific documentation for more.
    """


class NoParamSphere3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_no_param_sphere
    - geometry_nodes.geonodes.create_sphere_node_seq
    Read evaluator specific documentation for more.
    """


class NoParamCylinder3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_no_param_cylinder
    - geometry_nodes.geonodes.create_cylinder_node_seq
    Read evaluator specific documentation for more.
    """


class InexactSuperQuadrics3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_super_quadrics
    Read evaluator specific documentation for more.
    """


class InexactAnisotropicGaussian3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf3d_inexact_anisotropic_gaussian
    Read evaluator specific documentation for more.
    """


class PreBakedPrimitive3D(Primitive3D):

    """
    Only for Geometry Nodes. This class is mapped to the following evaluator function(s):
    - geometry_nodes.geonodes.create_prebaked_primitive_node_seq
    Read evaluator specific documentation for more.
    """


class NullExpression3D(Primitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_3d.sdf_null_op
    Read evaluator specific documentation for more.
    """
