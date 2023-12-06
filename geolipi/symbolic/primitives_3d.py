from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
from inspect import signature
from sympy import Tuple as SympyTuple


class Primitive3D(GLFunction):
    """Functions for declaring 3D primitives."""
    ...

    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.pretty_print(tabs=tabs+1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    item = [f"{x:.3f}" for x in arg]
                    item = ', '.join(item)
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
    ...


class Box3D(Primitive3D):
    ...

class Cuboid3D(Box3D):
    ...


class RoundedBox3D(Primitive3D):
    ...


class BoxFrame3D(Primitive3D):
    ...

class Torus3D(Primitive3D):
    ...

class CappedTorus3D(Primitive3D):
    ...


class Link3D(Primitive3D):
    ...


class InfiniteCylinder3D(Primitive3D):
    ...


class Cone3D(Primitive3D):
    ...


class InexactCone3D(Primitive3D):
    ...


class InfiniteCone3D(Primitive3D):
    ...


class Plane3D(Primitive3D):
    ...


class HexPrism3D(Primitive3D):
    ...


class TriPrism3D(Primitive3D):
    ...


class Capsule3D(Primitive3D):
    ...


class VerticalCapsule3D(Primitive3D):
    ...


class VerticalCappedCylinder3D(Primitive3D):
    ...


class CappedCylinder3D(Primitive3D):
    ...


class Cylinder3D(CappedCylinder3D):
    ...


class ArbitraryCappedCylinder3D(Primitive3D):
    ...


class RoundedCylinder3D(Primitive3D):
    ...


class CappedCone3D(Primitive3D):
    ...


class ArbitraryCappedCone(Primitive3D):
    ...


class SolidAngle3D(Primitive3D):
    ...


class CutSphere3D(Primitive3D):
    ...


class CutHollowSphere(Primitive3D):
    ...


class DeathStar3D(Primitive3D):
    ...


class RoundCone3D(Primitive3D):
    ...


class ArbitraryRoundCone3D(Primitive3D):
    ...


class InexactEllipsoid3D(Primitive3D):
    ...


class RevolvedVesica3D(Primitive3D):
    ...


class Rhombus3D(Primitive3D):
    ...


class Octahedron3D(Primitive3D):
    ...


class InexactOctahedron3D(Primitive3D):
    ...


class Pyramid3D(Primitive3D):
    ...


class Triangle3D(Primitive3D):
    ...


class Quadrilateral3D(Primitive3D):
    ...


class NoParamCuboid3D(Primitive3D):
    ...


class NoParamSphere3D(Primitive3D):
    ...


class NoParamCylinder3D(Primitive3D):
    ...


class InexactSuperQuadrics3D(Primitive3D):
    ...

class InexactAnisotropicGaussian3D(Primitive3D):
    ...
class PreBakedPrimitive3D(Primitive3D):
    ...


class NullExpression3D(Primitive3D):
    ...