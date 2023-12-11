import torch as th
from sympy import Tuple as SympyTuple
from .base_symbolic import GLExpr, GLFunction
from .primitives_higher import LinearExtrude3D, QuadraticBezierExtrude3D


class Point3D(GLFunction):
    """
    Class for defining 3D points. Used for attachment operators.
    """


class PointRef(GLFunction):
    """
    This function is used to create reference to parameterized attachment points.
    Used for GeoCODE primitives.
    """
    def __str__(self):
        args = self.args[1:]
        expr_hash = hash(args[0])
        replaced_args = [expr_hash] + [self.lookup_table.get(arg, arg) for arg in args]
        return f"{self.func.__name__}({', '.join(map(str, replaced_args))})"

    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args[1:]
        n_tabs = tab_str * tabs
        expr_hash = hash(args[0])
        replaced_args = [expr_hash] + [self.lookup_table.get(arg, arg) for arg in args]
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

    # No need to resolve it before hand?
    def doit(self, deep=True, **hints):
        curve_expression = self.args[0]
        t_val = self.lookup_table[self.args[1]]
        # delta = self.args[2] - if required.
        if isinstance(curve_expression, QuadraticBezierExtrude3D):
            start_point = curve_expression.args[1]
            control_point = curve_expression.args[2]
            end_point = curve_expression.args[3]
            start_point = curve_expression.lookup_table[start_point]
            control_point = curve_expression.lookup_table[control_point]
            end_point = curve_expression.lookup_table[end_point]
            point = (
                th.pow(1 - t_val, 2) * start_point
                + 2 * (1 - t_val) * t_val * control_point
                + th.pow(t_val, 2) * end_point
            )
            # point_expr = Point3D(point)
        elif isinstance(curve_expression, LinearExtrude3D):
            start_point = curve_expression.args[0]
            end_point = curve_expression.args[1]
            start_point = curve_expression.lookup_table[start_point]
            end_point = curve_expression.lookup_table[end_point]
            delta = end_point - start_point
            point = start_point + delta * t_val
            # point_expr = Point3D(point)
        return point
