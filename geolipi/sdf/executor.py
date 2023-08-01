
from sympy import Function
from geolipi.symbolic import Primitive, Combinator, Transform
from geolipi.symbolic.primitives import Cuboid, Sphere, Cylinder, Cone
from geolipi.symbolic.combinators import Union, Intersection, Difference, Complement
from geolipi.symbolic.transforms import Translate, EulerRotate, QuaternionRotate, Scale, ReflectTransform
from .sketcher import Sketch3D
from .transforms import get_affine_translate, get_affine_scale, get_affine_rotate_euler
from .sdf_functions import sdf_cuboid, sdf_sphere, fixed_cuboid, fixed_sphere
from .sdf_functions import sdf_union, sdf_intersection, sdf_difference

transform_map = {
    Translate: get_affine_translate,
    EulerRotate: get_affine_rotate_euler,
    # QuaternionRotate: quat_rotate,
    Scale: get_affine_scale,
    # ReflectTransform: reflect,
}
primitive_map = {
    Cuboid: sdf_cuboid,
    Sphere: sdf_sphere,
    # Cylinder: sdf_cylinder,
    # Cone: sdf_cone,
}

combinator_map = {
    Union: sdf_union,
    Intersection: sdf_intersection,
    Difference: sdf_difference,
    # Complement: sdf_complement,
}


def execute(expression: Function, sketcher: Sketch3D, mode: str = 'naive'):
    if mode == 'naive':
        return naive_execute(expression, sketcher)
    elif mode == 'fast':
        compiled_obj = fast_compile(expression, sketcher)
        return fast_execute(compiled_obj)
    else:
        raise ValueError(f'Unknown mode {mode}')


def batch_execute(batch_compiled_objs):
    ...


def naive_execute(expression: Function, sketcher: Sketch3D, doit=False):
    transforms_stack = [sketcher.get_affine_identity()]
    canvas_stack = []
    operator_stack = []
    operator_args_stack = []
    if doit:
        expression = expression.doit()
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, Combinator):
            operator_stack.append(cur_expr)
            n_args = len(cur_expr.args)
            operator_args_stack.append(n_args)
            transform = transforms_stack.pop()
            transform_chain = [transform] * n_args
            transforms_stack.extend(transform_chain)
            next_parse = cur_expr.args[::-1]
            parser_list.extend(next_parse)
        elif isinstance(cur_expr, Transform):
            transform = transforms_stack.pop()
            params = cur_expr.args[1]
            transform = transform_map[type(cur_expr)](transform, params)
            transforms_stack.append(transform)
            next_parse = cur_expr.args[0]
            parser_list.append(next_parse)
        elif isinstance(cur_expr, Primitive):
            transform = transforms_stack.pop()
            coords = sketcher.get_coords(transform)
            params = cur_expr.args
            if len(params) == 0:
                params = sketcher.default_params[type(cur_expr)].detach()
            canvas = primitive_map[type(cur_expr)](coords, params)
            canvas_stack.append(canvas)
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

        while (operator_stack and len(canvas_stack) >= operator_args_stack[-1]):
            n_args = operator_args_stack.pop()
            operator = operator_stack.pop()
            args = canvas_stack[-n_args:]
            new_canvas = combinator_map[type(operator)](*args)
            canvas_stack = canvas_stack[:-n_args] + [new_canvas]

    assert len(canvas_stack) == 1
    return canvas_stack[0]


def fast_compile(expression: Function):
    # In first traversal, make all transformations chains, and combine primitives to execute.
    # also convert the CSG formula into DNF or CNF form.
    ...


def fast_execute(compiled_obj):
    ...
