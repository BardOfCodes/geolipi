
import sys
import sympy as sp
import torch as th
from typing import Optional, Union, Callable, Any, TypeVar

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from .patched_functools import singledispatch

from geolipi.symbolic.base import GLExpr, GLFunction
import geolipi.symbolic as gls
from geolipi.symbolic.resolve import resolve_macros
from geolipi.symbolic import Revolution3D
from geolipi.symbolic.symbol_types import (
    MACRO_TYPE,
    MOD_TYPE,
    PRIM_TYPE,
    COMBINATOR_TYPE,
    TRANSFORM_TYPE,
    POSITIONALMOD_TYPE,
    SDFMOD_TYPE,
    HIGHER_PRIM_TYPE,
    COLOR_MOD,
    APPLY_COLOR_TYPE,
    SVG_COMBINATORS,
    UNOPT_ALPHA,
    EXPR_TYPE,
    SUPERSET_TYPE
)
from .sketcher import Sketcher
from .maps import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP, COLOR_FUNCTIONS, COLOR_MAP
from .constants import EPSILON
from .sympy_to_torch import SYMPY_TO_TORCH, TEXT_TO_SYMPY


### Create a Evaluate wrapper -> This will create the coords and may be different in different derivative languages.

def recursive_evaluate(expression: SUPERSET_TYPE, sketcher: Sketcher, 
             secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None, 
             *args, **kwargs) -> th.Tensor:
    """
    Evaluates a GeoLIPI expression using the provided sketcher and coordinates.
    
    Parameters:
        expression (SUPERSET_TYPE): The GeoLIPI expression to evaluate.
        sketcher (Sketcher): The sketcher object used for evaluation.
        secondary_sketcher (Sketcher, optional): A secondary sketcher for higher-order primitives.
        coords (th.Tensor, optional): Coordinates for evaluation. If None, generated from sketcher.
        
    Returns:
        th.Tensor: The result of evaluating the expression.
    """
    if coords is None:
        coords = sketcher.get_homogenous_coords()
    else:
        coords_dim = coords.shape[-1]
        if coords_dim == sketcher.n_dims:
            coords = sketcher.make_homogenous_coords(coords)
        elif coords_dim == sketcher.n_dims + 1:
            pass
        else:
            raise ValueError("Coordinates must have n_dims or n_dims - 1 dimensions.")
    return rec_eval(
        expression,
        sketcher,
        secondary_sketcher=secondary_sketcher,
        coords=coords,
        *args, **kwargs
    )

@singledispatch
def rec_eval(expression: SUPERSET_TYPE, sketcher: Sketcher,
             secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
             *args, **kwargs) -> th.Tensor:
    raise NotImplementedError(
        f"Expression type {type(expression)} is not supported for recursive evaluation."
    )

@rec_eval.register
def eval_macro(expression: MACRO_TYPE, sketcher: Sketcher,
               secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
               *args, **kwargs) -> th.Tensor:
    """
    Evaluates a GeoLIPI macro expression by resolving it to a concrete expression.
    """
    resolved_expr = resolve_macros(expression, device=sketcher.device)
    return rec_eval(resolved_expr, sketcher, secondary_sketcher, coords, 
                    *args, **kwargs)

@rec_eval.register
def eval_mod(expression: MOD_TYPE, sketcher: Sketcher, 
             secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
             *args, **kwargs) -> th.Tensor:
    sub_expr = expression.args[0]
    params = expression.args[1:]
    params = _parse_param_from_expr(expression, params, sketcher)
    # This is a hack unclear how to deal with other types)
    if isinstance(expression, TRANSFORM_TYPE):
        identity_mat = sketcher.get_affine_identity()
        new_transform = MODIFIER_MAP[type(expression)](identity_mat, *params)
        coords = th.einsum("ij,mj->mi", new_transform, coords)
        return rec_eval(sub_expr, sketcher, secondary_sketcher, coords,
                        *args, **kwargs)
    elif isinstance(expression, POSITIONALMOD_TYPE):
        # instantiate positions and send that as input with affine set to None
        coords = MODIFIER_MAP[type(expression)](coords, *params)
        return rec_eval(sub_expr, sketcher, secondary_sketcher, coords,
                        *args, **kwargs)
    elif isinstance(expression, SDFMOD_TYPE):
        # calculate sdf then create change before returning.
        sdf_estimate = rec_eval(sub_expr, sketcher, secondary_sketcher, coords,
                                *args, **kwargs)
        updated_sdf = MODIFIER_MAP[type(expression)](sdf_estimate, *params)
        return updated_sdf
    else:
        raise NotImplementedError(f"Modifier {expression} not implemented")
    
@rec_eval.register
def eval_prim(expression: PRIM_TYPE, sketcher: Sketcher,
              secondary_sketcher: Optional[Sketcher] = None, coords: th.Tensor = None,
              *args, **kwargs) -> th.Tensor:
    
    if isinstance(expression, HIGHER_PRIM_TYPE):
        params = expression.args[1:]
    else:
        params = expression.args

    params = _parse_param_from_expr(expression, params, sketcher)
    n_dims = sketcher.n_dims
    coords = coords[..., :n_dims] / (coords[..., n_dims : n_dims + 1] + EPSILON)

    if isinstance(expression, HIGHER_PRIM_TYPE):
        param_points, scale_factor = PRIMITIVE_MAP[type(expression)](
            coords, *params
        )
        distance_field_2d = param_points[..., :2].clone()
        pad = th.ones_like(distance_field_2d[:, :1])
        homo_dist_field_2d = th.cat([distance_field_2d, pad], dim=1)
        scale_t = 1.0  # get_scale(param_points)
        sub_expr = expression.args[0]
        # No Color OP but is 3D tracked scale to be used for 2D scale?
        assert secondary_sketcher is not None, "Secondary sketcher is required for higher-order primitives"
        in_plane_distance = rec_eval(sub_expr, secondary_sketcher, secondary_sketcher=None,
            coords=homo_dist_field_2d, *args, **kwargs)
        assert isinstance(in_plane_distance, th.Tensor), "In-plane distance must be a tensor"
        in_plane_distance = in_plane_distance / scale_t
        # TODO: Very Unclean.
        if isinstance(expression, Revolution3D):
            sdf = in_plane_distance
        else:
            height = (param_points[..., 2].clone() - 0.5) * scale_factor
            height = th.abs(height) - (0.5 * scale_factor)
            vec2 = th.stack([in_plane_distance, height], dim=-1)
            sdf = th.amax(vec2, -1) + th.norm(th.clip(vec2, min=0.0), -1)
        return sdf
    else:
        sdf = PRIMITIVE_MAP[type(expression)](coords, *params)
        return sdf


@rec_eval.register
def eval_comb(expression: COMBINATOR_TYPE, sketcher: Sketcher,
              secondary_sketcher: Optional[Sketcher] = None, coords: th.Tensor = None,
              *args, **kwargs) -> th.Tensor:
    
    tree_branches, param_list = [], []
    for arg in expression.args:
        if arg in expression.lookup_table:
            assert isinstance(arg, sp.Symbol), "Argument must be a symbol"
            param_list.append(expression.lookup_table[arg])
        else:
            tree_branches.append(arg)
    sdf_list = []
    for child in tree_branches:
        cur_sdf = rec_eval(child, sketcher, secondary_sketcher, coords=coords.clone(),
                           *args, **kwargs)
        sdf_list.append(cur_sdf)
    new_sdf = COMBINATOR_MAP[type(expression)](*sdf_list, *param_list)
    return new_sdf

@rec_eval.register
def eval_svg_comb(expression: SVG_COMBINATORS, sketcher: Sketcher,
                  secondary_sketcher: Optional[Sketcher] = None, coords: th.Tensor = None,
                  *args, **kwargs) -> th.Tensor:

    output_seq = []
    for expr in expression.args:
        canvas = rec_eval(expr, sketcher, secondary_sketcher,
            coords=coords.clone(), *args, **kwargs)
        output_seq.append(canvas)
    output_canvas = COLOR_FUNCTIONS[type(expression)](*output_seq)
    return output_canvas

@rec_eval.register
def eval_apply_color(expression: APPLY_COLOR_TYPE, sketcher: Sketcher,
                     secondary_sketcher: Optional[Sketcher] = None, coords: th.Tensor = None,
                     relaxed_occupancy: bool = False, relax_temperature: float = 0.0,
                     *args, **kwargs) -> th.Tensor:

    sdf_expr = expression.args[0]
    color = expression.args[1]
    # Get the sdf_expr:
    if not color in expression.lookup_table:
        assert isinstance(color, sp.Symbol), "Color must be a symbol"
        color = COLOR_MAP[color.name].to(sketcher.device)
    else:
        assert isinstance(color, sp.Symbol), "Color must be a symbol"
        color = expression.lookup_table[color]
    cur_sdf = rec_eval(sdf_expr, sketcher, secondary_sketcher, coords,
                       *args, **kwargs)

    if relaxed_occupancy:
        cur_occ = _smoothen_sdf(cur_sdf, relax_temperature)
    else:
        assert isinstance(cur_sdf, th.Tensor), "Current SDF must be a tensor"
        cur_occ = cur_sdf <= 0
    colored_canvas = COLOR_FUNCTIONS[type(expression)](cur_occ, color)
    return colored_canvas

def _smoothen_sdf(execution, temperature):
    output_tanh = th.tanh(execution * temperature)
    output_shape = th.nn.functional.sigmoid(-output_tanh * temperature)
    return output_shape


@rec_eval.register
def eval_color_mod(expression: COLOR_MOD, sketcher: Sketcher,
                   secondary_sketcher: Optional[Sketcher] = None, coords: th.Tensor = None,
                   *args, **kwargs) -> th.Tensor:
    color_expr = expression.args[0]
    colors = expression.args[1:]
    # Get the sdf_expr:
    eval_colors = []
    for color in colors:
        assert isinstance(color, sp.Symbol), "Color must be a symbol"
        if not color in expression.lookup_table:
            color = COLOR_MAP[color.name].to(sketcher.device)
        else:
            color = expression.lookup_table[color]
        eval_colors.append(color)
    colored_canvas = rec_eval(color_expr, sketcher, secondary_sketcher,coords,
                              *args, **kwargs)
    colored_canvas = COLOR_FUNCTIONS[type(expression)](colored_canvas, *eval_colors)
    return colored_canvas

@rec_eval.register
def eval_unopt_alpha(expression: UNOPT_ALPHA, sketcher: Sketcher,
                     secondary_sketcher: Optional[Sketcher] = None, coords: th.Tensor = None,
                     *args, **kwargs) -> th.Tensor:
    output_seq = []
    expr = expression.args[0]
    params = expression.args[1:]
    params = _parse_param_from_expr(expression, params, sketcher)
    canvas = rec_eval(expr, sketcher, secondary_sketcher, coords.clone(),
                      *args, **kwargs)
    output_canvas = COLOR_FUNCTIONS[type(expression)](canvas, *params)
    return output_canvas


@rec_eval.register
def eval_gl_expr(expression: EXPR_TYPE, sketcher: Sketcher,
                 secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
                 *args, **kwargs) -> th.Tensor:
    
    evaluated_args = []
    # print("expr args", expr.args)
    for arg in expression.args:
        if isinstance(arg, (GLFunction, GLExpr)):
            output = rec_eval(arg, sketcher, secondary_sketcher, coords, *args, **kwargs)
            evaluated_args.append(output)
            # just use tensor symbols here
        elif isinstance(arg, (sp.Float, sp.Integer)):
            new_value = float(arg)
            evaluated_args.append(new_value)
        else:
            print(type(arg))
            raise NotImplementedError(f"Params for{expression} not implemented")
    op = SYMPY_TO_TORCH[expression.func]
    output = op(*evaluated_args)
    return output

@rec_eval.register
def eval_gl_param(expression: gls.Param, sketcher: Sketcher,
                 secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
                 *args, **kwargs) -> th.Tensor:
    assert isinstance(expression.args[0], sp.Symbol), "Argument must be a symbol"
    return expression.lookup_table[expression.args[0]]

@rec_eval.register
def eval_gl_op(expression: gls.Operator, sketcher: Sketcher,
               secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
               *args, **kwargs) -> th.Tensor:
    if isinstance(expression, (gls.UnaryOperator, gls.BinaryOperator)):
        outputs = []
        args = expression.args[:-1]
        for arg in args:
            eval = rec_eval(arg, sketcher, secondary_sketcher, coords, *args, **kwargs)
            outputs.append(eval)
        op_arg = expression.args[-1]
        assert isinstance(op_arg, sp.Symbol), "Operator argument must be a string"
        op = SYMPY_TO_TORCH[TEXT_TO_SYMPY[op_arg.name]]
        output = op(*outputs)
        return output
    elif isinstance(expression, gls.VectorOperator):
        raise NotImplementedError(f"Vector Operator {expression} not implemented")
    else:
        raise NotImplementedError(f"Operator {expression} not implemented")

@rec_eval.register
def eval_gl_var(expression: gls.Variable, sketcher: Sketcher,
                secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
                *args, **kwargs) -> th.Tensor:
    raise NotImplementedError(f"Variable {expression} not supported in GeoLIPI. It is supported in derivatives")
# Need to add eval ops for Ops. 

# So as to have gs.Param + have OPs on top. 

def _parse_param_from_expr(expression, params, sketcher: Sketcher):
    # This can also be a graph.
    if params:
        param_list = []
        for ind, param in enumerate(params):
            if param in expression.lookup_table:
                cur_param = expression.lookup_table[param]
                param_list.append(cur_param)
            elif isinstance(param, SUPERSET_TYPE):
                param_list.append(rec_eval(param, sketcher))
            elif isinstance(param, (sp.Float, sp.Integer)):
                param_list.append(float(param))
            else:
                param_list.append(param)
        params = param_list
    return params


