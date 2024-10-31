from sympy import Symbol
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import torch as th
from geolipi.symbolic.resolve import resolve_macros
from geolipi.symbolic.types import (
    MACRO_TYPE,
    MOD_TYPE,
    TRANSLATE_TYPE,
    SCALE_TYPE,
    PRIM_TYPE,
    TRANSSYM_TYPE,
    COMBINATOR_TYPE,
    TRANSFORM_TYPE,
    POSITIONALMOD_TYPE,
    SDFMOD_TYPE,
    HIGERPRIM_TYPE,
    COLOR_MOD,
    APPLY_COLOR_TYPE,
    SVG_COMBINATORS,
    UNOPT_ALPHA
)
from .sketcher import Sketcher
from .utils import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP, COLOR_FUNCTIONS
from .common import EPSILON, RECTIFY_TRANSFORM
from .utils import COLOR_MAP
from geolipi.symbolic import Union, Intersection, Difference, Revolution3D
from .color_functions import source_over_seq


def recursive_evaluate(
    expression: GLFunction,
    sketcher: Sketcher,
    secondary_sketcher: Sketcher = None,
    initialize: bool = True,
    rectify_transform: bool = RECTIFY_TRANSFORM,
    coords: th.Tensor = None,
    tracked_scale: th.Tensor = None,
    relaxed_occupancy: bool = False,
    relax_temperature: float = 0.0,
    existing_canvas: th.Tensor = None,
):
    """
    Recursively evaluates a GeoLIPI expression to generate a signed distance field (SDF) or a color canvas.

    This function can handles all GeoLIPI operations but is slower than the other evaluation methods.

    Parameters:
        expression (GLFunction): The GLFunction expression to evaluate.
        sketcher (Sketcher): Primary sketcher object for SDF or color generation.
        secondary_sketcher (Sketcher, optional): Secondary sketcher for higher-order primitives.
        initialize (bool): Flag to initialize coordinates and scale if True. Used for the first call.
        rectify_transform (bool): Flag to rectify transformations.
        coords (th.Tensor, optional): Coordinates for evaluation. If None, generated from sketcher.
        tracked_scale (th.Tensor, optional): Scale tracking tensor. If None, generated from sketcher.
        relaxed_occupancy (bool): Flag to use relaxed occupancy for soft SDFs. Useful with Parameter Optimization of SVG expressions.
        relax_temperature (float): Temperature parameter for relaxed occupancy. Defaults to 0.0.

    Returns:
        th.Tensor: The resulting SDF or color canvas from evaluating the expression.
    """
    if initialize:
        if coords is None:
            coords = sketcher.get_homogenous_coords()
        else:
            coords = sketcher.make_homogenous_coords(coords)
        tracked_scale = sketcher.get_scale_identity()

    if isinstance(expression, MACRO_TYPE):
        resolved_expr = resolve_macros(expression, device=sketcher.device)
        return recursive_evaluate(
            resolved_expr,
            sketcher,
            secondary_sketcher=secondary_sketcher,
            initialize=False,
            rectify_transform=rectify_transform,
            coords=coords,
            tracked_scale=tracked_scale,
            relaxed_occupancy=relaxed_occupancy,
            relax_temperature=relax_temperature,
            existing_canvas=existing_canvas,

        )
    elif isinstance(expression, MOD_TYPE):
        sub_expr = expression.args[0]
        params = expression.args[1:]
        params = _parse_param_from_expr(expression, params)
        # This is a hack unclear how to deal with other types)
        if isinstance(expression, TRANSFORM_TYPE):
            if rectify_transform:
                if isinstance(expression, (TRANSLATE_TYPE, TRANSSYM_TYPE)):
                    params[0] = params[0] / tracked_scale
                elif isinstance(expression, SCALE_TYPE):
                    tracked_scale *= params[0]
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(expression)](identity_mat, *params)
            coords = th.einsum("ij,mj->mi", new_transform, coords)
            return recursive_evaluate(
                sub_expr,
                sketcher,
                secondary_sketcher=secondary_sketcher,
                initialize=False,
                rectify_transform=rectify_transform,
                coords=coords,
                tracked_scale=tracked_scale,
                relaxed_occupancy=relaxed_occupancy,
                relax_temperature=relax_temperature,
                existing_canvas=existing_canvas,
            )
        elif isinstance(expression, POSITIONALMOD_TYPE):
            # instantiate positions and send that as input with affine set to None
            coords = MODIFIER_MAP[type(expression)](coords, *params)
            return recursive_evaluate(
                sub_expr,
                sketcher,
                secondary_sketcher=secondary_sketcher,
                initialize=False,
                rectify_transform=rectify_transform,
                coords=coords,
                tracked_scale=tracked_scale,
                relaxed_occupancy=relaxed_occupancy,
                relax_temperature=relax_temperature,
                existing_canvas=existing_canvas,
            )
        elif isinstance(expression, SDFMOD_TYPE):
            # calculate sdf then create change before returning.
            sdf_estimate = recursive_evaluate(
                sub_expr,
                sketcher,
                secondary_sketcher=secondary_sketcher,
                initialize=False,
                rectify_transform=rectify_transform,
                coords=coords,
                tracked_scale=tracked_scale,
                relaxed_occupancy=relaxed_occupancy,
                relax_temperature=relax_temperature,
                existing_canvas=existing_canvas,
            )
            updated_sdf = MODIFIER_MAP[type(expression)](sdf_estimate, *params)
            return updated_sdf
    elif isinstance(expression, PRIM_TYPE):
        # create sdf and return.

        if isinstance(expression, HIGERPRIM_TYPE):
            params = expression.args[1:]
        else:
            params = expression.args

        params = _parse_param_from_expr(expression, params)
        n_dims = sketcher.n_dims
        coords = coords[..., :n_dims] / (coords[..., n_dims : n_dims + 1] + EPSILON)

        if isinstance(expression, HIGERPRIM_TYPE):
            param_points, scale_factor = PRIMITIVE_MAP[type(expression)](
                coords, *params
            )

            distance_field_2d = param_points[..., :2].clone()
            pad = th.ones_like(distance_field_2d[:, :1])
            homo_dist_field_2d = th.cat([distance_field_2d, pad], dim=1)
            scale_t = 1.0  # get_scale(param_points)
            sub_expr = expression.args[0]
            # No Color OP but is 3D tracked scale to be used for 2D scale?
            in_plane_distance = recursive_evaluate(
                sub_expr,
                secondary_sketcher,
                secondary_sketcher=None,
                initialize=False,
                rectify_transform=rectify_transform,
                coords=homo_dist_field_2d,
                tracked_scale=None,
            )
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
    elif isinstance(expression, COMBINATOR_TYPE):
        # what about parameterized combinators?
        tree_branches, param_list = [], []
        for arg in expression.args:
            if arg in expression.lookup_table:
                param_list.append(expression.lookup_table[arg])
            else:
                tree_branches.append(arg)
        sdf_list = []
        for child in tree_branches:
            cur_sdf = recursive_evaluate(
                child,
                sketcher,
                secondary_sketcher=secondary_sketcher,
                initialize=False,
                rectify_transform=rectify_transform,
                coords=coords.clone(),
                tracked_scale=tracked_scale.clone(),
                relaxed_occupancy=relaxed_occupancy,
                relax_temperature=relax_temperature,
                existing_canvas=existing_canvas,
            )
            sdf_list.append(cur_sdf)
        channel_count = sdf_list[0].shape[-1]
        if channel_count == 4:
            new_sdf = source_over_seq(*sdf_list)
        else:
            new_sdf = COMBINATOR_MAP[type(expression)](*sdf_list, *param_list)
        return new_sdf

    elif isinstance(expression, SVG_COMBINATORS):
        output_seq = []
        for expr in expression.args:
            canvas = recursive_evaluate(
                expr,
                sketcher,
                secondary_sketcher=secondary_sketcher,
                initialize=False,
                rectify_transform=rectify_transform,
                coords=coords.clone(),
                tracked_scale=tracked_scale.clone(),
                relaxed_occupancy=relaxed_occupancy,
                relax_temperature=relax_temperature,
                existing_canvas=existing_canvas,
            )
            output_seq.append(canvas)
        output_canvas = COLOR_FUNCTIONS[type(expression)](*output_seq)
        if existing_canvas is not None:
            output_canvas = source_over_seq(existing_canvas, output_canvas)
        return output_canvas

    elif isinstance(expression, APPLY_COLOR_TYPE):
        sdf_expr = expression.args[0]
        color = expression.args[1]
        # Get the sdf_expr:
        if not color in expression.lookup_table:
            color = COLOR_MAP[color.name].to(sketcher.device)
        else:
            color = expression.lookup_table[color]
        cur_sdf = recursive_evaluate(
            sdf_expr,
            sketcher,
            secondary_sketcher=secondary_sketcher,
            initialize=False,
            rectify_transform=rectify_transform,
            coords=coords,
            tracked_scale=tracked_scale,
            relaxed_occupancy=relaxed_occupancy,
            relax_temperature=relax_temperature,
            existing_canvas=existing_canvas,
        )
        if relaxed_occupancy:
            cur_occ = _smoothen_sdf(cur_sdf, relax_temperature)
        else:
            cur_occ = cur_sdf <= 0
        colored_canvas = COLOR_FUNCTIONS[type(expression)](cur_occ, color)
        return colored_canvas

    elif isinstance(expression, COLOR_MOD):
        color_expr = expression.args[0]
        colors = expression.args[1:]
        # Get the sdf_expr:
        eval_colors = []
        for color in colors:
            if not color in expression.lookup_table:
                color = COLOR_MAP[color.name].to(sketcher.device)
            else:
                color = expression.lookup_table[color]
            eval_colors.append(color)
        colored_canvas = recursive_evaluate(
            color_expr,
            sketcher,
            secondary_sketcher=secondary_sketcher,
            initialize=False,
            rectify_transform=rectify_transform,
            coords=coords,
            tracked_scale=tracked_scale,
            relaxed_occupancy=relaxed_occupancy,
            relax_temperature=relax_temperature,
            existing_canvas=existing_canvas,
        )
        colored_canvas = COLOR_FUNCTIONS[type(expression)](colored_canvas, *eval_colors)
        return colored_canvas
    elif isinstance(expression, UNOPT_ALPHA):

        output_seq = []
        expr = expression.args[0]
        params = expression.args[1:]
        params = _parse_param_from_expr(expression, params)
        canvas = recursive_evaluate(
            expr,
            sketcher,
            secondary_sketcher=secondary_sketcher,
            initialize=False,
            rectify_transform=rectify_transform,
            coords=coords.clone(),
            tracked_scale=tracked_scale.clone(),
            relaxed_occupancy=relaxed_occupancy,
            relax_temperature=relax_temperature,
            existing_canvas=existing_canvas,
        )
        output_canvas = COLOR_FUNCTIONS[type(expression)](canvas, *params)
        return output_canvas


def _parse_param_from_expr(expression, params):
    if params:
        param_list = []
        for ind, param in enumerate(params):
            if param in expression.lookup_table:
                cur_param = expression.lookup_table[param]
                param_list.append(cur_param)
            else:
                param_list.append(param)
        params = param_list
    return params


def _smoothen_sdf(execution, temperature):
    output_tanh = th.tanh(execution * temperature)
    output_shape = th.nn.functional.sigmoid(-output_tanh * temperature)
    return output_shape


def expr_to_sdf(
    expression: GLFunction,
    sketcher: Sketcher,
    secondary_sketcher: Sketcher = None,
    rectify_transform: bool = RECTIFY_TRANSFORM,
    coords: th.Tensor = None,
):
    """
    Converts a GeoLIPI SDF expression into a Signed Distance Field (SDF) using a sketcher.
    This function is faster than `recursive_evaluate` as it evaluates the expression using a stack-based approach. 
    However, it does not support all GeoLIPI operations, notably higher-order primitives, and certain modifiers. 

    Parameters:
        expression (GLExpr): The GLExpr expression to be converted to an SDF.
        sketcher (Sketcher): The primary sketcher object used for generating SDFs.
        rectify_transform (bool): Flag to apply rectified transformations. Defaults to RECTIFY_TRANSFORM.
        secondary_sketcher (Sketcher, optional): Secondary sketcher - Never used.
        coords (Tensor, optional): Custom coordinates to use for the SDF generation.

    Returns:
        Tensor: The generated SDF corresponding to the input expression.
    """
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    operator_stack = []
    operator_nargs_stack = []
    operator_params_stack = []
    execution_pointer_index = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]
    parser_list = [expression]
    while parser_list:
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=sketcher.device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, COMBINATOR_TYPE):
            operator_stack.append(type(cur_expr))
            # what about parameterized combinators?
            tree_branches, cur_params = [], []
            for arg in cur_expr.args:
                if arg in cur_expr.lookup_table:
                    cur_params.append(cur_expr.lookup_table[arg])
                else:
                    tree_branches.append(arg)
            n_args = len(tree_branches)
            operator_nargs_stack.append(n_args)
            operator_params_stack.append(cur_params)
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)
            next_to_parse = tree_branches[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1:]
            params = _parse_param_from_expr(cur_expr, params)
            # This is a hack unclear how to deal with other types)
            if rectify_transform:
                if isinstance(cur_expr, (TRANSLATE_TYPE, TRANSSYM_TYPE)):
                    scale = scale_stack[-1]
                    params[0] = params[0] / scale
                elif isinstance(cur_expr, SCALE_TYPE):
                    scale_stack[-1] *= params[0]

            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, *params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            params = _parse_param_from_expr(cur_expr, params)
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            cur_coords = sketcher.get_coords(transform, points=coords)
            execution = PRIMITIVE_MAP[type(cur_expr)](cur_coords, *params)
            execution_stack.append(execution)
        else:
            raise ValueError(f"Unknown expression type {type(cur_expr)}")

        while (
            operator_stack
            and len(execution_stack) - execution_pointer_index[-1]
            >= operator_nargs_stack[-1]
        ):
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = execution_pointer_index.pop()
            params = operator_params_stack.pop()
            args = execution_stack[-n_args:]
            new_canvas = COMBINATOR_MAP[operator](*args, *params)
            execution_stack = execution_stack[:-n_args] + [new_canvas]

    assert len(execution_stack) == 1
    sdf = execution_stack[0]
    return sdf


def expr_to_colored_canvas(
    expression: GLExpr,
    sketcher: Sketcher,
    rectify_transform=RECTIFY_TRANSFORM,
    relaxed_occupancy=False,
    relax_temperature=0.0,
    coords=None,
    canvas=None,
):
    """
    TODO: This function is to be tested.
    """
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    execution_pointer_index = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]
    parser_list = [expression]
    color_stack = [Symbol("gray")]
    if canvas is None:
        colored_canvas = sketcher.get_color_canvas()
    else:
        colored_canvas = canvas
    while parser_list:
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=sketcher.device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, COMBINATOR_TYPE):
            n_args = len(cur_expr.args)
            # chain extensions
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            color = color_stack.pop()
            if isinstance(color, th.Tensor):
                color_chain = [color.clone() for x in range(n_args)]
            else:
                color_chain = [color for x in range(n_args)]
            color_stack.extend(color_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)

            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, APPLY_COLOR_TYPE):
            color_stack.pop()
            color_stack.append(cur_expr.args[1])
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):
            params = expression.args[1:]
            params = _parse_param_from_expr(expression, params)
            if rectify_transform:
                if isinstance(expression, (TRANSLATE_TYPE, TRANSSYM_TYPE)):
                    params[0] = params[0] / scale_stack[-1]
                elif isinstance(expression, SCALE_TYPE):
                    scale_stack[-1] *= params[0]
            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            params = _parse_param_from_expr(cur_expr, params)
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            cur_coords = sketcher.get_coords(transform, points=coords)
            execution = PRIMITIVE_MAP[type(cur_expr)](cur_coords, *params)
            # At this point use color code to color the primitive
            color = color_stack.pop()
            if isinstance(color, Symbol):
                valid_color = COLOR_MAP[color.name].to(sketcher.device)
            else:
                valid_color = color
            # For differentiable relax, this also has to be relaxed.

            if relaxed_occupancy:
                # from the sdf execution compute occupancy
                occ = relaxed_occupancy(execution, temperature=relax_temperature)
            else:
                occ = execution <= 0
            # Amazing source: https://ciechanow.ski/alpha-compositing/
            alpha_a = occ[..., None] * valid_color[0, 3:4]
            color_a = occ.view(occ.shape[0], 1) * valid_color[..., :3]
            alpha_b = colored_canvas[..., 3:4]
            color_b = colored_canvas[..., :3]
            a_o = alpha_a + alpha_b * (1 - alpha_a)
            color_o = (color_a * alpha_a + color_b * alpha_b * (1 - alpha_a)) / a_o
            colored_canvas = th.cat([color_o, a_o], dim=-1)
        else:
            raise ValueError(f"Unknown expression type {type(cur_expr)}")

    return colored_canvas
