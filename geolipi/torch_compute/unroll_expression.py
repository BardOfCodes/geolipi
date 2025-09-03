
from sqlite3 import paramstyle
import sys
import sympy as sp
import ast
import torch as th
from typing import Optional, Union, Callable, Any, TypeVar, List, Tuple, Dict

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from .patched_functools import singledispatch

from geolipi.symbolic.base import GLExpr, GLFunction, GLBase
import geolipi.symbolic as gls
from geolipi.symbolic.resolve import resolve_macros
from geolipi.symbolic import Revolution3D
from geolipi.symbolic.types import (
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
from .evaluate_expression import _parse_param_from_expr
"""
Unroll the expression into a list of codelines. 
inline_params: bool -> else we store the params as var_x - but that is useless. 
"""

class LocalContext:
    def __init__(self):
        self.transform_stack = ["transform_0"]
        self.coords_stack = ["coords_0"]
        self.res_stack = []
        self.var_count = 0
        self.transform_count = 0
        self.coords_count = 0
        self.res_count = 0
        self.codelines = []
        self.dependencies = dict()

    def add_codeline(self, codeline: str):
        self.codelines.append(codeline)

    def add_dependency(self, func_name: str, func: Any):
        self.dependencies[func_name] = func
    

### Create a Evaluate wrapper -> This will create the coords and may be different in different derivative languages.

def unroll_expression(in_expr: GLBase, sketcher: Sketcher, 
             secondary_sketcher: Optional[Sketcher] = None,
             varname_expr: bool = True,
             param_mode: str = "unrolled", 
             isolated_vars: bool = True,
             param_mapping: Optional[Dict[str, th.Tensor]] = None,
             *args, **kwargs) -> Tuple[Callable, ast.FunctionDef, LocalContext]:
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
    # PRELIMS
    # Middle
    if varname_expr:
        expression, _= in_expr.get_varnamed_expr()

        if param_mapping is None and param_mode == "embedded":
            # generate param_dict directly.
            tensor_list = in_expr.gather_tensor_list()
            assert isinstance(tensor_list[0], th.Tensor), "Tensor list must be a list of tensors"
            param_mapping = {f"var_{i}": (tensor if isinstance(tensor, th.Tensor) else tensor[0]) for i, tensor in enumerate(tensor_list)}
    else:
        expression = in_expr

        
    local_context = LocalContext()
    local_context.dependencies['th'] = th
    local_context.dependencies['transform_0'] = sketcher.get_affine_identity()

    local_context = rec_unroll(
        expression,
        local_context=local_context,
        sketcher=sketcher,
        secondary_sketcher=secondary_sketcher, isolated_vars=isolated_vars,
        *args, **kwargs
    )

    # 1. Extract parameter names from codelines (var_X variables)
    import re
    param_names = set()
    for codeline in local_context.codelines:
        # Find var_X references in the codeline
        matches = re.findall(r'\bvar_(\d+)\b', codeline)
        for match in matches:
            param_names.add(f'var_{match}')

    # sort param_names
    param_names = sorted(param_names, key=lambda x: int(x.split('_')[-1]))
    
    # 2. Create function signature
    if param_mode == "embedded":
        assert param_mapping is not None, "Param mapping must be provided if inline_params is True"
        local_context.dependencies.update(param_mapping)
        fn_args = [ast.arg(arg='coords_0')]
    elif param_mode == "varlist":
        # varlist and also add varlist unpacking to codelines.
        fn_args = [ast.arg(arg='coords_0'), ast.arg(arg='varlist')]
        unpack_line = f"{', '.join(param_names)} = varlist"
        local_context.codelines = [unpack_line,] + local_context.codelines
    elif param_mode == "unrolled":
        fn_args = [ast.arg(arg='coords_0')]
        for param in param_names:
            fn_args.append(ast.arg(arg=param))
    else:
        raise NotImplementedError(f"Param mode {param_mode} not implemented")

    # 3. Parse codelines into AST statements
    body_statements = []
    for codeline in local_context.codelines:
        parsed = ast.parse(codeline).body[0]
        body_statements.append(parsed)
    
    # 4. Add return statement based on res_stack
    assert len(local_context.res_stack) == 1, "There should be exactly one result in the res_stack"
    final_result = local_context.res_stack.pop()
    return_stmt = ast.Return(value=ast.Name(id=final_result, ctx=ast.Load()))
    body_statements.append(return_stmt)
    
    # 5. Create AST function definition
    func_def = ast.FunctionDef(
        name="compiled_fn",
        args=ast.arguments(
            posonlyargs=[],
            args=fn_args,
            vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=body_statements,
        decorator_list=[]
    )
    
    # 6. Wrap in module and compile
    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)
    
    # 7. Compile and execute
    env = {x:y for x, y in local_context.dependencies.items()}

    exec(compile(module, "<ast>", "exec"), env)
    
    compiled_function = env["compiled_fn"]  # type: ignore
    return compiled_function, func_def, local_context # type: ignore

@singledispatch
def rec_unroll(expression: SUPERSET_TYPE, local_context: LocalContext, sketcher: Sketcher,
             secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
             *args, **kwargs) -> LocalContext:
    raise NotImplementedError(
        f"Expression type {type(expression)} is not supported for recursive evaluation."
    )

@rec_unroll.register
def unroll_macro(expression: MACRO_TYPE, local_context: LocalContext, sketcher: Sketcher,
               secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
               *args, **kwargs) -> LocalContext:
    """
    Evaluates a GeoLIPI macro expression by resolving it to a concrete expression.
    """
    resolved_expr = resolve_macros(expression, device=sketcher.device)
    return rec_unroll(resolved_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

@rec_unroll.register
def unroll_mod(expression: MOD_TYPE, local_context: LocalContext, sketcher: Sketcher, 
             secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
             *args, **kwargs) -> LocalContext:

    sub_expr = expression.args[0]
    params = expression.args[1:]
    func_name = expression.__class__.__name__
    params = _process_params(expression, params, local_context, sketcher)

    function = MODIFIER_MAP[type(expression)]
    # This is a hack unclear how to deal with other types)
    if isinstance(expression, TRANSFORM_TYPE):
        cur_transform = local_context.transform_stack.pop()
        if isolated_vars:
            local_context.transform_count += 1
        new_transform = f"transform_{local_context.transform_count}"
        code_line = f"{new_transform} = {func_name}(IDENTITY.clone(), {params})"
        local_context.add_codeline(code_line)
        code_line = f"{new_transform} = th.matmul({new_transform}, {cur_transform})"
        local_context.add_codeline(code_line)
        local_context.add_dependency(func_name, function)
        local_context.add_dependency("IDENTITY", sketcher.get_affine_identity())

        local_context.transform_stack.append(new_transform)
        assert isinstance(sub_expr, (gls.GLFunction, gls.GLExpr)), "Sub expression must be a GLFunction or GLExpr"
        return rec_unroll(sub_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
    elif isinstance(expression, POSITIONALMOD_TYPE):
        # instantiate positions and send that as input with affine set to None
        cur_coords = local_context.coords_stack.pop()
        # Maybe we only need to add when we are duplicating, i.e. Union etc. otherwise no.
        if isolated_vars:
            local_context.coords_count += 1
        new_coords = f"coords_{local_context.coords_count}"
        code_line = f"{new_coords} = {func_name}({cur_coords}, {params})"
        local_context.add_codeline(code_line)
        local_context.add_dependency(func_name, function)
        local_context.coords_stack.append(new_coords)
        assert isinstance(sub_expr, (gls.GLFunction, gls.GLExpr)), "Sub expression must be a GLFunction or GLExpr"
        return rec_unroll(sub_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

    elif isinstance(expression, SDFMOD_TYPE):
        # calculate sdf then create change before returning.
        local_context = rec_unroll(sub_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
        cur_res = local_context.res_stack.pop()
        if isolated_vars:
            local_context.res_count += 1
        new_res = f"res_{local_context.res_count}"
        code_line = f"{new_res} = {func_name}({cur_res}, {params})"
        local_context.add_codeline(code_line)
        local_context.add_dependency(func_name, function)
        local_context.res_stack.append(new_res)
        return local_context
    else:
        raise NotImplementedError(f"Modifier {expression} not implemented")
    
@rec_unroll.register
def unroll_prim(expression: PRIM_TYPE, local_context: LocalContext, sketcher: Sketcher,
              secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
              *args, **kwargs) -> LocalContext:
    
    if isinstance(expression, HIGHER_PRIM_TYPE):
        params = expression.args[1:]
    else:
        params = expression.args

    func_name = expression.__class__.__name__
    params = _process_params(expression, params, local_context, sketcher)
    # This is for correction? 
    n_dims = sketcher.n_dims
    function = PRIMITIVE_MAP[type(expression)]
    if isinstance(expression, HIGHER_PRIM_TYPE):
        raise NotImplementedError("Higher-order primitives not implemented")
    else:
        cur_transform = local_context.transform_stack.pop()
        cur_coords = local_context.coords_stack.pop()
        if isolated_vars:
            local_context.coords_count += 1
        
        new_coords = f"coords_{local_context.coords_count}"
        # code_line = f"{new_coords} = th.einsum('ij,mj->mi', {cur_transform}, {cur_coords})"
        code_line = f"{new_coords} = th.matmul({cur_coords}, {cur_transform}.transpose(0, 1))"
        local_context.add_codeline(code_line)
        code_line = f"{new_coords} = {new_coords}[..., :{n_dims}] / ({new_coords}[..., {n_dims} : {n_dims + 1}] + {float(EPSILON)})"
        local_context.add_codeline(code_line)
        local_context.res_count += 1
        new_res = f"res_{local_context.res_count}"
        code_line = f"{new_res} = {func_name}({new_coords}, {params});"
        local_context.add_codeline(code_line)
        local_context.add_dependency(func_name, function)
        local_context.res_stack.append(new_res)
        return local_context


@rec_unroll.register
def unroll_comb(expression: COMBINATOR_TYPE, local_context: LocalContext, sketcher: Sketcher,
              secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
              *args, **kwargs) -> LocalContext:
    
    func_name = expression.__class__.__name__
    tree_branches, param_list = [], []
    function = COMBINATOR_MAP[type(expression)]
    if isinstance(expression, (gls.SmoothUnion, gls.SmoothIntersection, gls.SmoothDifference)):
        tree_branches = [arg for arg in expression.args[:-1]]
        param_list = [expression.args[-1]]
    else:
        tree_branches, param_list = [], []
        tree_branches = [arg for arg in expression.args]
    if param_list:
        param_list = _process_params(expression, param_list, local_context, sketcher)
    n_children = len(tree_branches)
    
    cur_coords = local_context.coords_stack.pop()
    cur_transform = local_context.transform_stack.pop()
    for child in tree_branches:
        if not isolated_vars: 
            # we do a copy here of the transform and of the coords.
            local_context.transform_count += 1
            local_context.coords_count += 1
            new_transform = f"transform_{local_context.transform_count}"
            new_coords = f"coords_{local_context.coords_count}"
            code_line = f"{new_transform} = {cur_transform}.clone()"
            local_context.add_codeline(code_line)
            code_line = f"{new_coords} = {cur_coords}.clone()"
            local_context.add_codeline(code_line)
            local_context.transform_stack.append(new_transform)
            local_context.coords_stack.append(new_coords)
        else:
            local_context.transform_stack.append(cur_transform)
            local_context.coords_stack.append(cur_coords)
        local_context.coords_stack.append(cur_coords)
        assert isinstance(child, (gls.GLFunction, gls.GLExpr)), "Child must be a GLFunction or GLExpr"
        local_context = rec_unroll(child, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

    children = [local_context.res_stack.pop() for _ in range(n_children)]
    # reverse the children
    children = children[::-1]
    local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    input_params = ", ".join(children)
    if param_list:
        input_params += f", {param_list}"
    code_line = f"{new_res} = {func_name}({input_params})"
    local_context.add_codeline(code_line)
    local_context.add_dependency(func_name, function)
    local_context.res_stack.append(new_res)
    return local_context

@rec_unroll.register
def unroll_svg_comb(expression: SVG_COMBINATORS, local_context: LocalContext, sketcher: Sketcher,
                  secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                  *args, **kwargs) -> LocalContext:
    function = COLOR_FUNCTIONS[type(expression)]
    tree_branches = [arg for arg in expression.args]
    func_name = expression.__class__.__name__
    n_children = len(tree_branches)
    cur_coords = local_context.coords_stack.pop()
    cur_transform = local_context.transform_stack.pop()
    for child in tree_branches:
        if not isolated_vars: 
            # we do a copy here of the transform and of the coords.
            local_context.transform_count += 1
            local_context.coords_count += 1
            new_transform = f"transform_{local_context.transform_count}"
            new_coords = f"coords_{local_context.coords_count}"
            code_line = f"{new_transform} = {cur_transform}.clone()"
            local_context.add_codeline(code_line)
            code_line = f"{new_coords} = {cur_coords}.clone()"
            local_context.add_codeline(code_line)
        local_context.coords_stack.append(cur_coords)
        assert isinstance(child, (gls.GLFunction, gls.GLExpr)), "Child must be a GLFunction or GLExpr"
        local_context = rec_unroll(child, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

    children = [local_context.res_stack.pop() for _ in range(n_children)]
    # reverse the children
    children = children[::-1]
    local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    input_params = ", ".join(children)
    code_line = f"{new_res} = {func_name}({input_params})"
    local_context.add_codeline(code_line)   
    local_context.add_dependency(func_name, function)
    local_context.res_stack.append(new_res)
    return local_context

@rec_unroll.register
def unroll_apply_color(expression: APPLY_COLOR_TYPE, local_context: LocalContext, sketcher: Sketcher,
                     secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                     relaxed_occupancy: bool = False, relax_temperature: float = 0.0,
                     *args, **kwargs) -> LocalContext:

    sdf_expr = expression.args[0]
    color = expression.args[1]
    func_name = expression.__class__.__name__
    function = COLOR_FUNCTIONS[type(expression)]
    if not color in expression.lookup_table:
        assert isinstance(color, sp.Symbol), "Color must be a symbol"
        color = COLOR_MAP[color.name].to(sketcher.device)
    else:
        assert isinstance(color, sp.Symbol), "Color must be a symbol"
        color = expression.lookup_table[color]
    params = f"{color}"

    local_context = rec_unroll(sdf_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
    cur_res = local_context.res_stack.pop()
    if isolated_vars: 
        local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    if relaxed_occupancy:
        code_line = f"{new_res} = _smoothen_sdf({cur_res}, {relax_temperature})"
        local_context.add_codeline(code_line)
        local_context.add_dependency("_smoothen_sdf", _smoothen_sdf)
    else:
        code_line = f"{new_res} = {cur_res} <= 0"
        local_context.add_codeline(code_line)
    if isolated_vars:
        local_context.res_count += 1
    new_colored_res = f"res_{local_context.res_count}"
    code_line = f"{new_colored_res} = {func_name}({new_res}, {params})"
    local_context.add_codeline(code_line)
    local_context.add_dependency(func_name, function)
    local_context.res_stack.append(new_colored_res)
    return local_context

def _smoothen_sdf(execution, temperature):
    output_tanh = th.tanh(execution * temperature)
    output_shape = th.nn.functional.sigmoid(-output_tanh * temperature)
    return output_shape


@rec_unroll.register
def unroll_color_mod(expression: COLOR_MOD, local_context: LocalContext, sketcher: Sketcher,
                   secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                   *args, **kwargs) -> LocalContext:
    color_expr = expression.args[0]
    colors = expression.args[1:]
    func_name = expression.__class__.__name__
    function = COLOR_FUNCTIONS[type(expression)]
    # Get the sdf_expr:
    eval_colors = []
    for color in colors:
        assert isinstance(color, sp.Symbol), "Color must be a symbol"
        if color in COLOR_MAP[color.name]:
            color = COLOR_MAP[color.name].to(sketcher.device)
        elif color in expression.lookup_table:
            color = expression.lookup_table[color]
        else:
            color = _parse_param_from_expr(color, [], sketcher)
        eval_colors.append(color)
    params = ", ".join(eval_colors)
    local_context = rec_unroll(color_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
    cur_res = local_context.res_stack.pop()
    if isolated_vars:
        local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    code_line = f"{new_res} = {func_name}({cur_res}, {params})"
    local_context.add_codeline(code_line)
    local_context.add_dependency(func_name, function)
    local_context.res_stack.append(new_res)
    return local_context

@rec_unroll.register
def unroll_unopt_alpha(expression: UNOPT_ALPHA, local_context: LocalContext, sketcher: Sketcher,
                     secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                     *args, **kwargs) -> LocalContext:
    raise NotImplementedError("Unopt alpha not implemented")


@rec_unroll.register
def unroll_gl_expr(expression: EXPR_TYPE, local_context: LocalContext, sketcher: Sketcher,
                 secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                 *args, **kwargs) -> LocalContext:
    evaluated_args = []
    # print("expr args", expr.args)
    for arg in expression.args:
        if isinstance(arg, (GLFunction, GLExpr)):
            local_context = rec_unroll(arg, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
            evaluated_args.append(local_context.res_stack.pop())
            # just use tensor symbols here
        elif isinstance(arg, (sp.Float, sp.Integer)):
            new_value = float(arg)
            evaluated_args.append(new_value)
        else:
            print(type(arg))
            raise NotImplementedError(f"Params for{expression} not implemented")
    op = SYMPY_TO_TORCH[expression.func]
    if isolated_vars:
        local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    code_line = f"{new_res} = {expression.func.__name__}({evaluated_args})"
    local_context.add_codeline(code_line)
    local_context.add_dependency(expression.func.__name__, op)
    local_context.res_stack.append(new_res)
    return local_context

@rec_unroll.register
def unroll_gl_param(expression: gls.Param, local_context: LocalContext, sketcher: Sketcher,
                 secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                 *args, **kwargs) -> LocalContext:
    assert isinstance(expression.args[0], sp.Symbol), "Argument must be a symbol"
    variable = expression.lookup_table[expression.args[0]]

    local_context.var_count += 1
    cur_color = f"var_{local_context.var_count} = {variable}"
    local_context.add_codeline(cur_color)
    return local_context

@rec_unroll.register
def unroll_gl_op(expression: gls.Operator, local_context: LocalContext, sketcher: Sketcher,
               secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
               *args, **kwargs) -> LocalContext:
    if isinstance(expression, (gls.UnaryOperator, gls.BinaryOperator)):
        
        args = expression.args[:-1]
        for arg in args:
            local_context = rec_unroll(arg, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)    
        n_evals = len(args)

        children = [local_context.res_stack.pop() for _ in range(n_evals)]
        children = ", ".join(children)
        # reverse the children
        children = children[::-1]
        op_arg = expression.args[-1]
        assert isinstance(op_arg, sp.Symbol), "Operator argument must be a string"
        func_name = TEXT_TO_SYMPY[op_arg.name]
        function = SYMPY_TO_TORCH[TEXT_TO_SYMPY[op_arg.name]]
        if isolated_vars:
            local_context.res_count += 1
        new_res = f"res_{local_context.res_count}"
        code_line = f"{new_res} = {func_name}({children})"
        local_context.add_codeline(code_line)
        local_context.add_dependency(func_name, function)
        local_context.res_stack.append(new_res)
        return local_context
    else:
        raise NotImplementedError(f"Vector Operator {expression} not implemented")

@rec_unroll.register
def unroll_gl_var(expression: gls.Variable, local_context: LocalContext, sketcher: Sketcher,
                secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
                *args, **kwargs) -> LocalContext:
    raise NotImplementedError(f"Variable {expression} not supported in GeoLIPI. It is supported in derivatives")
# Need to add eval ops for Ops. 

def _process_params(expression, params, local_context: LocalContext, sketcher: Sketcher):
    """
    Process variables for expression unrolling.
    
    Args:
        expression: The expression to process
        params: The parameters to process
        local_context: The local context containing variable counts
        sketcher: The sketcher instance
        inline_params: Whether to inline parameters or create variable names
        
    Returns:
        tuple: (func_name, processed_params)
    """
    param_list = _parse_param_from_expr(expression, params, sketcher)
    processed_params = ", ".join([str(x) for x in param_list])
    
    return processed_params
