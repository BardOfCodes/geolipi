
from sqlite3 import paramstyle
import sys
import sympy as sp
import ast
import torch as th
from typing import Optional, Union, Callable, Any, TypeVar, List, Tuple, Dict
from collections import defaultdict

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
    HIGERPRIM_TYPE,
    COLOR_MOD,
    APPLY_COLOR_TYPE,
    SVG_COMBINATORS,
    UNOPT_ALPHA,
    EXPR_TYPE,
    SUPERSET_TYPE
)
from .sketcher import Sketcher
from .maps import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP, COLOR_FUNCTIONS, COLOR_MAP
from .common import EPSILON
from .sympy_to_torch import SYMPY_TO_TORCH, TEXT_TO_SYMPY
from .evaluate_expression import _parse_param_from_expr
from .unroll_expression import _process_params, LocalContext
"""
Unroll the expression into a list of codelines. 
inline_params: bool -> else we store the params as var_x - but that is useless. 
"""

class CompiledLocalContext(LocalContext):
    def __init__(self):
        self.transform_stack = ["transform_0"]
        self.coords_stack = ["coords_0"]
        self.res_stack = []
        self.var_count = 0
        self.transform_count = 0
        self.coords_count = 0
        self.res_count = 0
        self.prim_dict = {}
        self.dependencies = dict()
        self.transform_codelines = []
        self.prim_codelines = []
        self.codelines = []


    def add_codeline(self, codeline: str, codeline_type: str):
        if codeline_type == "transform":
            self.transform_codelines.append(codeline)
        elif codeline_type == "prim":
            self.prim_codelines.append(codeline)
        else:
            raise NotImplementedError(f"Codeline type {codeline_type} not implemented")

    def add_dependency(self, func_name: str, func: Any):
        self.dependencies[func_name] = func
    
    def add_prim_ref(self, prim_name: str, prim_type: Tuple[type, str, str]):
        self.prim_dict[prim_name] = prim_type
    
    def compile_codelines(self, sketcher: Sketcher):
        # Generate the transform lines. 
        for line in self.transform_codelines:
            self.codelines.append(line)
        
        # gather the transforms based on prim_dict.
        selected_transforms = []
        prim_type_to_transform = defaultdict(list)
        prim_type_to_params = defaultdict(list)
        prim_type_to_name = defaultdict(list)
        for ind, (prim_name, prim_info) in enumerate(self.prim_dict.items()):
            prim_type, transform, params = prim_info
            selected_transforms.append(transform)
            prim_type_to_transform[prim_type].append(ind)
            prim_type_to_params[prim_type].append(params)
            prim_type_to_name[prim_type].append(prim_name)
        # do th.stack(selected_transforms)
        # and then do th.einsum('ij,mj->mi', transform, coords)
        # Here we can reorder so that we can use slicing better. 

        n_dims = sketcher.n_dims
        n_transforms = len(selected_transforms)
        selected_transforms = ", ".join(selected_transforms)
        code_line = f"stacked_transform = th.stack([{selected_transforms}], dim=0)"
        self.codelines.append(code_line)
        code_line = f"stacked_coords = coords_0.unsqueeze(0).expand({n_transforms}, -1, -1)"
        self.codelines.append(code_line)
        code_line = f"stacked_coords = th.matmul(stacked_coords, stacked_transform.transpose(-1, -2))"
        self.codelines.append(code_line)
        code_line = f"stacked_coords = stacked_coords[..., :{n_dims}] / (stacked_coords[..., {n_dims} : {n_dims + 1}] + {float(EPSILON)})"
        self.codelines.append(code_line)


        # then do the grouped processing of the primitives.
        for ind, (prim_type, prim_inds) in enumerate(prim_type_to_transform.items()):
            params = prim_type_to_params[prim_type]
            code_line = f"stacked_coords_{ind} = stacked_coords[{prim_inds}]"
            self.codelines.append(code_line)
            first_param = params[0]
            if len(first_param) == 0:
                code_line = f"stacked_res_{ind} = {prim_type.__name__}(stacked_coords_{ind})"
                self.codelines.append(code_line)
            else:
                split_params = [param.split(",") for param in params]
                n_params = len(split_params[0])
                stacked_params = []
                for param_ind in range(n_params):
                    sel_param_names = [x[param_ind] for x in split_params]
                    sel_param_names = ", ".join(sel_param_names)
                    # stack it
                    code_line = f"stacked_params_{ind}_{param_ind} = th.stack([{sel_param_names}], dim=0)"
                    self.codelines.append(code_line)
                    stacked_params.append(f"stacked_params_{ind}_{param_ind}")

                stacked_params = ", ".join(stacked_params)
                code_line = f"stacked_res_{ind} = {prim_type.__name__}(stacked_coords_{ind}, {stacked_params})"
                self.codelines.append(code_line)
            for prim_ind, prim_name in enumerate(prim_type_to_name[prim_type]):
                code_line = f"{prim_name} = stacked_res_{ind}[{prim_ind}]"
                self.codelines.append(code_line)

            
        # finally prim_codelines.
        for line in self.prim_codelines:
            self.codelines.append(line)


### Create a Evaluate wrapper -> This will create the coords and may be different in different derivative languages.

def compile_expression(in_expr: GLBase, sketcher: Sketcher, 
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
    local_context = CompiledLocalContext()
    local_context.dependencies['th'] = th
    local_context.dependencies['transform_0'] = sketcher.get_affine_identity()

    local_context = rec_compiled_unroll(
        expression,
        local_context=local_context,
        sketcher=sketcher,
        secondary_sketcher=secondary_sketcher, isolated_vars=isolated_vars,
        *args, **kwargs
    )
    local_context.compile_codelines(sketcher)

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
        assoc_lines = []
        for ind, param in enumerate(param_names):
            assoc_lines.append(f"{param} = varlist[{ind}]")
        local_context.codelines = assoc_lines + local_context.codelines
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
def rec_compiled_unroll(expression: SUPERSET_TYPE, local_context: CompiledLocalContext, sketcher: Sketcher,
             secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
             *args, **kwargs) -> CompiledLocalContext:
    raise NotImplementedError(
        f"Expression type {type(expression)} is not supported for recursive evaluation."
    )

@rec_compiled_unroll.register
def compiled_unroll_mod(expression: MOD_TYPE, local_context: CompiledLocalContext, sketcher: Sketcher, 
             secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
             *args, **kwargs) -> CompiledLocalContext:

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
        local_context.add_codeline(code_line, "transform")
        code_line = f"{new_transform} = th.matmul({new_transform}, {cur_transform})"
        local_context.add_codeline(code_line, "transform")
        local_context.add_dependency(func_name, function)
        local_context.add_dependency("IDENTITY", sketcher.get_affine_identity())

        local_context.transform_stack.append(new_transform)
        assert isinstance(sub_expr, (gls.GLFunction, gls.GLExpr)), "Sub expression must be a GLFunction or GLExpr"
        return rec_compiled_unroll(sub_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
    elif isinstance(expression, POSITIONALMOD_TYPE):
        # instantiate positions and send that as input with affine set to None
        cur_coords = local_context.coords_stack.pop()
        # Maybe we only need to add when we are duplicating, i.e. Union etc. otherwise no.
        if isolated_vars:
            local_context.coords_count += 1
        new_coords = f"coords_{local_context.coords_count}"
        code_line = f"{new_coords} = {func_name}({cur_coords}, {params})"
        local_context.add_codeline(code_line, "transform")
        local_context.add_dependency(func_name, function)
        local_context.coords_stack.append(new_coords)
        assert isinstance(sub_expr, (gls.GLFunction, gls.GLExpr)), "Sub expression must be a GLFunction or GLExpr"
        return rec_compiled_unroll(sub_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

    elif isinstance(expression, SDFMOD_TYPE):
        # calculate sdf then create change before returning.
        local_context = rec_compiled_unroll(sub_expr, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)
        cur_res = local_context.res_stack.pop()
        if isolated_vars:
            local_context.res_count += 1
        new_res = f"res_{local_context.res_count}"
        code_line = f"{new_res} = {func_name}({cur_res}, {params})"
        local_context.add_codeline(code_line, "prim")
        local_context.add_dependency(func_name, function)
        local_context.res_stack.append(new_res)
        return local_context
    else:
        raise NotImplementedError(f"Modifier {expression} not implemented")
    
@rec_compiled_unroll.register
def compiled_unroll_prim(expression: PRIM_TYPE, local_context: CompiledLocalContext, sketcher: Sketcher,
              secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
              *args, **kwargs) -> CompiledLocalContext:
    
    if isinstance(expression, HIGERPRIM_TYPE):
        params = expression.args[1:]
    else:
        params = expression.args

    func_name = expression.__class__.__name__
    params = _process_params(expression, params, local_context, sketcher)
    # This is for correction? 
    n_dims = sketcher.n_dims
    function = PRIMITIVE_MAP[type(expression)]
    if isinstance(expression, HIGERPRIM_TYPE):
        raise NotImplementedError("Higher-order primitives not implemented")
    else:
        cur_transform = local_context.transform_stack.pop()
        cur_coords = local_context.coords_stack.pop()
        if isolated_vars:
            local_context.coords_count += 1
        # Keeping track of coords and transforms is a bit confusing.
        # new_coords = f"coords_{local_context.coords_count}"
        # code_line = f"{new_coords} = th.einsum('ij,mj->mi', {cur_transform}, {cur_coords})"
        # instead just associate transform prim. and index.
        # local_context.add_codeline(code_line)
        # code_line = f"{new_coords} = {new_coords}[..., :{n_dims}] / ({new_coords}[..., {n_dims} : {n_dims + 1}] + {float(EPSILON)})"
        # local_context.add_codeline(code_line)
        local_context.res_count += 1
        new_res = f"res_{local_context.res_count}"
        # type, transform, params
        local_context.add_prim_ref(new_res, (expression.__class__, cur_transform, params))
        # code_line = f"{new_res} = {func_name}({new_coords}, {params});"
        # local_context.add_codeline(code_line)
        local_context.add_dependency(func_name, function)
        local_context.res_stack.append(new_res)
        return local_context


@rec_compiled_unroll.register
def compiled_unroll_comb(expression: COMBINATOR_TYPE, local_context: CompiledLocalContext, sketcher: Sketcher,
              secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
              *args, **kwargs) -> CompiledLocalContext:
    
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
            local_context.add_codeline(code_line, "transform")
            code_line = f"{new_coords} = {cur_coords}.clone()"
            local_context.add_codeline(code_line, "transform")
            local_context.transform_stack.append(new_transform)
            local_context.coords_stack.append(new_coords)
        else:
            local_context.transform_stack.append(cur_transform)
            local_context.coords_stack.append(cur_coords)
        local_context.coords_stack.append(cur_coords)
        assert isinstance(child, (gls.GLFunction, gls.GLExpr)), "Child must be a GLFunction or GLExpr"
        local_context = rec_compiled_unroll(child, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

    children = [local_context.res_stack.pop() for _ in range(n_children)]
    # reverse the children
    children = children[::-1]
    local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    input_params = ", ".join(children)
    if param_list:
        input_params += f", {param_list}"
    code_line = f"{new_res} = {func_name}({input_params})"
    local_context.add_codeline(code_line, "prim")
    local_context.add_dependency(func_name, function)
    local_context.res_stack.append(new_res)
    return local_context
