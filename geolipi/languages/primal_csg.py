from typing import List
import re
import numpy as np
import torch as th
from sympy import Symbol
from geolipi.symbolic import Primitive3D, Combinator
from geolipi.symbolic.primitives_3d import NoParamCuboid, NoParamSphere
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.transforms import Translate, Scale

# Define the regular expression pattern (ChatGPT Help!)
PATTERN = r'(\w+|\$)(?:\(([-0-9.,\s]*)\))?|\$'

BOOL_MAP = {
    'union': Union,
    'intersection': Intersection,
    'difference': Difference
}
PRIM_MAP = {
    'cuboid': NoParamCuboid,
    'sphere': NoParamSphere,
}

def str_to_expr(expression_string_list: List[str], to_cuda=False, dtype=th.float32):
    """Parse a list of expression strings into a geolipi expression.
    """
    expr_stack = []
    operator_stack = []
    stack_pointer = []
    symbol_lookup = {}
    var_count = 0

    for expression_string in expression_string_list:
        match = re.match(PATTERN, expression_string)
        cmd_name = match.group(1)
        if cmd_name in BOOL_MAP.keys():
            operator_stack.append(BOOL_MAP[cmd_name])
            stack_pointer.append(len(expr_stack))
        elif cmd_name in PRIM_MAP.keys():
            if to_cuda:
                params = th.tensor([float(x.strip()) for x in match.group(2).split(',')],
                                   dtype=dtype, device='cuda')
            else:
                params = np.array([float(x.strip()) for x in match.group(2).split(',')])
            
            scale_param = params[3:]
            translate_param = params[:3]
            scale_var = Symbol(f'scale_{var_count}')
            translate_var = Symbol(f'translate_{var_count}')
            var_count += 1
            symbol_lookup[scale_var.name] = scale_param
            symbol_lookup[translate_var.name] = translate_param
            cmd = Translate(Scale(PRIM_MAP[cmd_name](), scale_var),
                            translate_var)
            expr_stack.append(cmd)
        else:
            raise ValueError(f'Unknown command {expression_string}')

        while (operator_stack and len(expr_stack)- stack_pointer[-1] >= 2):
            cur_operator = operator_stack.pop()
            expr_2 = expr_stack.pop()
            expr_1 = expr_stack.pop()
            _ = stack_pointer.pop()
            new_expr = cur_operator(expr_1, expr_2)
            expr_stack.append(new_expr)

    assert len(expr_stack) == 1, 'Error! Stack should have only one element'
    expression = expr_stack[0]
    return expression, symbol_lookup