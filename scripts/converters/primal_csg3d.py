from typing import List
import re
import numpy as np
import torch as th
from sympy import Symbol
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamSphere3D
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D

# Define the regular expression pattern (ChatGPT Help!)
PATTERN = r'(\w+|\$)(?:\(([-0-9.,\s]*)\))?|\$'

BOOL_MAP = {
    'union': Union,
    'intersection': Intersection,
    'difference': Difference
}
PRIM_MAP = {
    'cuboid': NoParamCuboid3D,
    'sphere': NoParamSphere3D,
}

def str_to_expr(expression_string_list: List[str], to_cuda=False, dtype=th.float32):
    """Parse a list of expression strings into a geolipi expression.
    """
    expr_stack = []

    for expression_string in expression_string_list[::-1]:
        match = re.match(PATTERN, expression_string)
        cmd_name = match.group(1)
        if cmd_name in BOOL_MAP.keys():
            expr_1 = expr_stack.pop()
            expr_2 = expr_stack.pop()
            cmd = BOOL_MAP[cmd_name](expr_1, expr_2)
            expr_stack.append(cmd)
        elif cmd_name in PRIM_MAP.keys():
            if to_cuda:
                params = th.tensor([float(x.strip()) for x in match.group(2).split(',')],
                                   dtype=dtype, device='cuda')
            else:
                params = np.array([float(x.strip()) for x in match.group(2).split(',')])
            
            translate_param = params[:3]
            scale_param = params[3:]
            cmd = Translate3D(Scale3D(PRIM_MAP[cmd_name](), scale_param),
                            translate_param)
            expr_stack.append(cmd)
        elif cmd_name == '$':
            continue
        else:
            raise ValueError(f'Unknown command {expression_string}')


    assert len(expr_stack) == 1, 'Error! Stack should have only one element'
    expression = expr_stack[0]
    return expression