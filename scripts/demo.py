from geolipi.symbolic import  *
# from geolipi.languages.primal_csg3d import str_to_expr
from geolipi.languages.macro_csg3d import str_to_expr
# from geolipi.languages.cp_fusion import str_to_expr

from geolipi.symbolic.utils import resolve_macros
import numpy as np
import torch as th
from sympy import Symbol
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.evaluate_sdf import expr_to_sdf
from geolipi.torch_compute.evaluate_color import expr_to_colored_canvas
from geolipi.torch_compute.compile_expression import compile_expr, expr_prim_count, expr_to_dnf
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate

import _pickle as cPickle
demo_file = "/media/aditya/DATA/data/synthetic_data/MCSG3D_data/synthetic/three_ops/expressions.txt"
# demo_file = "/home/aditya/projects/edit_vpi/ProgFixer/executors/data/toy_fix_train.txt"
# np_var = np.random.uniform(size=(1))
# variable = th.from_numpy(np_var).float().cuda()

# expr = Scale2D(TriangleEquilateral2D(variable), th.tensor([0.5, 0.5], dtype=th.float32).cuda())
# expr = ColorTree2D(TranslationSymmetry2D(expr, th.tensor([0.2, 0.2], dtype=th.float32).cuda(), 3), Symbol("GREEN"))
sketcher = Sketcher(device="cuda", resolution=64, n_dims=3)

# sdf = expr_to_colored_canvas(expr, sketcher=sketcher, rectify_transform=True)
# sdf = sdf.cpu().numpy()

expressions = open(demo_file, 'r').readlines()
expressions = [x.strip().split("__") for x in expressions]
# expressions = [x.strip().split(" ") for x in expressions]

stack = []
## Data loading
for expression in expressions[2:3]:
    parsed_expr = str_to_expr(expression, to_cuda=True)
    parsed_expr = resolve_macros(parsed_expr, device="cuda")
    # sdf = expr_to_colored_canvas(parsed_expr, sketcher=sketcher, rectify_transform=True)
    
    prim_count = expr_prim_count(parsed_expr)

    compiled_expr = compile_expr(parsed_expr, prim_count, sketcher=sketcher, rectify_transform=True)
    expr = compiled_expr[0]
    resolved_expr = expr_to_dnf(expr)
    transforms_in_numpy = {x:y.detach().cpu() for x, y in compiled_expr[1].items()}
    inversions_in_numpy = {x:y.detach().cpu() for x, y in compiled_expr[2].items()}
    params_in_numpy = {x:y.detach().cpu() for x, y in compiled_expr[3].items()}
    stack.append([resolved_expr, transforms_in_numpy, inversions_in_numpy, params_in_numpy])

# Hypothetical execution
expr_set = create_evaluation_batches(stack, convert_to_cuda=True)
all_sdf = batch_evaluate(expr_set, sketcher)

print("done!")