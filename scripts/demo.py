from geolipi.symbolic import  *
# from geolipi.languages.primal_csg3d import str_to_expr
# from geolipi.languages.macro_csg3d import str_to_expr
# from geolipi.languages.cp_fusion import str_to_expr

from geolipi.symbolic.resolve import resolve_macros
import numpy as np
import torch as th
from sympy import Symbol
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.evaluate_sdf import expr_to_sdf, recursive_expr_to_sdf
from geolipi.torch_compute.evaluate_color import expr_to_colored_canvas
from geolipi.torch_compute.compile_expression import compile_expr, expr_prim_count, expr_to_dnf
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate


import _pickle as cPickle

sketcher = Sketcher(device="cuda", resolution=64, n_dims=3)
sketcher_2d = Sketcher(device="cuda", resolution=64, n_dims=2)
# expressions = open(demo_file, 'r').readlines()
# expressions = [x.strip().split("__") for x in expressions]

expression = Union(SmoothUnion(Translate3D(
                            Bend3D(NoParamCuboid3D(), 
                                     th.tensor([0.1], requires_grad=True, device="cuda")),
                            th.tensor([0.0, -0.6, 0.0], device="cuda")),
                         Translate3D(Scale3D(NoParamCuboid3D(), 
                                             th.tensor([0.1, 0.6, 0.1], device="cuda")),
                                 th.tensor([-0.1, -0.6, 0.0], device="cuda")),
                     th.tensor([0.1], device="cuda")),
                   QuadraticBezierExtrude3D(NoParamRectangle2D(), 
                                 th.tensor([0.1, 0.3, 0.5], device="cuda"),
                                 th.tensor([0.0, 0.0, 0.0], device="cuda"),
                                 th.tensor([0.0, 0.3, 0.0], device="cuda"),
                                 th.tensor([0.5], device="cuda"))
)


output = recursive_expr_to_sdf(expression, sketcher, secondary_sketcher=sketcher_2d, rectify_transform=True)

print("Done!")
# stack = []
# ## Data loading
# for expression in expressions[2:3]:
#     parsed_expr = str_to_expr(expression, to_cuda=True)
#     parsed_expr = resolve_macros(parsed_expr, device="cuda")
#     # sdf = expr_to_colored_canvas(parsed_expr, sketcher=sketcher, rectify_transform=True)
    
#     prim_count = expr_prim_count(parsed_expr)

#     compiled_expr = compile_expr(parsed_expr, prim_count, sketcher=sketcher, rectify_transform=True)
#     expr = compiled_expr[0]
#     resolved_expr = expr_to_dnf(expr)
#     transforms_in_numpy = {x:y.detach().cpu() for x, y in compiled_expr[1].items()}
#     inversions_in_numpy = {x:y.detach().cpu() for x, y in compiled_expr[2].items()}
#     params_in_numpy = {x:y.detach().cpu() for x, y in compiled_expr[3].items()}
#     stack.append([resolved_expr, transforms_in_numpy, inversions_in_numpy, params_in_numpy])

# # Hypothetical execution
# expr_set = create_evaluation_batches(stack, convert_to_cuda=True)
# all_sdf = batch_evaluate(expr_set, sketcher)
