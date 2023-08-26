from geolipi.symbolic import  *
from geolipi.languages.primal_csg import str_to_expr
from geolipi.torch_sdf.sketcher import Sketcher
from geolipi.torch_sdf.evaluate import expr_to_sdf, compile_expr, expr_prim_count, expr_to_dnf
from geolipi.torch_sdf.evaluate import create_evaluation_batches, batch_evaluate

demo_file = "/media/aditya/DATA/data/synthetic_data/PCSG3D_data/synthetic/four_ops/expressions.txt"

expressions = open(demo_file, 'r').readlines()
expressions = [x.strip().split("__") for x in expressions]

sketcher = Sketcher(device="cuda", resolution=64)
stack = []
## Data loading
for expression in expressions[:100]:
    parsed_expr, var_dict = str_to_expr(expression, to_cuda=True)

    prim_count = expr_prim_count(parsed_expr)

    compiled_expr = compile_expr(parsed_expr, var_dict, prim_count, sketcher=sketcher, rectify_transform=True)
    expr = compiled_expr[0]
    resolved_expr = expr_to_dnf(expr)
    transforms_in_numpy = {x:y.detach().cpu().numpy() for x, y in compiled_expr[1].items()}
    inversions_in_numpy = {x:y.detach().cpu().numpy() for x, y in compiled_expr[2].items()}
    params_in_numpy = {x:y.detach().cpu().numpy() for x, y in compiled_expr[3].items()}
    stack.append([resolved_expr, transforms_in_numpy, inversions_in_numpy, params_in_numpy])

# Hypothetical execution
expr_set = create_evaluation_batches(stack, convert_to_cuda=True)
all_sdf = batch_evaluate(expr_set, sketcher)

print("done!")