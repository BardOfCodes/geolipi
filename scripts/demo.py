import sys
sys.path.append('../')
import geolipi.symbolic as gls
# from geolipi.languages.primal_csg3d import str_to_expr
# from geolipi.languages.macro_csg3d import str_to_expr
# from geolipi.languages.cp_fusion import str_to_expr

from geolipi.symbolic.resolve import resolve_macros
import numpy as np
import torch as th
from sympy import Symbol
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.evaluate_expression import expr_to_sdf, recursive_evaluate
from geolipi.torch_compute.compile_expression import compile_expr, expr_prim_count, expr_to_dnf
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate
from geolipi.torch_compute.visualizer import get_figure

import plotly.io as pio
pio.renderers
pio.renderers.default = "browser"
x = th.tensor([0.2], requires_grad=False)
# expression = gls.Cuboid3D((0.1, 1.2, 0.4))
device = th.device("cuda")
x = th.tensor([0.1, 0.2], device=device, requires_grad=True)
y = th.tensor([0.2], device=device, requires_grad=True)
expr = gls.Rectangle2D(x)
expr = gls.Twist3D(expr, y)

# Now also render it
from geolipi.torch_compute.sphere_marcher import Renderer

sketcher = Sketcher(resolution=(64), device="cuda")
sketcher_2d = Sketcher(resolution=(64), device="cuda", n_dims=2)
sdf = recursive_evaluate(expr, sketcher, sketcher_2d)

resolution = (512, 1024)
renderer = Renderer(resolution=resolution, recursive_evaluator=False,
                    )

image = renderer.render(expr)

print("Done")
# import matplotlib.pyplot as plt
# plt.imshow(image)