import sys

import numpy as np
import torch as th
import matplotlib.pyplot as plt

import geolipi.symbolic as gls
from geolipi.torch_compute.sphere_marcher import Renderer

dtype = th.float32
device = th.device("cuda")
resolution = (1024, 1024)

# Just use the default settings in the renderer
renderer = Renderer(resolution=resolution, device=device, dtype=dtype)

# Group them in a hierarchy of unions
# longer expressions
import time
expr = gls.RotationSymmetryY3D(
            gls.Translate3D(
                gls.Union(
                    gls.RoundedBox3D((0.1, 0.5, 0.1), (0.05,)),
                    gls.EulerRotate3D(gls.Torus3D((0.5, 0.1)), (np.pi/2, 0, 0))),
                (1.5, 0., 1.5)),
            np.pi/4, 8)
expr = expr.tensor().cuda()
print(expr.pretty_print())
# use compile_expression for faster rendering
renderer = Renderer(resolution=resolution, device=device, dtype=dtype, 
                    recursive_evaluator=False, compile_expression=True)

st = time.time()
image = renderer.render(expr)
et = time.time()
print("Time taken with compilation: ", et - st)

plt.figure(figsize=(10, 5))
plt.imshow(image.detach().cpu().numpy())
plt.axis('off')
