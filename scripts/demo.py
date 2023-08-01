from geolipi.symbolic import  *
from geolipi.languages.plad_csg import parser
from geolipi.sdf.executor import execute, Sketch3D

demo_file = "/media/aditya/DATA/data/synthetic_data/PCSG3D_data/synthetic/four_ops/expressions.txt"

expressions = open(demo_file, 'r').readlines()
expressions = [x.strip().split("__") for x in expressions]
expression = expressions[0]

parsed_expr = parser(expression)
sketcher = Sketch3D(device="cuda", resolution=64)
sdf = execute(parsed_expr, sketcher=sketcher, mode='naive')




expression = Cuboid(1, 2, 3)