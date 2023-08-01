# GeoLipi

Language for modelling 3D shapes. This is to be treated as a meta-language, from which certain specific 3D Visual programming languages can be derived. Targets:

1) 3D CSG
2) ShapeAssembly
3) GeoCode

## Next steps

1) Get P-CSG and M-CSG working.
2) Get Batch executors integrated.
3) Create data-loader examples.

## Goal

Language or python -> Symbol soup -> compiled form  -> SDF eval, or blender graph.

1) Fast Batched execution with baking.
2) Blender graph creation + visualization
3) Expression to Mesh, PC, Occupancy Grid, SDF Grid.

### Primitives

1) Basic: Cuboid, Ellipse, Cylinder, Cone,
2) Infinite: HalfSpace, Inf. Cy/cone
3) Superquadrics
4) Gaussian Occupancy: SIF and spaghetti
5) More complicated primitives: torus, link, hex prism, triangle prism, capsule, pyramid, octahedron
6) sphere-triangles - Sphere + box triangle.
7) meta-balls - rounded output (in sdf land)
8) 2D/3D extrude with:
   1) Straight line (start and end points)
   2) 2D Curve Quad and cubic line (Iq's solution)
   3) 2D closed loop beizier (extrudeNet)
9) 3D Surfaces -> IQ's triangle and quad,
   1) ParseNet's diff nurbs
   2) Iq's Triangle and quads (is it volumetric?)
10) Neural Primitives
11) Colored primitives? (density and color) [Suitable for lego]

## Combinators

1) fusion: Union Only
2) Base CSG: Union, Intersection, Difference
3) Base +: Translate, Reflect, rotational sym
4) Base +: smooth union, intersection, difference
5) Infinite translate -> Mod
6) CADDY: map and fold operations -> out of scope. Szalinski
7) Attachment - combinators
8) Sub-programs - Shape Assembly

## Example

```python
from geolipi.primitives import *
from geolipi.combinators import *
from geolipi.functions import *

@geolipi
def shape_expr():
    x_translate = (5, 0, 0)
    cube_1 = Cuboid(1, 2, 3)
    cube_2 = Cuboid(2, 0.4, 1)
    shape = Union(cube_1, Translate(cube_2, x_translate))
    return shape

sdf = geolipi.sdf_executor.execute(shape_expr)
bpy_graph = geolipi.bpy_graph_generator.generate(shape_expr)
expr_string = geolipi.string_parser.generate_string(shape_expr)


@geolipi
def shape_expr():
    shape = Union(Cuboid(0.1, 0.2, 0.3), 
                  Translate(Cuboid(0.4, 0.6, 0.2), 
                            (0.3, 0.5, 0)
                  )
    )

    return shape
```
