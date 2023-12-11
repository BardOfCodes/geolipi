# GeoLIPI: A DSL for implicit geometric modelling

[banner]()

Language for modelling 3D shapes. This is to be treated as a meta-language, from which visual programming languages can be derived. Some of the languages/visual programs that can be executed in this framework are:

1) CSG 3D Variants
2) GeoCode
3) SVG 2D

and many more. Check out `languages.md` for more details. 

## Important: Research Code - Use at your own risk

I have added some documentation (with the help of ChatGPT) [here]().


## Main usecase

Mainly, GeoLIPI attempts to embed a generic visual language in python, making it easier to use and extend for research. Additionally, it provides the following benefits:

1) Fast Batched Execution of programs - useful for training neural networks on large batches of program executions. See a demonstration of this in `notebooks/compiled_execution.ipynb`.

2) Single "symbolic" object, multiple execution strategies. This helps with "executing" the program in different platforms/systems (such as blender for rendering, and pytorch for optimization). See `scripts/blender_render_example.py`.

3) Parameter Optimization of arbitrary visual programs (All operations are created to allow differentiable optimization of its parameters). See `notebooks/parameter_optimization.ipynb`.
4) [TBD] Help with searching programs for a desired execution (refer to our recent [paper]()).
5) Batched PyTorch execution code for all the shader toy 2D and 3D primitives described by Inigo Quilez. See `notebooks/torch_sdf2d_example.ipynb` and `notebooks/torch_sdf3d_example.ipynb`.

## Installation

```bash
```

## Examples

Check out the python notebooks in `notebooks/` for more details.

## Next steps

1) DearPyGui based visualizer for 2D 3D both.
2) Add Marching Primitives / Layer wise SVG optimization.
3) Add SDS optimization example -> Connecting it to natural language directly.
4) Add differentiable CSG operation and draw operations (probability over types).
5) Stochastic primitives?

## High level issues

1) Execution time. Something like a BVH is a must for *really* complex programs.

2) Initialization time - Sympy Functions are not the simplest to initialize. Furthermore, the lookup_table for tensors is not the most efficient.

3) What to do about numerical precision? Also SDFs are almost often not exact (after boolean operations or scaling etc.)

4) Aliasing - If we want beautiful output images, we need to do something about aliasing.

5) Which symbols should have 2D/3D explicit and which ones not? The code can be made more uniform in general.

## Other related works

Many other awesome libraries exist which help with pythonic 3D modelling. I have taken inspiration from many of them. Some of them are:

1) [Geometry script](https://github.com/carson-katri/geometry-script).

2) [openPySCAD](https://github.com/taxpon/openpyscad).

3) [sdf-torch](https://github.com/unixpickle/sdf-torch).

## Acknowledgements

1) A big shoutout to Inigo Quilez for his awesome [work](https://www.iquilezles.org/www/index.htm) on SDFs and shader toy.
2) Thanks to Carson Katri for his [geometry script](https://github.com/carson-katri/geometry-script) work which helped with the blender side of things.
3) [Derek](https://www.dgp.toronto.edu/~hsuehtil/)'s Blender toolbox ([link](https://github.com/HTDerekLiu/BlenderToolbox)) was quite helpful for materials.
4) Patrick-Kidger's [sympytorch](https://github.com/patrick-kidger/sympytorch) helped thinking about how to integrate sympy here.
5) Thanks to [Tim Nelson](https://cs.brown.edu/~tbn/)'s Logic for Systems course made the DNF/CNF stuff much easier to understand.
6) Thanks to my PhD Advisor [Daniel Ritchie](https://dritchie.github.io/) for his support and guidance.
7) Thanks to my lab mates [Arman Maesumi](https://armanmaesumi.github.io/) and [R. Kenny Jones](https://rkjones4.github.io/) for their feedback.
8) [Hiroki Sakuma](https://hirokisakuma.com/)'s [Torch-Sphere-Tracer](https://github.com/skmhrk1209/Torch-Sphere-Tracer) helped write my tiny dirty sphere tracer.
