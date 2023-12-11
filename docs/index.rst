.. GeoLIPI documentation master file, created by
   sphinx-quickstart on Mon Dec 11 09:43:07 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GeoLIPI: DSL for implicit geometric modelling
===============================================================

GeoLIPI is a language for modelling 3D shapes. This is to be treated as a meta-language, from which visual programming languages can be derived. Some of the languages/visual programs that can be executed in this framework are:

1) CSG 3D Variants
2) GeoCode
3) SVG 2D

and many more. Check out `languages.md` for more details.

Main usecase
============

Mainly, GeoLIPI attempts to embed a generic visual language in python, making it easier to use and extend for research. Additionally, it provides the following benefits:

1) Fast Batched Execution of programs - useful for training neural networks on large batches of program executions.
2) Single "symbolic" object, multiple execution strategies. This helps with "executing" the program in different platforms/systems (such as blender for rendering, and pytorch for optimization).
3) Parameter Optimization of arbitrary visual programs (All operations are created to allow differentiable optimization of its parameters).
4) [TBD] Help with searching programs for a desired execution (refer to our recent [paper](https://bardofcodes.github.io/coref/)).
5) Batched PyTorch execution code for all the shader toy 2D and 3D primitives described by Inigo Quilez.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contents
--------

.. toctree::
    intro
    install
    languages
    modules
