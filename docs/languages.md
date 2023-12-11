# Visual Languages in GeoLIPI

GeoLIPI is meant to be a super-set of many primitive composition based languages. Here, we show the mapping between existing part-based shape languages used recently in research and GeoLIPI expressions. This is not meant to be a complete list of all languages, although the repository does target to support as many as possible.

## 2D Languages

1) CSG 2D- used in [1], and [2]. This is a simple language that supports boolean operations between simple shapes. Examples are shown in `notebooks/csg2d_language.ipynb`.
    1) GeoLIPI supports macros such as `Translation Symmetry` and `Rotation Symmetry`.
    2) GeoLIPI supports all the 43 2D sdf functions described by the amazing Inigo Quilez [here](https://iquilezles.org/articles/distfunctions2d/). All the functions and their executions can be found in `notebooks/torch_sdf2D_examples.ipynb`.
    3) Smooth combinators (`SmoothUnion` etc.), and other operators (`Onion` etc.) are also supported.

2) Curves 2D - Some papers simply model the visual data as a sequenece of "strokes" or linear/bezier curves, for example [3], and [4]. With GeoLIPI, we can use `gls.Segment2D` and `gls.QuadraticBezier2D` to represent these curves. Examples are shown in `notebooks/csg2d_language.ipynb`.

3) SVG 2D - Many papers directly model curves with color [5], or even patterns [6]. In GeoLIPI, we can use all the sdf shape functions, and the curve elements, with color operators to represent these patterns. Examples are shown in `notebooks/svg_2d_language.ipynb`. GeoLIPI also models 8 different ways to combine two SVG layers, as specified [here](https://ciechanow.ski/alpha-compositing/) (Thanks to Bartosz Ciechanowski for the amazing overview).

4) Advanced SVG 2D/CSG 2D: Since GeoLIPI is embedded in python, we can create expressions pythonically, with for loops, and if statements, and even stochastically sampling parameters (which can also be samples from an optimizable parameterized distribution / NN). An example of using for loops is in `notebooks/csg2d_language.ipynb`, and an example of stochastically generating a program is in `notebooks/svg_2d_language.ipynb`.

## 3D Languages

Since for 3D languages, constructing examples by hand can be hard, we show many examples after optimization. Please read the `parameter_optimization.md` file for details on the optimization process. Also, please take a look at the `rendering.md` file for details on how to render the shapes.

1) Primitive Fusion: Many works [7], [8], [9] model 3D shapes as a composition of cubic primitives. Some other approaches use a different (yet single type) of primitive, such as rounded-cuboids[10], or SuperQuadrics [11]. GeoLIPI supports all these primitives, and their fusion (i.e. Union) operations. Examples are shown in `notebooks/primitive_fusion_3d_examples.ipynb`. Note that modelling hierarchies is quite simple - by using multiple `Union` operators we can essentially construct a multi-level tree which respects the operation hierarchy which can be a proxy for the semantic hierarchy.

2) CSG 3D: Modelling shapes with CSG operations has been a very popular choice. However, there are many many ways to do this. Here is what GeoLIPI offers:
    1) Over 30 simple 3D primitive functions from `gls.Cuboid3D` to `gls.Octahedron3D` (Thanks to Inigo Quilez's examples [here](https://iquilezles.org/articles/distfunctions/)). These along with CSG operation can be used to model many works [12], [13], [14], [15] etc.
    2) Extrusion and Revolution based primitives. This allows GeoLIPI to (almost) model works such as [16] and [17].
    3) Macros such as `Translation Symmetry` and `Rotation Symmetry`, which can be used to model works such as [18]. ([18] can be modelled in python directly with GeoLIPI as well)
    4) Additional 3D operations - Smooth Combinators (`SmoothUnion` etc.), and other operators (`Twist`, `Dilate` etc.).
    Examples are show in `notebooks/csg_3d_language.ipynb`.

3) Advanced: Its easy to extend GeoLIPI to other langauges. Two are already supported here. One is GeoCODE [19]. Geocode creates generalized cylinder extrusions for its primitives (which can be created in GeoLIPI with `gls.QuadraticBezierExtrude3D`), and uses attachment points to attach primitives to each other(which can be done via `gls.PointRef`). Examples are shown in `notebooks/advanced_3d_language.ipynb`. Another example is modelling shapes as as set of 3D gaussian as done in [20]. An Example (with optimization) is shown in `notebooks/advanced_3d_language.ipynb`.

4) Python + GeoLIPI: As GeoLIPI is embedded in python, it is possible to build higher level languages on top of it. These higher level languages, can infact be imperative as well -> being converted into a geolipi expression when executed (GeoCode partially follows this pattern, where the `.doit()` function of `gls.PointRef` returns a tensor, converting the expression into a GeoLIPI expression without references). However, other higher level functions are also possible. Would be cool to see scene level examples!

## Not in GeoLIPI (yet)

1) ShapeAssembly - SA's attach operator works in a peculiar smartly designed way. However, creating a batched version seems non-trivial. Furthermore, its really a question of should imperative languages be supported as GeoLIPI expressions? While it is possible, it might not be the most efficient. Each expressions entails a lot of torch tensor book-keeping etc. Referencing in its current form will lead to really long and complicated expressions.

2) Neural Primitives - Its fairly simple - create primitives which store a latent code tensor, along with shape/size parameters. This will help GeoLIPI support SALAD [21], ProGRIP [22], and Spaghetti [23]. However, the usecase has not yet arisen.

3) Constraint-based languages - These are languages which are used to model shapes with constraints - just specify the constraints, and use a solver to "synthezies" shapes. No reason for GeoLIPI to support them as of now.

4) Some other primitives - We could try to support sphere-triangles, meta-balls, and other primitives. However, lack of types of primitives is not currently a bottleneck (over 40 2D primitives, and 30 3D primitives are already supported).

5) The massive Geometry Nodes in Blender. Currently, some of GeoLIPI maps to operations in Geometry Nodes, and can be used for creating beautiful renders of these simple shapes. However, it would be really cool to support more of (or all) the geometry nodes functions. A key difference is that GeoLIPI doles out functional expressions, while Geometry Nodes is an imperative state-based. So, thinking about (1) will help solve this problem as well. Carson Katrin's amazing work on [geometry-script](https://github.com/carson-katri/geometry-script) already does a lot we could add to GeoLIPI. Why should we? It might be useful to have differentiable alternatives to the operations in geometry nodes along with their batched execution - optimize in GeoLIPI and then export to blender.

## Reference

[1] CSGNet: Neural Shape Parser for Constructive Solid Geometry, Gopal Sharma, Rishabh Goyal, Difan Liu, Evangelos Kalogerakis, Subhransu Maji, CVPR 2018.

[2] UCSG-Net -- Unsupervised Discovering of Constructive Solid Geometry Tree, Kacper Kania, Maciej Zięba, Tomasz Kajdanowicz, NeurIPS 2020.

[3] Sketchformer: Transformer-based Representation for Sketched Structure, Leo Sampaio Ferraz Ribeiro, Tu Bui, John Collomosse, Moacir Ponti, CVPR 2020.

[4] A Learned Representation for Scalable Vector Graphics, Raphael Gontijo Lopes, David Ha, Douglas Eck, Jonathon Shlens, ICCV 2019.

[5] Differentiable vector graphics rasterization for editing and learning, Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, Jonathan Ragan-Kelley, SIGGRAPH Asia 2020.

[6] Towards Layer-wise Image Vectorization, Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, Humphrey Shi, CVPR 2022.

[7] Learning Shape Abstractions by Assembling Volumetric Primitives, Shubham Tulsiani, Hao Su, Leonidas J. Guibas, Alexei A. Efros, Jitendra Malik, CVPR 2017.

[8] 3D-PRNN: Generating Shape Primitives with Recurrent Neural Networks, Chuhang Zou, Ersin Yumer, Jimei Yang, Duygu Ceylan, Derek Hoiem, ICCV 2017.

[9] Split, Merge, and Refine: Fitting Tight Bounding Boxes via Learned Over-Segmentation and Iterative Search, Chanhyeok Park, Minhyuk Sung, Arxiv 2023.

[10] Deep Parametric Shape Predictions using Distance Fields, Dmitriy Smirnov, Matthew Fisher, Vladimir G. Kim, Richard Zhang, Justin Solomon, CVPR 2020

[11] Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids, Despoina Paschalidou, Ali Osman Ulusoy, Andreas Geiger, CVPR 2019

[12] CSGNet: Neural Shape Parser for Constructive Solid Geometry, Gopal Sharma, Rishabh Goyal, Difan Liu, Evangelos Kalogerakis, Subhransu Maji, CVPR 2018.

[13] CSG-Stump: A Learning Friendly CSG-Like Representation for Interpretable Shape Parsing, Daxuan Ren, Jianmin Zheng, Jianfei Cai, Jiatong Li, Haiyong Jiang, Zhongang Cai, Junzhe Zhang, Liang Pan, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, ICCV 2021

[14] CAPRI-Net: Learning Compact CAD Shapes with Adaptive Primitive Assembly, Fenggen Yu, Zhiqin Chen, Manyi Li, Aditya Sanghi, Hooman Shayani, Ali Mahdavi-Amiri, Hao Zhang, CVPR 2022

[15] CvxNet: Learnable Convex Decomposition, Boyang Deng, Kyle Genova, Soroosh Yazdani, Sofien Bouaziz, Geoffrey Hinton, Andrea Tagliasacchi, CVPR 2020.

[16] DeepCAD: A Deep Generative Network for Computer-Aided Design Models, Rundi Wu, Chang Xiao, Changxi Zheng, ICCV 2021.

[17] SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations, Pu Li, Jianwei Guo, Xiaopeng Zhang, Dong-ming Yan, CVPR 2023.

[18] Synthesizing structured CAD models with equality saturation and inverse transformations, Nandi, Chandrakana and Willsey, Max and Anderson, Adam and Wilcox, James R. and Darulova, Eva and Grossman, Dan and Tatlock, Zachary, PLDI 2020.

[19] GeoCode: Interpretable Shape Programs, Ofek Pearl, Itai Lang, Yuhua Hu, Raymond A. Yeh, Rana Hanocka, Arxiv 2023.

[20] Learning Shape Templates with Structured Implicit Functions, Kyle Genova, Forrester Cole, Daniel Vlasic, Aaron Sarna, William T. Freeman, Thomas Funkhouser, ICCV 2019.

[21] SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation, Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, Minhyuk Sung, ICCV 2023.

[22] Unsupervised Learning of Shape Programs with Repeatable Implicit Parts, Boyang Deng, Sumith Kulal, Zhengyang Dong, Congyue Deng, Yonglong Tian, Jiajun Wu, NeurIPS 2022.
[23] SPAGHETTI: Editing Implicit Shapes Through Part Aware Generation, Amir Hertz, Or Perel, Raja Giryes, Olga Sorkine-Hornung, Daniel Cohen-Or, SIGGRAPH 2022.
