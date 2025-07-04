import os
import torch as th
import matplotlib.colors

import geolipi.symbolic.combinators as sym_comb
import geolipi.symbolic.primitives_3d as sym_prim3d
import geolipi.symbolic.primitives_2d as sym_prim2d
import geolipi.symbolic.transforms_3d as sym_t3d
import geolipi.symbolic.transforms_2d as sym_t2d
import geolipi.symbolic.primitives_higher as sym_higher
import geolipi.symbolic.color as sym_color

import geolipi.torch_compute.sdf_operators as sdf_op_bank
import geolipi.torch_compute.sdf_functions_2d as sdf2d_bank
import geolipi.torch_compute.sdf_functions_3d as sdf3d_bank
import geolipi.torch_compute.transforms as transform_bank
import geolipi.torch_compute.sdf_functions_higher as higher_sdf_bank
import geolipi.torch_compute.color_functions as color_func_bank
import geolipi.torch_compute.colorspace_functions as colorspace_bank

COMBINATOR_MAP = {
    sym_comb.Union: sdf_op_bank.sdf_union,
    sym_comb.JoinUnion: sdf_op_bank.sdf_union,
    sym_comb.Intersection: sdf_op_bank.sdf_intersection,
    sym_comb.Complement: sdf_op_bank.sdf_complement,
    sym_comb.Difference: sdf_op_bank.sdf_difference,
    sym_comb.SwitchedDifference: sdf_op_bank.sdf_switched_difference,
    sym_comb.SmoothUnion: sdf_op_bank.sdf_smooth_union,
    sym_comb.SmoothIntersection: sdf_op_bank.sdf_smooth_intersection,
    sym_comb.SmoothDifference: sdf_op_bank.sdf_smooth_difference,
}

PRIMITIVE_MAP = {
    # 2D
    sym_prim2d.Circle2D: sdf2d_bank.sdf2d_circle,
    sym_prim2d.RoundedBox2D: sdf2d_bank.sdf2d_rounded_box,
    sym_prim2d.Box2D: sdf2d_bank.sdf2d_box,
    sym_prim2d.Rectangle2D: sdf2d_bank.sdf2d_box,
    sym_prim2d.OrientedBox2D: sdf2d_bank.sdf2d_oriented_box,
    sym_prim2d.Rhombus2D: sdf2d_bank.sdf2d_rhombus,
    sym_prim2d.Trapezoid2D: sdf2d_bank.sdf2d_trapezoid,
    sym_prim2d.Parallelogram2D: sdf2d_bank.sdf2d_parallelogram,
    sym_prim2d.EquilateralTriangle2D: sdf2d_bank.sdf2d_equilateral_triangle,
    sym_prim2d.IsoscelesTriangle2D: sdf2d_bank.sdf2d_isosceles_triangle,
    sym_prim2d.Triangle2D: sdf2d_bank.sdf2d_triangle,
    sym_prim2d.UnevenCapsule2D: sdf2d_bank.sdf2d_uneven_capsule,
    sym_prim2d.RegularPentagon2D: sdf2d_bank.sdf2d_regular_pentagon,
    sym_prim2d.RegularHexagon2D: sdf2d_bank.sdf2d_regular_hexagon,
    sym_prim2d.RegularOctagon2D: sdf2d_bank.sdf2d_regular_octagon,
    sym_prim2d.Hexagram2D: sdf2d_bank.sdf2d_hexagram,
    sym_prim2d.Star2D: sdf2d_bank.sdf2d_star_5,
    sym_prim2d.RegularStar2D: sdf2d_bank.sdf2d_regular_star,
    sym_prim2d.Pie2D: sdf2d_bank.sdf2d_pie,
    sym_prim2d.CutDisk2D: sdf2d_bank.sdf2d_cut_disk,
    sym_prim2d.Arc2D: sdf2d_bank.sdf2d_arc,
    sym_prim2d.HorseShoe2D: sdf2d_bank.sdf2d_horse_shoe,
    sym_prim2d.Vesica2D: sdf2d_bank.sdf2d_vesica,
    sym_prim2d.OrientedVesica2D: sdf2d_bank.sdf2d_oriented_vesica,
    sym_prim2d.Moon2D: sdf2d_bank.sdf2d_moon,
    sym_prim2d.RoundedCross2D: sdf2d_bank.sdf2d_rounded_cross,
    sym_prim2d.Egg2D: sdf2d_bank.sdf2d_egg,
    sym_prim2d.Heart2D: sdf2d_bank.sdf2d_heart,
    sym_prim2d.Cross2D: sdf2d_bank.sdf2d_cross,
    sym_prim2d.RoundedX2D: sdf2d_bank.sdf2d_rounded_x,
    sym_prim2d.Polygon2D: sdf2d_bank.sdf2d_polygon,
    sym_prim2d.Ellipse2D: sdf2d_bank.sdf2d_ellipse,
    sym_prim2d.Parabola2D: sdf2d_bank.sdf2d_parabola,
    sym_prim2d.ParabolaSegment2D: sdf2d_bank.sdf2d_parabola_segment,
    sym_prim2d.BlobbyCross2D: sdf2d_bank.sdf2d_blobby_cross,
    sym_prim2d.Tunnel2D: sdf2d_bank.sdf2d_tunnel,
    sym_prim2d.Stairs2D: sdf2d_bank.sdf2d_stairs,
    sym_prim2d.QuadraticCircle2D: sdf2d_bank.sdf2d_quadratic_circle,
    sym_prim2d.CoolS2D: sdf2d_bank.sdf2d_cool_s,
    sym_prim2d.CircleWave2D: sdf2d_bank.sdf2d_circle_wave,
    sym_prim2d.Hyperbola2D: sdf2d_bank.sdf2d_hyperbola,
    sym_prim2d.QuadraticBezierCurve2D: sdf2d_bank.sdf2d_quadratic_bezier_curve,
    sym_prim2d.Segment2D: sdf2d_bank.sdf2d_segment,
    sym_prim2d.NoParamRectangle2D: sdf2d_bank.sdf2d_no_param_rectangle,
    sym_prim2d.NoParamCircle2D: sdf2d_bank.sdf2d_no_param_circle,
    sym_prim2d.NoParamTriangle2D: sdf2d_bank.sdf2d_no_param_triangle,
    sym_prim2d.NullExpression2D: sdf_op_bank.sdf_null_op,
    sym_prim2d.InstantiatedPrim2D: sdf2d_bank.nonsdf2d_instantiated_prim,
    # These should be shifted to the other repositories - based on declarations.
    sym_prim2d.PolyLine2D: sdf2d_bank.sdf2d_polyline,
    # Tile
    sym_prim2d.TileUV2D: sdf2d_bank.nonsdf2d_tile_uv,
    # SingleDim
    sym_prim2d.SinRepeatX2D: sdf2d_bank.nonsdf2d_sin_x,
    sym_prim2d.SinRepeatY2D: sdf2d_bank.nonsdf2d_sin_y,
    sym_prim2d.SinAlongAxisY2D: sdf2d_bank.nonsdf2d_sin_along_axis_y,
    sym_prim2d.SinDiagonal2D: sdf2d_bank.nonsdf2d_sin_diagonal,
    sym_prim2d.SinDiagonalFlip2D: sdf2d_bank.nonsdf2d_sin_diagonal_flip,
    sym_prim2d.SinRadial2D: sdf2d_bank.nonsdf2d_sin_radial,
    sym_prim2d.SquiggleX2D: sdf2d_bank.nonsdf2d_squiggle_lines_x,
    sym_prim2d.SquiggleY2D: sdf2d_bank.nonsdf2d_squiggle_lines_y,
    sym_prim2d.SquiggleDiagonal2D: sdf2d_bank.nonsdf2d_squiggle_diagonal,
    sym_prim2d.SquiggleDiagonalFlip2D: sdf2d_bank.nonsdf2d_squiggle_diagonal_flip,
    sym_prim2d.SquiggleRadial2D: sdf2d_bank.nonsdf2d_squiggle_radial,
    sym_prim2d.SquiggleDistortion2D: sdf2d_bank.nonsdf2d_squiggle_radial_distortion,
    # 3D
    sym_prim3d.Sphere3D: sdf3d_bank.sdf3d_sphere,
    sym_prim3d.Box3D: sdf3d_bank.sdf3d_box,
    sym_prim3d.Cuboid3D: sdf3d_bank.sdf3d_box,
    sym_prim3d.RoundedBox3D: sdf3d_bank.sdf3d_rounded_box,
    sym_prim3d.BoxFrame3D: sdf3d_bank.sdf3d_box_frame,
    sym_prim3d.Torus3D: sdf3d_bank.sdf3d_torus,
    sym_prim3d.CappedTorus3D: sdf3d_bank.sdf3d_capped_torus,
    sym_prim3d.Link3D: sdf3d_bank.sdf3d_link,
    sym_prim3d.InfiniteCylinder3D: sdf3d_bank.sdf3d_infinite_cylinder,
    sym_prim3d.InfiniteCone3D: sdf3d_bank.sdf3d_infinite_cone,
    sym_prim3d.Cone3D: sdf3d_bank.sdf3d_cone,
    sym_prim3d.InexactCone3D: sdf3d_bank.sdf3d_inexact_cone,
    sym_prim3d.Plane3D: sdf3d_bank.sdf3d_plane,
    sym_prim3d.HexPrism3D: sdf3d_bank.sdf3d_hex_prism,
    sym_prim3d.TriPrism3D: sdf3d_bank.sdf3d_tri_prism,
    sym_prim3d.Capsule3D: sdf3d_bank.sdf3d_capsule,
    sym_prim3d.VerticalCapsule3D: sdf3d_bank.sdf3d_vertical_capsule,
    sym_prim3d.CappedCylinder3D: sdf3d_bank.sdf3d_capped_cylinder,
    sym_prim3d.Cylinder3D: sdf3d_bank.sdf3d_capped_cylinder,
    sym_prim3d.ArbitraryCappedCylinder3D: sdf3d_bank.sdf3d_arbitrary_capped_cylinder,
    sym_prim3d.RoundedCylinder3D: sdf3d_bank.sdf3d_rounded_cylinder,
    sym_prim3d.CappedCone3D: sdf3d_bank.sdf3d_capped_cone,
    sym_prim3d.ArbitraryCappedCone: sdf3d_bank.sdf3d_arbitrary_capped_cone,
    sym_prim3d.SolidAngle3D: sdf3d_bank.sdf3d_solid_angle,
    sym_prim3d.CutSphere3D: sdf3d_bank.sdf3d_cut_sphere,
    sym_prim3d.CutHollowSphere: sdf3d_bank.sdf3d_cut_hollow_sphere,
    sym_prim3d.DeathStar3D: sdf3d_bank.sdf3d_death_star,
    sym_prim3d.RoundCone3D: sdf3d_bank.sdf3d_round_cone,
    sym_prim3d.ArbitraryRoundCone3D: sdf3d_bank.sdf3d_arbitrary_round_cone,
    sym_prim3d.InexactEllipsoid3D: sdf3d_bank.sdf3d_inexact_ellipsoid,
    sym_prim3d.RevolvedVesica3D: sdf3d_bank.sdf3d_revolved_vesica,
    sym_prim3d.Rhombus3D: sdf3d_bank.sdf3d_rhombus,
    sym_prim3d.Octahedron3D: sdf3d_bank.sdf3d_octahedron,
    sym_prim3d.InexactOctahedron3D: sdf3d_bank.sdf3d_inexact_octahedron,
    sym_prim3d.Pyramid3D: sdf3d_bank.sdf3d_pyramid,
    sym_prim3d.Triangle3D: sdf3d_bank.sdf3d_triangle,
    sym_prim3d.Quadrilateral3D: sdf3d_bank.sdf3d_quadrilateral,
    sym_prim3d.NoParamCuboid3D: sdf3d_bank.sdf3d_no_param_cuboid,
    sym_prim3d.NoParamSphere3D: sdf3d_bank.sdf3d_no_param_sphere,
    sym_prim3d.NoParamCylinder3D: sdf3d_bank.sdf3d_no_param_cylinder,
    sym_prim3d.InexactSuperQuadrics3D: sdf3d_bank.sdf3d_inexact_super_quadrics,
    sym_prim3d.InexactAnisotropicGaussian3D: sdf3d_bank.sdf3d_inexact_anisotropic_gaussian,
    sym_prim3d.NullExpression3D: sdf_op_bank.sdf_null_op,
    # Higher Order
    sym_higher.LinearExtrude3D: higher_sdf_bank.sdf3d_linear_extrude,
    sym_higher.QuadraticBezierExtrude3D: higher_sdf_bank.sdf3d_quadratic_bezier_extrude,
    sym_higher.Revolution3D: higher_sdf_bank.sdf3d_revolution,
    sym_higher.SimpleExtrusion3D: higher_sdf_bank.sdf3d_simple_extrusion,
    sym_higher.LinearCurve1D: higher_sdf_bank.linear_curve_1d,
    sym_higher.QuadraticCurve1D: higher_sdf_bank.quadratic_curve_1d,
}

# Try compile
# PRIMITIVE_MAP = {x: th.compile(y) for x, y in PRIMITIVE_MAP.items()}


MODIFIER_MAP = {
    # 2D
    sym_t2d.Translate2D: transform_bank.get_affine_translate_2D,
    sym_t2d.EulerRotate2D: transform_bank.get_affine_rotate_2D,
    sym_t2d.Scale2D: transform_bank.get_affine_scale_2D,
    sym_t2d.Shear2D: transform_bank.get_affine_shear_2D,
    sym_t2d.Affine2D: transform_bank.get_affine_matrix_2D,
    sym_t2d.Distort2D: transform_bank.position_distort,
    sym_t2d.ReflectCoords2D: transform_bank.get_affine_reflection_2D,
    sym_t2d.Dilate2D: sdf_op_bank.sdf_dilate,
    sym_t2d.Erode2D: sdf_op_bank.sdf_erode,
    sym_t2d.Onion2D: sdf_op_bank.sdf_onion,
    # 3D
    sym_t3d.Translate3D: transform_bank.get_affine_translate_3D,
    sym_t3d.EulerRotate3D: transform_bank.get_affine_rotate_euler_3D,
    sym_t3d.AxisAngleRotate3D: transform_bank.get_affine_rotate_axis_angle_3D,
    sym_t3d.RotateMatrix3D: transform_bank.get_affine_rotate_matrix_3D,
    sym_t3d.Scale3D: transform_bank.get_affine_scale_3D,
    sym_t3d.Shear3D: transform_bank.get_affine_shear_3D,
    sym_t3d.Distort3D: transform_bank.position_distort,
    sym_t3d.Twist3D: transform_bank.position_twist,
    sym_t3d.Bend3D: transform_bank.position_cheap_bend,
    sym_t3d.ReflectCoords3D: transform_bank.get_affine_reflection_3D,
    sym_t3d.Dilate3D: sdf_op_bank.sdf_dilate,
    sym_t3d.Erode3D: sdf_op_bank.sdf_erode,
    sym_t3d.Onion3D: sdf_op_bank.sdf_onion,
}

# use XKCD colors: https://xkcd.com/color/rgb.txt
color_file = open(os.path.dirname(__file__) + "/xkcd_rgb.txt", "r").readlines()
color_names = [x.strip().split("\t")[0].strip() for x in color_file[1:]]
color_hexes = [x.strip().split("\t")[1].strip() for x in color_file[1:]]
color_val = [th.tensor([matplotlib.colors.to_rgba(h)]) for h in color_hexes]
COLOR_MAP = dict(zip(color_names, color_val))

COLOR_FUNCTIONS = {
    sym_color.DestinationAtop: color_func_bank.destination_atop,
    sym_color.DestinationIn: color_func_bank.destination_in,
    sym_color.DestinationOut: color_func_bank.destination_out,
    sym_color.DestinationOver: color_func_bank.destination_over,
    sym_color.SourceIn: color_func_bank.source_in,
    sym_color.SourceOut: color_func_bank.source_out,
    sym_color.SourceOver: color_func_bank.source_over,
    sym_color.SourceAtop: color_func_bank.source_atop,
    sym_color.SVGXOR: color_func_bank.svg_xor,
    sym_color.SourceOverSequence: color_func_bank.source_over_seq,
    sym_color.ApplyColor2D: color_func_bank.apply_color,
    # These also go into Splitweave.
    sym_color.ModifyOpacity2D: color_func_bank.modify_opacity,
    sym_color.ModifyColor2D: color_func_bank.modify_color,
    sym_color.ModifyColorTritone2D: color_func_bank.depreciated_modify_color_tritone,
    sym_color.AlphaMask2D: color_func_bank.alpha_mask,
    sym_color.RGB2HSL: colorspace_bank.rgb2hsl_torch,
    sym_color.RGB2HSV: colorspace_bank.rgb2hsv_torch,
    sym_color.HSV2RGB: colorspace_bank.hsv2rgb_torch,
    sym_color.HSL2RGB: colorspace_bank.hsl2rgb_torch,
    sym_color.HueShift: colorspace_bank.hue_shift_torch,
    # COLOR SDF
    sym_color.AlphaToSDF2D: color_func_bank.unopt_alpha_to_sdf,
}


# Used during compilation:
INVERTED_MAP = {
    sym_comb.Union: sym_comb.Intersection,
    sym_comb.Intersection: sym_comb.Union,
    sym_comb.Difference: sym_comb.Union,
    sym_comb.Complement: sym_comb.Complement,
    
}
NORMAL_MAP = {
    sym_comb.Union: sym_comb.Union,
    sym_comb.Intersection: sym_comb.Intersection,
    sym_comb.Difference: sym_comb.Intersection,
    sym_comb.Complement: sym_comb.Complement,
}
ONLY_SIMPLIFY_RULES = set(
    [(sym_comb.Intersection, sym_comb.Intersection), (sym_comb.Union, sym_comb.Union)]
)
ALL_RULES = set(
    [
        (sym_comb.Intersection, sym_comb.Intersection),
        (sym_comb.Union, sym_comb.Union),
        (sym_comb.Intersection, sym_comb.Union),
    ]
)
# CNF-specific rules
ONLY_SIMPLIFY_RULES_CNF = set([
    (sym_comb.Intersection, sym_comb.Intersection),
    (sym_comb.Union, sym_comb.Union)
])

ALL_RULES_CNF = set([
    (sym_comb.Intersection, sym_comb.Intersection),
    (sym_comb.Union, sym_comb.Union),
    (sym_comb.Union, sym_comb.Intersection),  # Key difference: Distribute Union over Intersection
])
