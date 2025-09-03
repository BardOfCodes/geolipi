from .base import GLFunction
from .registry import register_symbol

class SVGCombinator(GLFunction):
    """Base class for SVG color blending operations."""


@register_symbol
class DestinationIn(SVGCombinator):
    pass


@register_symbol
class DestinationOut(SVGCombinator):
    pass


@register_symbol
class DestinationOver(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.destination_over
    Read evaluator specific documentation for more.
    """


@register_symbol
class DestinationAtop(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.destination_atop
    Read evaluator specific documentation for more.
    """


@register_symbol
class SourceIn(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_in
    Read evaluator specific documentation for more.
    """


@register_symbol
class SourceOut(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_out
    Read evaluator specific documentation for more.
    """


@register_symbol
class SourceOver(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_over
    Read evaluator specific documentation for more.
    """


@register_symbol
class SourceAtop(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_atop
    Read evaluator specific documentation for more.
    """


@register_symbol
class SVGXOR(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.svg_xor
    Read evaluator specific documentation for more.
    """


class ColorModifier2D(GLFunction):
    """
    This class is a base for color modifier functions in torch_compute.
    Read evaluator specific documentation for more.
    """


@register_symbol
class ApplyColor2D(ColorModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.apply_color
    Read evaluator specific documentation for more.
    """

@register_symbol
class ModifyOpacity2D(ColorModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.modify_opacity
    Read evaluator specific documentation for more.
    """


@register_symbol
class ModifyColor2D(ColorModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.modify_color
    Read evaluator specific documentation for more.
    """

@register_symbol
class ModifyColorTritone2D(ColorModifier2D):
    """
    This is used to recolor tiles.
    """

@register_symbol
class SourceOverSequence(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_over_seq
    Read evaluator specific documentation for more.
    """

@register_symbol
class AlphaMask2D(SVGCombinator):
    """
    Gather the alpha mask from a SVG eval output.
    """



@register_symbol
class AlphaToSDF2D(GLFunction):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.alpha_to_sdf
    Read evaluator specific documentation for more.
    """

@register_symbol
class RGB2HSL(AlphaToSDF2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.rgb2hsl
    Read evaluator specific documentation for more.
    """

@register_symbol
class RGB2HSV(AlphaToSDF2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.rgb2hsv
    Read evaluator specific documentation for more.
    """

@register_symbol
class HSV2RGB(AlphaToSDF2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.hsv2rgb
    Read evaluator specific documentation for more.
    """

@register_symbol
class HSL2RGB(AlphaToSDF2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.hsl2rgb
    Read evaluator specific documentation for more.
    """


@register_symbol
class HueShift(AlphaToSDF2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.change_hue
    Read evaluator specific documentation for more.
    """