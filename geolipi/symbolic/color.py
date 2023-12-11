from .base_symbolic import GLFunction


class SVGCombinator(GLFunction):
    """
    Base class for all SVG combinators.
    """


class DestinationIn(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.destination_in
    Read evaluator specific documentation for more.
    """


class DestinationOut(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.destination_out
    Read evaluator specific documentation for more.
    """


class DestinationOver(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.destination_over
    Read evaluator specific documentation for more.
    """


class DestinationAtop(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.destination_atop
    Read evaluator specific documentation for more.
    """


class SourceIn(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_in
    Read evaluator specific documentation for more.
    """


class SourceOut(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_out
    Read evaluator specific documentation for more.
    """


class SourceOver(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_over
    Read evaluator specific documentation for more.
    """


class SourceAtop(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_atop
    Read evaluator specific documentation for more.
    """


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


class ApplyColor2D(ColorModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.apply_color
    Read evaluator specific documentation for more.
    """


class ModifyOpacity2D(ColorModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.modify_opacity
    Read evaluator specific documentation for more.
    """


class ModifyColor2D(ColorModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.modify_color
    Read evaluator specific documentation for more.
    """


class SourceOverSequence(SVGCombinator):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.color_functions.source_over_seq
    Read evaluator specific documentation for more.
    """
