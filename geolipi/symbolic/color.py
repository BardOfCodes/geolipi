from typing import Tuple, List

from .base_symbolic import Expr
from .base_symbolic import GLExpr, GLFunction


class SVGCombinator(GLFunction):
    ...

class DestinationIn(SVGCombinator):
    ...

class DestinationOut(SVGCombinator):
    ...

class DestinationOver(SVGCombinator):
    ...

class DestinationAtop(SVGCombinator):
    ...

class SourceIn(SVGCombinator):
    ...

class SourceOut(SVGCombinator):
    ...

class SourceOver(SVGCombinator):
    ...

class SourceAtop(SVGCombinator):
    ...

class SVGXOR(SVGCombinator):
    ...


class ColorModifier2D(GLFunction):
    ...
class ApplyColor2D(ColorModifier2D):
    ...

class ModifyOpacity2D(ColorModifier2D):
    ...

class ModifyColor2D(ColorModifier2D):
    ...


class SourceOverSequence(SVGCombinator):
    ...