
from .base import GLFunction
from .registry import register_symbol

class Combinator(GLFunction):
    """Base class for SDF boolean operations."""

@register_symbol
class Union(Combinator):
    pass

@register_symbol
class JoinUnion(Union):
    """Blender-specific union variant."""

@register_symbol
class Intersection(Combinator):
    pass

@register_symbol
class Complement(Combinator):
    pass

@register_symbol
class Difference(Combinator):
    pass

@register_symbol
class SwitchedDifference(Combinator):
    pass

@register_symbol
class SmoothUnion(Combinator):
    pass

@register_symbol
class SmoothIntersection(Combinator):
    pass

@register_symbol
class SmoothDifference(Combinator):
    pass

@register_symbol
class NarySmoothUnion(Combinator):
    pass

@register_symbol
class NarySmoothIntersection(Combinator):
    pass

@register_symbol
class XOR(Combinator):
    pass