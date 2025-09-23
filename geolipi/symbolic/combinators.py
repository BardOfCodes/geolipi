
from .base import GLFunction
from .registry import register_symbol

class Combinator(GLFunction):
    """Base class for SDF boolean operations."""

@register_symbol
class Union(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr", "varadic": True}
        }

@register_symbol
class JoinUnion(Union):
    """Blender-specific union variant."""

@register_symbol
class Intersection(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr", "varadic": True}
        }

@register_symbol
class Complement(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"}
        }

@register_symbol
class Difference(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"}
        }

@register_symbol
class SwitchedDifference(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"}
        }

@register_symbol
class SmoothUnion(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"},
            "k": {"type": "float"}
        }

@register_symbol
class SmoothIntersection(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"},
            "k": {"type": "float"}
        }

@register_symbol
class SmoothDifference(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"},
            "k": {"type": "float"}
        }

@register_symbol
class NarySmoothUnion(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"},
            "k": {"type": "float"}
        }

@register_symbol
class NarySmoothIntersection(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"},
            "k": {"type": "float"}
        }

@register_symbol
class XOR(Combinator):
    @classmethod
    def default_spec(cls):
        return {
            "expr_0": {"type": "Expr"},
            "expr_1": {"type": "Expr"}
        }