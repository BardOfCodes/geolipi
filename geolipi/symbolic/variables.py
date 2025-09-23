from .base import GLFunction
from .registry import register_symbol

@register_symbol
class Param(GLFunction):
    """Stores parameter values as tensors."""

class Operator(GLFunction):
    ...

@register_symbol
class UnaryOperator(Operator):
    """Unary mathematical operator."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "op": {"type": "Enum[\"sin\"|\"cos\"|\"tan\"|\"log\"|\"exp\"|\"sqrt\"|\"abs\"|\"floor\"|\"ceil\"|\"round\"|\"frac\"|\"sign\"|\"normalize\"|\"norm\"|\"neg\"]"}}

@register_symbol
class BinaryOperator(Operator):
    """Binary mathematical operator."""
    @classmethod
    def default_spec(cls):
        return {"expr_0": {"type": "Expr"}, "expr_1": {"type": "Expr"}, "op": {"type": "Enum[\"add\"|\"sub\"|\"mul\"|\"div\"|\"pow\"|\"atan2\"|\"min\"|\"max\"|\"step\"|\"mod\"]"}}

@register_symbol
class VectorOperator(Operator):
    """Vector mathematical operator."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "op": {"type": "Enum[\"normalize\"]"}}


# Variable classes for derivative languages

class Variable(GLFunction):
    """Base class for variable types."""


@register_symbol
class VecList(Variable):
    """List of vectors."""
    @classmethod
    def default_spec(cls):
        return {"vectors": {"type": "List[Vector[3]]"}, "count": {"type": "int"}}

@register_symbol
class Float(Variable):
    """Float variable."""
    @classmethod
    def default_spec(cls):
        return {"value": {"type": "float"}}

@register_symbol
class Vec2(Variable):
    """2D vector variable."""
    @classmethod
    def default_spec(cls):
        return {"x": {"type": "float"}, "y": {"type": "float"}}

@register_symbol
class Vec3(Variable):
    """3D vector variable."""
    @classmethod
    def default_spec(cls):
        return {"x": {"type": "float"}, "y": {"type": "float"}, "z": {"type": "float"}}

@register_symbol
class Vec4(Variable):
    """4D vector variable."""
    @classmethod
    def default_spec(cls):
        return {"x": {"type": "float"}, "y": {"type": "float"}, "z": {"type": "float"}, "w": {"type": "float"}}

@register_symbol
class VarSplitter(Variable):
    """Variable splitter utility."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "index": {"type": "int"}}

@register_symbol
class UniformVariable(Variable):
    """Uniform variable base class."""

@register_symbol
class UniformFloat(UniformVariable):
    """Uniform float variable."""
    @classmethod
    def default_spec(cls):
        return {"min": {"type": "float"}, "default": {"type": "float"}, "max": {"type": "float"}, "name": {"type": "str"}}

@register_symbol
class UniformVec2(UniformVariable):
    """Uniform 2D vector variable."""
    @classmethod
    def default_spec(cls):
        return {"min": {"type": "Vector[2]"}, "default": {"type": "Vector[2]"}, "max": {"type": "Vector[2]"}, "name": {"type": "str"}}


@register_symbol
class UniformVec3(UniformVariable):
    """Uniform 3D vector variable."""
    @classmethod
    def default_spec(cls):
        return {"min": {"type": "Vector[3]"}, "default": {"type": "Vector[3]"}, "max": {"type": "Vector[3]"}, "name": {"type": "str"}}
    
@register_symbol
class UniformVec4(UniformVariable):
    """Uniform 4D vector variable."""
    @classmethod
    def default_spec(cls):
        return {"min": {"type": "Vector[4]"}, "default": {"type": "Vector[4]"}, "max": {"type": "Vector[4]"}, "name": {"type": "str"}}

