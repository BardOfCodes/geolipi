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

@register_symbol
class BinaryOperator(Operator):
    """Binary mathematical operator."""

@register_symbol
class VectorOperator(Operator):
    """Vector mathematical operator."""


# Variable classes for derivative languages

class Variable(GLFunction):
    """Base class for variable types."""


@register_symbol
class VecList(Variable):
    """List of vectors."""

@register_symbol
class Float(Variable):
    """Float variable."""

@register_symbol
class Vec2(Variable):
    """2D vector variable."""

@register_symbol
class Vec3(Variable):
    """3D vector variable."""

@register_symbol
class Vec4(Variable):
    """4D vector variable."""

@register_symbol
class VarSplitter(Variable):
    """Variable splitter utility."""

@register_symbol
class UniformVariable(Variable):
    """Uniform variable base class."""

@register_symbol
class UniformFloat(UniformVariable):
    """Uniform float variable."""

@register_symbol
class UniformVec2(UniformVariable):
    """Uniform 2D vector variable."""


@register_symbol
class UniformVec3(UniformVariable):
    """Uniform 3D vector variable."""
    
@register_symbol
class UniformVec4(UniformVariable):
    """Uniform 4D vector variable."""

