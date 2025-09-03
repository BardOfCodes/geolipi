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


class VecList(Variable):
    """List of vectors."""

class Float(Variable):
    """Float variable."""

class Vec2(Variable):
    """2D vector variable."""

class Vec3(Variable):
    """3D vector variable."""

class Vec4(Variable):
    """4D vector variable."""

class VarSplitter(Variable):
    """Variable splitter utility."""

class UniformVariable(Variable):
    """Uniform variable base class."""

class UniformFloat(UniformVariable):
    """Uniform float variable."""

class UniformVec2(UniformVariable):
    """Uniform 2D vector variable."""
class UniformVec3(UniformVariable):
    """Uniform 3D vector variable."""
    
class UniformVec4(UniformVariable):
    """Uniform 4D vector variable."""

