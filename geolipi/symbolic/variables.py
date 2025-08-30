from .base import GLFunction
from .registry import register_symbol

@register_symbol
class Param(GLFunction):
    """This stores evaluated tensors for a parameter."""

class Operator(GLFunction):
    ...

@register_symbol
class UnaryOperator(Operator):
    """
    Apply the height field to the current height.
    """
    ...

@register_symbol
class BinaryOperator(Operator):
    """
    Apply the height field to the current height.
    """
    ...

@register_symbol
class VectorOperator(Operator):
    """
    Apply the height field to the current height.
    """
    ...


"""
The following class will be used with derivative languages. 
"""

class Variable(GLFunction):
    """This stores evaluated tensors for a variable."""


class VecList(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class Float(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class Vec2(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class Vec3(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class Vec4(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class VarSplitter(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class UniformVariable(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class UniformFloat(UniformVariable):
    """
    Apply the height field to the current height.
    """
    ...

class UniformVec2(UniformVariable):
    """
    Apply the height field to the current height.
    """
    ...
class UniformVec3(UniformVariable):
    """
    Apply the height field to the current height.
    """
    ...
    
class UniformVec4(UniformVariable):
    """
    Apply the height field to the current height.
    """
    ...

