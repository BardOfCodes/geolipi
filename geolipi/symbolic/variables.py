from .base import GLFunction


class Param(GLFunction):
    """This stores evaluated tensors for a parameter."""

class Operator(GLFunction):
    ...


class UnaryOperator(Operator):
    """
    Apply the height field to the current height.
    """
    ...

class BinaryOperator(Operator):
    """
    Apply the height field to the current height.
    """
    ...
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

class UniformFloat(Variable):
    """
    Apply the height field to the current height.
    """
    ...

class UniformVec2(Variable):
    """
    Apply the height field to the current height.
    """
    ...
class UniformVec3(Variable):
    """
    Apply the height field to the current height.
    """
    ...
    
class UniformVec4(Variable):
    """
    Apply the height field to the current height.
    """
    ...

