""" Based on https://github.com/patrick-kidger/sympytorch """
import torch
import sympy
import functools as ft
from typing import Dict, Union, Type, Callable, Any, TypeVar


ExprType = TypeVar("ExprType", bound=sympy.Expr)
T = TypeVar("T")

def _I(*args: Any) -> torch.Tensor:
    return torch.tensor(1j)



def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_

SYMPY_TO_TORCH: Dict[
    Union[Type[sympy.Basic], Callable[..., Any]], Callable[..., torch.Tensor]
] = {
    sympy.Mul: _reduce(torch.mul),
    sympy.Add: _reduce(torch.add),
    sympy.div: torch.div,
    sympy.Abs: torch.abs,
    sympy.sign: torch.sign,
    # Note: May raise error for ints.
    sympy.ceiling: torch.ceil,
    sympy.floor: torch.floor,
    sympy.log: torch.log,
    sympy.exp: torch.exp,
    sympy.sqrt: torch.sqrt,
    sympy.cos: torch.cos,
    sympy.acos: torch.acos,
    sympy.sin: torch.sin,
    sympy.asin: torch.asin,
    sympy.tan: torch.tan,
    sympy.atan: torch.atan,
    sympy.atan2: torch.atan2,
    # Note: May give NaN for complex results.
    sympy.cosh: torch.cosh,
    sympy.acosh: torch.acosh,
    sympy.sinh: torch.sinh,
    sympy.asinh: torch.asinh,
    sympy.tanh: torch.tanh,
    sympy.atanh: torch.atanh,
    sympy.Pow: torch.pow,
    sympy.re: torch.real,
    sympy.im: torch.imag,
    sympy.arg: torch.angle,
    # Note: May raise error for ints and complexes
    sympy.erf: torch.erf,
    sympy.loggamma: torch.lgamma,
    sympy.Eq: torch.eq,
    sympy.Ne: torch.ne,
    sympy.StrictGreaterThan: torch.gt,
    sympy.StrictLessThan: torch.lt,
    sympy.LessThan: torch.le,
    sympy.GreaterThan: torch.ge,
    sympy.And: torch.logical_and,
    sympy.Or: torch.logical_or,
    sympy.Not: torch.logical_not,
    sympy.Max: torch.max,
    sympy.Min: torch.min,
    # Matrices
    sympy.MatAdd: torch.add,
    sympy.HadamardProduct: torch.mul,
    sympy.Trace: torch.trace,
    # Note: May raise error for integer matrices.
    sympy.Determinant: torch.det,
    sympy.core.numbers.ImaginaryUnit: _I,
    sympy.conjugate: torch.conj,
}

TEXT_TO_SYMPY = {
    "mul": sympy.Mul,
    "add": sympy.Add,
    "div": sympy.div,
    "abs": sympy.Abs,
    "sign": sympy.sign,
    "ceiling": sympy.ceiling,
    "floor": sympy.floor,
    "log": sympy.log,
    "exp": sympy.exp,
    "sqrt": sympy.sqrt,
    "cos": sympy.cos,
    "acos": sympy.acos,
    "sin": sympy.sin,
    "asin": sympy.asin,
    "tan": sympy.tan,
    "atan": sympy.atan,
    "atan2": sympy.atan2,
    "cosh": sympy.cosh,
    "acosh": sympy.acosh,
    "sinh": sympy.sinh,
    "asinh": sympy.asinh,
    "tanh": sympy.tanh,
    "atanh": sympy.atanh,
    "pow": sympy.Pow,
    "re": sympy.re,
    "im": sympy.im,
    "arg": sympy.arg,
    "erf": sympy.erf,
    "loggamma": sympy.loggamma,
    "eq": sympy.Eq,
    "ne": sympy.Ne,
    "gt": sympy.StrictGreaterThan,
    "lt": sympy.StrictLessThan,
    "le": sympy.LessThan,
    "ge": sympy.GreaterThan,
    "and": sympy.And,
    "or": sympy.Or,
    "not": sympy.Not,
    "max": sympy.Max,
    "min": sympy.Min,
    "matadd": sympy.MatAdd,
    "hadamard": sympy.HadamardProduct,
    "trace": sympy.Trace,
    "det": sympy.Determinant,
    "i": sympy.core.numbers.ImaginaryUnit,
    "conjugate": sympy.conjugate,
}