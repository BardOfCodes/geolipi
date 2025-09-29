""" Based on https://github.com/patrick-kidger/sptorch """
import torch
import sympy as sp
import functools as ft
from typing import Dict, Union, Type, Callable, Any, TypeVar


ExprType = TypeVar("ExprType", bound=sp.Expr)
T = TypeVar("T")

def _I(*args: Any) -> torch.Tensor:
    return torch.tensor(1j)



def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_

SYMPY_TO_TORCH: Dict[
    Union[Type[sp.Basic], Callable[..., Any]], Callable[..., torch.Tensor]
] = {
    sp.Mul: _reduce(torch.mul),
    sp.Add: _reduce(torch.add),
    sp.div: torch.div,
    sp.Abs: torch.abs,
    sp.sign: torch.sign,
    # Note: May raise error for ints.
    sp.ceiling: torch.ceil,
    sp.floor: torch.floor,
    sp.log: torch.log,
    sp.exp: torch.exp,
    sp.sqrt: torch.sqrt,
    sp.cos: torch.cos,
    sp.acos: torch.acos,
    sp.sin: torch.sin,
    sp.asin: torch.asin,
    sp.tan: torch.tan,
    sp.atan: torch.atan,
    sp.atan2: torch.atan2,
    # Note: May give NaN for complex results.
    sp.cosh: torch.cosh,
    sp.acosh: torch.acosh,
    sp.sinh: torch.sinh,
    sp.asinh: torch.asinh,
    sp.tanh: torch.tanh,
    sp.atanh: torch.atanh,
    sp.Pow: torch.pow,
    sp.re: torch.real,
    sp.im: torch.imag,
    sp.arg: torch.angle,
    # Note: May raise error for ints and complexes
    sp.erf: torch.erf,
    sp.loggamma: torch.lgamma,
    sp.Eq: torch.eq,
    sp.Ne: torch.ne,
    sp.StrictGreaterThan: torch.gt,
    sp.StrictLessThan: torch.lt,
    sp.LessThan: torch.le,
    sp.GreaterThan: torch.ge,
    sp.And: torch.logical_and,
    sp.Or: torch.logical_or,
    sp.Not: torch.logical_not,
    sp.Max: torch.max,
    sp.Min: torch.min,
    # Matrices
    sp.MatAdd: torch.add,
    sp.HadamardProduct: torch.mul,
    sp.Trace: torch.trace,
    # Note: May raise error for integer matrices.
    sp.Determinant: torch.det,
    sp.core.numbers.ImaginaryUnit: _I,
    sp.conjugate: torch.conj,
}

TEXT_TO_SYMPY = {
    "mul": sp.Mul,
    "add": sp.Add,
    "div": sp.div,
    "abs": sp.Abs,
    "sign": sp.sign,
    "ceiling": sp.ceiling,
    "floor": sp.floor,
    "log": sp.log,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "cos": sp.cos,
    "acos": sp.acos,
    "sin": sp.sin,
    "asin": sp.asin,
    "tan": sp.tan,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "cosh": sp.cosh,
    "acosh": sp.acosh,
    "sinh": sp.sinh,
    "asinh": sp.asinh,
    "tanh": sp.tanh,
    "atanh": sp.atanh,
    "pow": sp.Pow,
    "re": sp.re,
    "im": sp.im,
    "arg": sp.arg,
    "erf": sp.erf,
    "loggamma": sp.loggamma,
    "eq": sp.Eq,
    "ne": sp.Ne,
    "gt": sp.StrictGreaterThan,
    "lt": sp.StrictLessThan,
    "le": sp.LessThan,
    "ge": sp.GreaterThan,
    "and": sp.And,
    "or": sp.Or,
    "not": sp.Not,
    "max": sp.Max,
    "min": sp.Min,
    "matadd": sp.MatAdd,
    "hadamard": sp.HadamardProduct,
    "trace": sp.Trace,
    "det": sp.Determinant,
    "i": sp.core.numbers.ImaginaryUnit,
    "conjugate": sp.conjugate,
}

SYMPY_TO_TEXT = {
    sp.Mul: "mul",
    sp.Add: "add",
    sp.div: "div",
    sp.Abs: "abs",
    sp.sign: "sign",
    sp.ceiling: "ceiling",
    sp.floor: "floor",
    sp.log: "log",
    sp.exp: "exp",
    sp.sqrt: "sqrt",
    sp.cos: "cos",
    sp.acos: "acos",
    sp.sin: "sin",
    sp.asin: "asin",
    sp.tan: "tan",
    sp.atan: "atan",
    sp.atan2: "atan2",
    sp.cosh: "cosh",
    sp.acosh: "acosh",
    sp.sinh: "sinh",
    sp.asinh: "asinh",
    sp.tanh: "tanh",
    sp.atanh: "atanh",
    sp.Pow: "pow",
    sp.re: "re",
    sp.im: "im",
    sp.arg: "arg",
    sp.erf: "erf",
    sp.loggamma: "loggamma",
    sp.Eq: "eq",
    sp.Ne: "ne",
    sp.StrictGreaterThan: "gt",
    sp.StrictLessThan: "lt",
    sp.LessThan: "le",
    sp.GreaterThan: "ge",
    sp.And: "and",
    sp.Or: "or",
    sp.Not: "not",
    sp.Max: "max",
    sp.Min: "min",
    sp.MatAdd: "matadd",
    sp.HadamardProduct: "hadamard",
    sp.Trace: "trace",
    sp.Determinant: "det",
    sp.core.numbers.ImaginaryUnit: "i",
    sp.conjugate: "conjugate",
}