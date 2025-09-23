import numpy as np
import torch as th
from typing import Union as type_union, Tuple
from sympy import Symbol, Tuple as SympyTuple, Expr
from .base import GLExpr, GLFunction

param_type_1D = type_union[float, Expr, GLExpr, th.Tensor]
param_type_2D = type_union[Tuple, SympyTuple, np.ndarray, th.Tensor, Expr, GLExpr]
param_type_3D = type_union[Tuple, SympyTuple, np.ndarray, th.Tensor, Expr, GLExpr]
param_type_4D = type_union[Tuple, SympyTuple, np.ndarray, th.Tensor, Expr, GLExpr]
expr_type = type_union[Expr, GLExpr, GLFunction]

axis_selector = (
    Symbol("AX2D"),
    Symbol("AY2D"),
    Symbol("AX3D"),
    Symbol("AY3D"),
    Symbol("AZ3D"),
)
