import numpy as np
import torch as th
from sympy import Function, Expr, Symbol, Tuple as SympyTuple, Integer as SympyInteger, Float as SympyFloat
from typing import Dict, Tuple
import sympy
import re


def magic_method_decorator(cls):
    magic_methods = ['__add__', '__sub__', '__mul__',
                     '__truediv__', '__pow__']  # extend this list as needed

    def make_magic_method(magic):
        def method(self, other):
            base_method = getattr(Expr, magic)

            if isinstance(other, (GLExpr, GLFunction)):
                merged_lookup_table = {
                    **self.lookup_table, **other.lookup_table}
                return GLExpr(base_method(self.expr, other.expr), merged_lookup_table)
            else:
                return GLExpr(base_method(self.expr, other), self.lookup_table)

        return method

    for magic in magic_methods:
        setattr(cls, magic, make_magic_method(magic))

    return cls


def magic_method_decorator_for_function(cls):
    magic_methods = ['__add__', '__sub__', '__mul__',
                     '__truediv__', '__pow__']  # extend this list as needed

    def make_magic_method(magic):
        base_method = getattr(Function, magic)

        def method(self, other):
            if isinstance(other, GLExpr):
                new_expr = base_method(self, other.expr)
                merged_lookup_table = {**self.lookup_table,
                                       **other.lookup_table}
                return GLExpr(new_expr, merged_lookup_table)
            elif isinstance(other, GLFunction):
                new_expr = base_method(self, other)
                merged_lookup_table = {**self.lookup_table,
                                       **other.lookup_table}
                return GLExpr(new_expr, merged_lookup_table)
            else:
                new_expr = base_method(self, other)
                return GLExpr(new_expr, self.lookup_table)

        return method

    for magic in magic_methods:
        setattr(cls, magic, make_magic_method(magic))

    return cls


@magic_method_decorator
class GLExpr:
    __sympy__ = True  # To avoid sympifying the expression which raises errors.

    def __init__(self, expr: Expr, lookup_table: Dict = None):
        self.expr = expr
        self.lookup_table = lookup_table or {}

    # Implement other arithmetic operations like __sub__ similarly
    @property
    def args(self):
        return self.expr.args

    def __str__(self):
        # Build a pattern that matches any of the symbols in the lookup table
        expr_str = self.expr.__str__() + "\n" + self.lookup_table.__str__()
        return expr_str

    def __repr__(self):
        return self.__str__()

    # TODO: Implement this.
    def to(self):
        raise NotImplementedError

# Helper function to convert a tensor to a GLExpr


def GLSymbol(tensor: th.Tensor) -> GLExpr:
    symbol = sympy.Symbol(f"tensor_{id(tensor)}")
    lookup_table = {symbol: tensor}
    return GLExpr(symbol, lookup_table)


@magic_method_decorator_for_function
class GLFunction(Function):
    def __new__(cls, *args, **kwargs):
        new_args = []
        merged_lookup_table = {}
        for arg in args:
            if isinstance(arg, GLExpr):
                merged_lookup_table.update(arg.lookup_table)
                new_args.append(arg)
            elif isinstance(arg, GLFunction):
                merged_lookup_table.update(arg.lookup_table)
                new_args.append(arg)
            elif isinstance(arg, th.Tensor):
                symbol = sympy.Symbol(cls._generate_symbol_name(arg))
                new_args.append(symbol)
                merged_lookup_table[symbol] = arg
            else:
                new_args.append(arg)

        instance = super(GLFunction, cls).__new__(cls, *new_args, **kwargs)
        if not hasattr(instance, "lookup_table"):
            instance.lookup_table = merged_lookup_table
        else:
            instance.lookup_table.update(merged_lookup_table)
        return instance

    @staticmethod
    def _generate_symbol_name(tensor):
        return f"tensor_{id(tensor)}"

    @classmethod
    def _should_evalf(cls, arg):
        return -1

    def __str__(self, tabs=0):
        args = self.args
        n_tabs = '\t' * tabs
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.__str__(tabs=tabs+1))
            else:
                str_args.append(str(arg))
        n_tabs_1 = '\t' * (tabs + 1)
        str_args = f",\n{n_tabs_1}" .join(str_args)
        return f"{self.func.__name__}({str_args})"

    def string_format(self):
        args = self.args
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        return f"{self.func.__name__}({', '.join(map(str, replaced_args))})"

    def to(self, device):
        """convert the expression to cuda or cpu"""
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLFunction):
                arg = sub_expr.to(device)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = self.lookup_table[sub_expr].to(device)
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyInteger, SympyFloat)):
                arg = sub_expr
            else:
                raise ValueError(
                    f"Error while converting {sub_expr} to device {device}.")
            resolved_args.append(arg)

        new_expr = type(self)(*resolved_args)
        return new_expr

    def cuda(self):
        # convert all Tensors to numpy arrays
        device = th.device("cuda")
        expr = self.to(device)
        return expr

    def cpu(self):
        device = th.device("cpu")
        expr = self.to(device)
        return expr

    def numpy(self):
        # convert all Tensors to numpy arrays
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLFunction):
                arg = sub_expr.numpy()
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = tuple(self.lookup_table[sub_expr].cpu().numpy())
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyInteger, SympyFloat)):
                arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to Sympy.")
            resolved_args.append(arg)

        new_expr = type(self)(*resolved_args)
        return new_expr

    def sympy(self):
        # convert all Tensors to Sympy Tuples
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLFunction):
                arg = sub_expr.sympy()
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = tuple(self.lookup_table[sub_expr].cpu().numpy().tolist())
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyInteger, SympyFloat)):
                arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to Sympy.")
            resolved_args.append(arg)

        new_expr = type(self)(*resolved_args)
        return new_expr
    
    def __len__(self):
        length = 1 + sum([len(arg) for arg in self.args])
        return length
