
import sympy
import inspect
import torch as th
from typing import Dict, Tuple
from sympy.core.basic import Basic
from sympy import (
    Function,
    Expr,
    Symbol,
    Tuple as SympyTuple,
    Integer as SympyInteger,
    Float as SympyFloat,
)
from sympy.logic.boolalg import Boolean as SympyBoolean
from sympy import FunctionClass as SympyFC

SYMPY_TYPES = (SympyTuple, SympyInteger, SympyFloat, SympyBoolean, SympyFC)


def magic_method_decorator(cls):
    magic_methods = [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__pow__",
    ]  # extend this list as needed

    def make_magic_method(magic):
        def method(self, other):
            base_method = getattr(Expr, magic)

            if isinstance(other, (GLExpr, GLFunction)):
                merged_lookup_table = {**self.lookup_table, **other.lookup_table}
                return GLExpr(base_method(self.expr, other.expr), merged_lookup_table)
            else:
                return GLExpr(base_method(self.expr, other), self.lookup_table)

        return method

    for magic in magic_methods:
        setattr(cls, magic, make_magic_method(magic))

    return cls


def magic_method_decorator_for_function(cls):
    magic_methods = [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__pow__",
    ]  # extend this list as needed

    def make_magic_method(magic):
        base_method = getattr(Function, magic)

        def method(self, other):
            if isinstance(other, GLExpr):
                new_expr = base_method(self, other.expr)
                merged_lookup_table = {**self.lookup_table, **other.lookup_table}
                return GLExpr(new_expr, merged_lookup_table)
            elif isinstance(other, GLFunction):
                new_expr = base_method(self, other)
                merged_lookup_table = {**self.lookup_table, **other.lookup_table}
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
    """
    GLFunction is a class that extends the functionality of a symbolic function, allowing it to work
    seamlessly with both symbolic expressions and PyTorch tensors. It serves as a bridge between
    symbolic computation (using Sympy) and tensor computation, enabling operations that combine these
    two domains.
    """

    def __new__(cls, *args, **kwargs):
        """
        Creates a new instance of GLFunction. It processes the arguments, converting any PyTorch
        tensors into symbolic variables and merging them into a lookup table for later use.
        """
        new_args = []
        merged_lookup_table = {}
        GL_TYPES = (GLFunction, GLExpr)
        for arg in args:
            if isinstance(arg, GL_TYPES):
                # merged_lookup_table.update(arg.lookup_table)
                new_args.append(arg)
            elif isinstance(arg, th.Tensor):
                symbol = sympy.Symbol(cls._generate_symbol_name(arg))
                new_args.append(symbol)
                merged_lookup_table[symbol] = arg
            else:
                new_args.append(arg)

        instance = super(GLFunction, cls).__new__(cls, *new_args, **kwargs)
        instance.lookup_table = merged_lookup_table
        # if not hasattr(instance, "lookup_table"):
        #     instance.lookup_table = merged_lookup_table
        # else:
        #     instance.lookup_table.update(merged_lookup_table)

        return instance

    def doit(self, **hints):
        """
        Executes reference operators. Can be extended to other operators as well.
        Used for simplifying expressions with PointReference Functions.
        """
        if hints.get("deep", True):
            # only "doit if not in symbolic form"
            terms = []
            for cur_term in self.args:
                if cur_term in self.lookup_table:
                    terms.append(self.lookup_table[cur_term])
                else:
                    if isinstance(cur_term, Basic):
                        terms.append(cur_term.doit(**hints))
                    else:
                        terms.append(cur_term)

            return self.func(*terms)
        else:
            return self

    @staticmethod
    def _generate_symbol_name(tensor):
        return f"tensor_{id(tensor)}"

    @classmethod
    def _should_evalf(cls, arg):
        return -1

    def pretty_print(self, tabs=0, tab_str="\t"):
        """
        Returns a formatted string representation of the function and its arguments for pretty
        printing.
        """
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.pretty_print(tabs=tabs + 1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    item = [f"{x:.3f}" for x in arg]
                    item = ", ".join(item)
                    str_args.append(f"({item})")
                else:
                    str_args.append(str(arg))
        if str_args:
            n_tabs_1 = tab_str * (tabs + 1)
            str_args = f",\n{n_tabs_1}".join(str_args)
            str_args = f"\n{n_tabs_1}" + str_args
            final = f"{self.func.__name__}({str_args})"
        else:
            final = f"{self.func.__name__}()"
        return final

    def __str__(self):
        args = self.args
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        return f"{self.func.__name__}({', '.join(map(str, replaced_args))})"

    @property
    def device(self):
        """return the device of the first tensor in the expression"""
        for arg in self.args:
            if isinstance(arg, (GLFunction, GLExpr)):
                return arg.device
            elif isinstance(arg, Symbol):
                if arg in self.lookup_table.keys():
                    return self.lookup_table[arg].device

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
            elif isinstance(sub_expr, SYMPY_TYPES):
                arg = sub_expr
            else:
                raise ValueError(
                    f"Error while converting {sub_expr} to device {device}."
                )
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

    def to_tensor(self, dtype=th.float32, device="cuda"):
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLFunction):
                arg = sub_expr.to_tensor()
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = self.lookup_table[sub_expr].to(dtype=dtype, device=device)
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyFloat)):
                arg = th.tensor(sub_expr, dtype=dtype, device=device)
            elif isinstance(sub_expr, SympyInteger):
                arg = th.tensor(sub_expr, dtype=th.int64, device=device)
            else:
                raise ValueError(f"Cannot convert {sub_expr} to Sympy.")
            resolved_args.append(arg)

        new_expr = type(self)(*resolved_args)
        return new_expr

    def gather_tensor_list(self, type_annotate=False):
        """
        Gathers a list of tensors present in the expression.
        Used for Parameter optimizing without converting form.
        """
        tensors = []
        for ind, sub_expr in enumerate(self.args):
            if isinstance(sub_expr, GLFunction):
                tensors += sub_expr.gather_tensor_list(type_annotate=type_annotate)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    if type_annotate:
                        tensors.append(
                            (self.lookup_table[sub_expr], self.__class__, ind)
                        )
                    else:
                        tensors.append(self.lookup_table[sub_expr])
        return tensors

    def _inject_tensor_list(self, tensor_list, cur_ind=0):
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLFunction):
                arg, cur_ind = sub_expr._inject_tensor_list(tensor_list, cur_ind)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = tensor_list[cur_ind]
                    cur_ind += 1
                else:
                    arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to Sympy.")
            resolved_args.append(arg)

        new_expr = type(self)(*resolved_args)
        return new_expr, cur_ind

    def inject_tensor_list(self, tensor_list):
        """
        Injects a list of tensors into the expression, using tensor occurence order to match the tensors.
        Used for Parameter optimizing without converting form.
        """
        new_expr, _ = self._inject_tensor_list(tensor_list)
        return new_expr

    @classmethod
    def eval(cls, *args, **kwargs):
        """
        TODO: To be used for type checking.
        """
        if cls._signature_1(cls, *args, **kwargs):
            return None
        else:
            class_sig = inspect.signature(cls._signature_1)
            error_message = f"Invalid arguments for the function. Here is the function signature: {str(class_sig)}"
            raise TypeError(error_message)

    def _signature_1(self, *args, **kwargs):
        # TODO: Find if type checking can be done cheaply.
        return True

    def __len__(self):
        length = 1
        for arg in self.args:
            if isinstance(arg, (GLFunction, GLExpr)):
                length += len(arg)
            else:
                length += 0
        return length


class PrimitiveSpec(GLFunction):
    """Base Class nodes (or expression) in the compiled expressions."""

    @classmethod
    def eval(cls, prim_type: type, shift: int):
        return None
