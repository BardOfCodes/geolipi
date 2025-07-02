import inspect
import sympy as sp
import torch as th
from typing import Dict, Tuple, Union
from sympy.core.basic import Basic
from sympy import (
    Function,
    Expr,
    Symbol,
    Tuple as SympyTuple,
    Integer as SympyInteger,
    Float as SympyFloat,
    Pow,
)
from sympy.logic.boolalg import Boolean as SympyBoolean
from sympy import FunctionClass as SympyFC
from sympy.core.operations import AssocOp

SYMPY_TYPES = (SympyTuple, SympyInteger, SympyFloat, SympyBoolean, SympyFC)
SYMPY_ARG_TYPES = (Symbol, SympyTuple, SympyInteger, SympyFloat, SympyBoolean)

# Shared magic methods to be supported by both GLExpr and GLFunction
MAGIC_METHODS = [
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__pow__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rtruediv__",
    "__rpow__",
]


def magic_method_decorator(base_class=Expr):
    """
    Decorates a class with binary magic methods for symbolic combination.
    Works for both GLExpr and GLFunction.

    Args:
        base_class: The symbolic class whose methods to use (Expr or Function).
    """
    def decorator(cls):
        def make_magic_method(magic):
            base_method = getattr(base_class, magic)

            def method(self, other):
                if isinstance(other, GLExpr):
                    expr_other = other.expr
                    lookup_other = other.lookup_table
                elif isinstance(other, GLFunction):
                    expr_other = other
                    lookup_other = other.lookup_table
                else:
                    expr_other = other
                    lookup_other = {}

                merged_lookup = {**self.lookup_table, **lookup_other}
                new_expr = base_method(getattr(self, "expr", self), expr_other)
                return GLExpr(new_expr, merged_lookup)

            return method

        for magic in MAGIC_METHODS:
            setattr(cls, magic, make_magic_method(magic))

        return cls
    
    return decorator


@magic_method_decorator(base_class=Expr)
class GLExpr:
    __sympy__ = True  # To avoid sympifying the expression which raises errors.

    def __init__(self, expr: Expr, lookup_table: Dict[sp.Symbol, th.Tensor] | None = None):
        self.expr = expr
        self.lookup_table = lookup_table or {}
        self.func = expr.func

    @property
    def args(self):
        return self.expr.args

    def __str__(self):
        # Build a pattern that matches any of the symbols in the lookup table
        expr_str = self.expr.__str__() + "\n" + self.lookup_table.__str__()
        return expr_str

    def pretty_print(self, tabs=0, tab_str="\t"):
        """
        Returns a formatted string representation of the function and its arguments for pretty
        printing.
        """
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = [self.lookup_table.get(arg, arg) if isinstance(
            arg, Symbol) else arg for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(
                    arg.pretty_print(tabs=tabs + 1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    # Can be a tuple of tuple But can't be an empty tuple
                    if arg and isinstance(arg[0], SympyTuple):  # type: ignore
                        item = [
                            f"({', '.join([f'{y:.3f}' for y in x])})" for x in arg]
                    elif arg:
                        item = [f"{x:.3f}" for x in arg]
                    else:
                        item = []
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

    def to(self):
        raise NotImplementedError
    def is_zero(self):
        return False
    def tensor(self):
        eval_args = []
        merged_lookup_table = {}
        for arg in self.args:
            if isinstance(arg, (GLExpr, GLFunction)):
                eval_arg = arg.tensor()
                merged_lookup_table.update(eval_arg.lookup_table)
            elif isinstance(arg, (AssocOp, Pow)):
                evaluated_args = []
                for under_arg in arg.args:
                    if isinstance(under_arg, (GLExpr, GLFunction)):
                        evaluated_args.append(under_arg.tensor())
                    else:
                        evaluated_args.append(under_arg)
                op = arg.__class__
                eval_arg = op(*evaluated_args)
            else:
                eval_arg = arg
            eval_args.append(eval_arg)
        new_expr = self.func(*eval_args)
        gl_expr = GLExpr(new_expr, merged_lookup_table)
        return gl_expr

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
            if isinstance(sub_expr, (GLExpr, GLFunction)):
                arg, cur_ind = sub_expr._inject_tensor_list(tensor_list, cur_ind)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = tensor_list[cur_ind]
                    cur_ind += 1
                else:
                    arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
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

    @staticmethod
    def _custom_srepr(expr, symbol_tensor_map=None, packages=None):
        if symbol_tensor_map is None:
            symbol_tensor_map = {}
        if packages is None:
            packages = set()
        if isinstance(expr, GLFunction):
            module = expr.__class__.__module__
            package = module.split('.')[-2] if '.' in module else module
            packages.add(package)
            class_name = expr.__class__.__name__
            args_repr = [GLExpr._custom_srepr(arg, symbol_tensor_map, packages) for arg in expr.args]
            return f"{package}.{class_name}({', '.join(args_repr)})"
        elif isinstance(expr, GLExpr):
            return GLExpr._custom_srepr(expr.expr, symbol_tensor_map, packages)
        elif isinstance(expr, th.Tensor):
            symbol = sp.Symbol(f"tensor_{id(expr)}")
            symbol_tensor_map[str(symbol)] = expr
            return str(symbol)
        elif isinstance(expr, Symbol):
            return str(expr)
        elif isinstance(expr, (tuple, list)):
            return f"({', '.join(GLExpr._custom_srepr(x, symbol_tensor_map, packages) for x in expr)})"
        else:
            return repr(expr)

    def __getstate__(self):
        symbol_tensor_map = {}
        packages = set()
        expr_str = self._custom_srepr(self, symbol_tensor_map, packages)
        return {
            "expr_str": expr_str,
            "symbol_tensor_map": symbol_tensor_map,
            "packages": list(packages),
        }

    def __setstate__(self, state):
        expr_str = state["expr_str"]
        symbol_tensor_map = state["symbol_tensor_map"]
        packages = state.get("packages", [])
        namespace = {}
        for pkg in packages:
            try:
                namespace[pkg] = __import__(pkg)
            except ImportError:
                pass  # Optionally, raise or warn here
        namespace.update(symbol_tensor_map)
        expr = eval(expr_str, namespace)
        self.__dict__.update(expr.__dict__)



@magic_method_decorator(base_class=Function)
class GLFunction(Function):
    """
    GLFunction is a class that extends the functionality of a symbolic function, allowing it to work
    seamlessly with both symbolic expressions and PyTorch tensors. It serves as a bridge between
    symbolic computation (using Sympy) and tensor computation, enabling operations that combine these
    two domains.
    """

    lookup_table: Dict[sp.Symbol, th.Tensor]

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
                symbol = sp.Symbol(cls._generate_symbol_name(arg))
                new_args.append(symbol)
                merged_lookup_table[symbol] = arg
            else:
                new_args.append(arg)

        instance = super(GLFunction, cls).__new__(cls, *new_args, **kwargs)
        assert isinstance(instance, GLFunction), "Instance must be a GLFunction"
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
                if isinstance(cur_term, Symbol) and cur_term in self.lookup_table:
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
        replaced_args = [self.lookup_table.get(arg, arg) if isinstance(
            arg, Symbol) else arg for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.pretty_print(tabs=tabs + 1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    # Can be a tuple of tuple But can't be an empty tuple
                    if arg and isinstance(arg[0], SympyTuple):
                        item = [f"({', '.join([f'{y:.3f}' for y in x])})" for x in arg]
                    elif arg:
                        item = [f"{x:.3f}" for x in arg]
                    else:
                        item = []
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
        replaced_args = [self.lookup_table.get(arg, arg) if isinstance(
            arg, Symbol) else arg for arg in args]
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
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
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
                    arg = self.lookup_table[sub_expr].detach().cpu().numpy().tolist()
                    if not isinstance(arg, list):
                        arg = [arg, ]
                    arg = to_nested_tuple(arg)
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyInteger, SympyFloat)):
                arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
            resolved_args.append(arg)

        new_expr = type(self)(*resolved_args)
        return new_expr

    def tensor(self, dtype=th.float32, device="cuda"):
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, (GLExpr, GLFunction)):
                arg = sub_expr.tensor(dtype=dtype, device=device)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = self.lookup_table[sub_expr].to(dtype=dtype, device=device)
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (AssocOp, Pow)):
                evaluated_args = []
                for under_arg in sub_expr.args:
                    if isinstance(under_arg, (GLExpr, GLFunction)):
                        evaluated_args.append(under_arg.tensor())
                    else:
                        evaluated_args.append(under_arg)
                op = sub_expr.__class__
                arg = op(*evaluated_args)
            elif isinstance(sub_expr, (SympyTuple, SympyFloat)):
                arg = th.tensor(sub_expr, dtype=dtype, device=device)
            elif isinstance(sub_expr, SympyInteger):
                arg = th.tensor(sub_expr, dtype=th.int64, device=device)
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
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
            if isinstance(sub_expr, (GLExpr, GLFunction)):
                arg, cur_ind = sub_expr._inject_tensor_list(tensor_list, cur_ind)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = tensor_list[cur_ind]
                    cur_ind += 1
                else:
                    arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
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
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            class_sig = inspect.signature(cls._signature_1)
            error_message = f"Invalid arguments for the function. Here is the function signature: {str(class_sig)}"
            raise TypeError(error_message)

    @classmethod
    def _signature_1(cls, *args, **kwargs):
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

    @staticmethod
    def _custom_srepr(expr, symbol_tensor_map=None, packages=None):
        return GLExpr._custom_srepr(expr, symbol_tensor_map, packages)

    def __getstate__(self):
        symbol_tensor_map = {}
        packages = set()
        expr_str = self._custom_srepr(self, symbol_tensor_map, packages)
        return {
            "expr_str": expr_str,
            "symbol_tensor_map": symbol_tensor_map,
            "packages": list(packages),
        }

    def __setstate__(self, state):
        expr_str = state["expr_str"]
        symbol_tensor_map = state["symbol_tensor_map"]
        packages = state.get("packages", [])
        namespace = {}
        for pkg in packages:
            try:
                namespace[pkg] = __import__(pkg)
            except ImportError:
                pass  # Optionally, raise or warn here
        namespace.update(symbol_tensor_map)
        expr = eval(expr_str, namespace)
        self.__dict__.update(expr.__dict__)


class PrimitiveSpec(GLFunction):
    """Base Class nodes (or expression) in the compiled expressions."""

    @classmethod
    def eval(cls, prim_type: type, shift: int):
        return None


def to_nested_tuple(obj):
    """Recursively convert a nested list (or scalar) into nested tuples."""
    if isinstance(obj, list):
        return tuple(to_nested_tuple(x) for x in obj)
    else:
        # If it's not a list, it's typically a scalar (int/float)
        return obj

