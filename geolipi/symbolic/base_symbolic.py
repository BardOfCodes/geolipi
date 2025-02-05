
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
SYMPY_ARG_TYPES = (Symbol, SympyTuple, SympyInteger, SympyFloat, SympyBoolean)


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

            if isinstance(other, GLExpr):
                merged_lookup_table = {**self.lookup_table, **other.lookup_table}
                return GLExpr(base_method(self.expr, other.expr), merged_lookup_table)
            if isinstance(other, GLFunction):
                merged_lookup_table = {**self.lookup_table, **other.lookup_table}
                return GLExpr(base_method(self.expr, other), merged_lookup_table)
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
        self.func = expr.func

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
    def is_zero(self):
        return False
    def tensor(self):
        eval_args = []
        merged_lookup_table = {}
        for arg in self.args:
            if isinstance(arg, (GLExpr, GLFunction)):
                eval_arg = arg.tensor()
                merged_lookup_table.update(eval_arg.lookup_table)
            elif isinstance(arg, (sympy.core.operations.AssocOp, sympy.core.power.Pow)):
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
                    arg = self.lookup_table[sub_expr].cpu().numpy().tolist()
                    if not isinstance(arg, list):
                        arg = [arg, ]
                    arg = tuple(arg)
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyInteger, SympyFloat)):
                arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to Sympy.")
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
            elif isinstance(sub_expr, (sympy.core.operations.AssocOp, sympy.core.power.Pow)):
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
            if isinstance(sub_expr, (GLExpr, GLFunction)):
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

    # for pickle
    # def __getnewargs__(self):
    #     args = []
    #     for arg in args:
    #         if arg in self.lookup_table:
    #             arg = self.lookup_table[arg]
    #         # elif isinstance(arg, (GLFunction, GLExpr)):
    #         #     args.append(arg.__getnewargs__())
    #         else:
    #             args.append(arg)
    #     return tuple(args)
    
    # Could be useful ref. in something else...
    # TODO: What was the error?
    # def __getstate__(self):
    #     expr_str = str(self)
    #     return expr_str
    
    #     tensor_dict = {}
    #     expression_list = []
    #     n_ops = []
    #     stack = [self]
    #     # Do a depth first traversal
    #     while stack:
    #         cur_expr = stack.pop()
    #         if isinstance(cur_expr, GLFunction):
    #             cur_token = cur_expr.func
    #             n_ops.append(len(cur_expr.args))
    #             expression_list.append(cur_token)
    #             stack.extend(cur_expr.args[::-1])
    #             tensor_dict.update(cur_expr.lookup_table)
    #         else:
    #             expression_list.append(cur_expr)
                
    #     state = {
    #         "tensor_args": tensor_dict,
    #         "n_ops": n_ops,
    #         "expression_list": expression_list,
    #     }
    #     return state

    # def __setstate__(self, state):
    #     from geolipi.symbolic import get_cmd_mapper
    #     expression = eval(state, get_cmd_mapper())
    #     return expression
    #     tensor_args = state["tensor_args"]
    #     n_ops = state["n_ops"]
    #     expression_list = state["expression_list"]
    #     arg_stack = []
    #     while(expression_list):
    #         cur_expr = expression_list.pop()
    #         if isinstance(cur_expr, SYMPY_ARG_TYPES):
    #             arg_stack.append(cur_expr)
    #         else:
    #             # Need this if condition or not?
    #             # if issubclass(cur_expr, GLFunction):
    #             n_op = n_ops.pop()
    #             cur_args = []
    #             for _ in range(n_op):
    #                 arg = arg_stack.pop()
    #                 if arg in tensor_args:
    #                     cur_args.append(tensor_args[arg])
    #                 else:
    #                     cur_args.append(arg)
    #             arg_stack.append(cur_expr(*cur_args))
    #     new_expr = arg_stack[0]
    #     return new_expr

class PrimitiveSpec(GLFunction):
    """Base Class nodes (or expression) in the compiled expressions."""

    @classmethod
    def eval(cls, prim_type: type, shift: int):
        return None
