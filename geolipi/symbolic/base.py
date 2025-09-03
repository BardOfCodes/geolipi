import inspect
import sympy as sp
import torch as th
import _pickle as cPickle
from abc import abstractmethod
from typing import Dict, Tuple, Any, List, Callable
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
from .registry import SYMBOL_REGISTRY

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

INFIX_OPERATORS = {
    sp.Add: "+",
    sp.Mul: "*",
    sp.Pow: "**",
}

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
                elif isinstance(other, GLFunction):
                    expr_other = other
                elif isinstance(other, (SympyTuple, th.Tensor)):
                    import geolipi.symbolic as gls
                    expr_other = gls.Param(other)
                elif isinstance(other, (int, float, SympyFloat, th.Tensor)):
                    import geolipi.symbolic as gls
                    expr_other = gls.Param((float(other),))
                else:
                    print(f"other: {other} if else {type(other)}")
                    expr_other = other

                # merged_lookup = {**self.lookup_table, **lookup_other}
                merged_lookup = {}
                new_expr = base_method(getattr(self, "expr", self), expr_other)
                return GLExpr(new_expr, merged_lookup)

            return method

        for magic in MAGIC_METHODS:
            setattr(cls, magic, make_magic_method(magic))

        return cls
    
    return decorator

class GLBase:
    """
    Container for some shared functions between GLExpr and GLFunction.
    1. Gathering and Injecting tensor list.
    2. Conversion to sympy / torch / cpu / cuda.
    3. Pretty printing, __str__ and __repr__ and pickling.
    """
    args: Tuple[sp.Basic, ...]
    lookup_table: Dict[sp.Symbol, th.Tensor]
    func: Any

    def gather_tensor_list(self, type_annotate: bool =False, index_annotate: bool =False,
            selected_classes: Tuple[type, ...] | None = None,) -> List[th.Tensor] | List[Tuple[th.Tensor, type, int]]:
        if selected_classes is None:
            selected_classes = (GLBase, )
        tensor_list, ind = self._gather_tensor_list(selected_classes=selected_classes, 
        type_annotate=type_annotate, index_annotate=index_annotate, cur_ind=0)
        return tensor_list
    
    def _gather_tensor_list(self, selected_classes: Tuple[type, ...], type_annotate: bool =False, index_annotate: bool =False, cur_ind: int =0):
        """
        Gathers a list of tensors present in the expression.
        Used for Parameter optimizing without converting form.
        """
        tensors = []
        for local_ind, sub_expr in enumerate(self.args):
            if isinstance(sub_expr, GLBase):
                new_tensors, cur_ind = sub_expr._gather_tensor_list(selected_classes=selected_classes, 
                type_annotate=type_annotate, index_annotate=index_annotate, cur_ind=cur_ind)
                tensors += new_tensors
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    if not isinstance(self, selected_classes):
                        continue
                    if type_annotate:
                        if index_annotate:
                            annotation = (self.lookup_table[sub_expr], self.__class__, cur_ind, local_ind)
                        else:
                            annotation = (self.lookup_table[sub_expr], self.__class__, cur_ind)
                        tensors.append(annotation)
                    else:
                        tensors.append(self.lookup_table[sub_expr])
                    cur_ind += 1
        return tensors, cur_ind
    
    def recursive_transform(self, transform_map: Dict[type, Callable[['GLBase'], 'GLBase']]) -> 'GLBase':
        """
        Recursively transforms the expression using a mapping of types to transformation functions.
        
        Args:
            transform_map: Dict mapping expression types to transformation functions.
                         Each function takes an expression and returns a transformed expression.
                         
        Returns:
            A new expression with transformations applied recursively.
        """
        # First, recursively transform all arguments
        new_args = []
        for arg in self.args:
            if isinstance(arg, GLBase):
                # Recursively transform GLBase arguments
                transformed_arg = arg.recursive_transform(transform_map)
                new_args.append(transformed_arg)
            else:
                # Keep non-GLBase arguments as is
                new_args.append(arg)
        
        # Rebuild the expression with transformed arguments
        rebuilt_expr = self.rebuild_expr(new_args)
        
        # Apply transformation to the rebuilt expression if there's a matching type
        expr_type = type(rebuilt_expr)
        if expr_type in transform_map:
            return transform_map[expr_type](rebuilt_expr)
        else:
            return rebuilt_expr
    

    def rebuild_expr(self, resolved_args: List[sp.Basic]):
        """
        Rebuilds the expression from the resolved arguments.
        """
        if isinstance(self, GLFunction):
            new_expr = type(self)(*resolved_args)
        elif isinstance(self, GLExpr):
            under_expr = self.expr.func(*resolved_args)
            new_expr = GLExpr(under_expr, self.lookup_table.copy())
        elif isinstance(self, (AssocOp, Pow)):
            # Will this come?
            op = self.__class__
            under_expr = op(*resolved_args)
            new_expr = GLExpr(under_expr)
        else:
            # Fallback for other types
            new_expr = type(self)(*resolved_args)
        return new_expr

    def _inject_tensor_list(self, tensor_list: List[Tuple[th.Tensor, int]], cur_ind: int =0):
        resolved_args = []
        valid_inds = [x[1] for x in tensor_list]
        for sub_expr in self.args:
            if isinstance(sub_expr, GLBase):
                arg, cur_ind = sub_expr._inject_tensor_list(tensor_list, cur_ind)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    if cur_ind in valid_inds:
                        arg = tensor_list[valid_inds.index(cur_ind)][0]
                    else:
                        arg = sub_expr
                    cur_ind += 1
                else:
                    arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
            resolved_args.append(arg)

        new_expr = self.rebuild_expr(resolved_args)
        return new_expr, cur_ind

    def inject_tensor_list(self, tensor_list: List[th.Tensor] | List[Tuple[th.Tensor, int]]):
        """
        Injects a list of tensors into the expression, using tensor occurrence order to match the tensors.
        Used for Parameter optimizing without converting form.
        """
        if len(tensor_list) == 0:
            return self
        
        if isinstance(tensor_list[0], th.Tensor):
            # tensor_list is List[th.Tensor]
            inner_tensor_list: List[Tuple[th.Tensor, int]] = [(x, i) for i, x in enumerate(tensor_list)]  # type: ignore
        elif isinstance(tensor_list[0], tuple) and len(tensor_list[0]) == 2 and isinstance(tensor_list[0][0], th.Tensor):
            # tensor_list is List[Tuple[th.Tensor, int]]
            inner_tensor_list = tensor_list  # type: ignore
        else:
            raise ValueError(f"Invalid tensor list: {tensor_list}")
        new_expr, _ = self._inject_tensor_list(inner_tensor_list)
        return new_expr

    def get_varnamed_expr(self, cur_ind = None):
        """
        Why only the ones in the lookup table?
        During Compilation we treating the remaining, i.e. Symbols as constants. 
        Other option -> 
        """
        if cur_ind is None:
            cur_ind = 0
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLBase):
                arg, cur_ind = sub_expr.get_varnamed_expr(cur_ind)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = sp.Symbol(f"var_{cur_ind}")
                    cur_ind += 1
                else:
                    arg = sub_expr
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
            resolved_args.append(arg)

        new_expr = self.rebuild_expr(resolved_args)
        return new_expr, cur_ind
    
    def _get_varnamed_expr(self, cur_ind = None, var_map = None, prefix = "var", exclude_uniforms = False):
        """
        TBD: Support Uniform Exclusion. (Keep them as is I guess).
        """
        if cur_ind is None:
            cur_ind = 0
        if var_map is None:
            var_map = {}
        resolved_args = []
        import geolipi.symbolic as gls
        for sub_expr in self.args:
            if isinstance(sub_expr, GLBase):
                if exclude_uniforms and isinstance(sub_expr, gls.UniformVariable):
                    arg = sub_expr
                else:
                    arg, cur_ind, var_map = sub_expr._get_varnamed_expr(cur_ind, var_map, prefix, exclude_uniforms)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    varname = f"{prefix}_{cur_ind}"
                    arg = sp.Symbol(varname)
                    var_map[varname] = sub_expr
                    cur_ind += 1
                else:
                    # These are treated as "cosntants" - Some Issues with this.
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyInteger, SympyFloat)):
                varname = f"{prefix}_{cur_ind}"
                arg = sp.Symbol(varname)
                var_map[varname] = sub_expr
                cur_ind += 1
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
            resolved_args.append(arg)

        new_expr = self.rebuild_expr(resolved_args)
        return new_expr, cur_ind, var_map

    def tensor(self, dtype=th.float32, device="cuda", restrict_int: bool = False):
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLBase):
                arg = sub_expr.tensor(dtype=dtype, device=device, restrict_int=restrict_int)
            elif isinstance(sub_expr, Symbol):
                if sub_expr in self.lookup_table.keys():
                    arg = self.lookup_table[sub_expr].to(dtype=dtype, device=device)
                else:
                    arg = sub_expr
            elif isinstance(sub_expr, (SympyTuple, SympyFloat, SympyInteger)):
                if restrict_int:
                    if isinstance(sub_expr, SympyInteger):
                        dtype = th.int64
                    elif isinstance(sub_expr, SympyTuple) and isinstance(sub_expr[0], SympyInteger):
                        dtype = th.int64
                arg = th.tensor(sub_expr, dtype=dtype, device=device)
            else:
                raise ValueError(f"Cannot convert {sub_expr} to sp.")
            resolved_args.append(arg)
        
        final_expr = self.rebuild_expr(resolved_args)
        return final_expr

    def sympy(self):
        # convert all Tensors to Sympy Tuples
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLBase):
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
        final_expr = self.rebuild_expr(resolved_args)
        return final_expr
        
    def numpy(self):
        """ Deprecated """
        raise NotImplementedError("Numpy is deprecated. Use tensor instead.")
         
         
    def to(self, device: str | th.device):
        """convert the expression to cuda or cpu"""
        resolved_args = []
        for sub_expr in self.args:
            if isinstance(sub_expr, GLBase):
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

        final_expr = self.rebuild_expr(resolved_args)
        return final_expr

    def cuda(self, device=None):
        """
        Moves all tensors in the expression to a CUDA device.

        Args:
            device (int | str | torch.device | None): 
                - None → use current CUDA device
                - int → cuda:<index>
                - str → e.g. 'cuda:1'
                - torch.device → used directly

        Returns:
            A new expression on the specified CUDA device.
        """
        if device is None:
            device = th.device("cuda")
        elif isinstance(device, int):
            device = th.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = th.device(device)
        elif not isinstance(device, th.device):
            raise TypeError(f"Invalid device argument: {device}")

        return self.to(device)

    def cpu(self):
        device = th.device("cpu")
        expr = self.to(device)
        return expr


    @property
    def device(self) -> str | th.device | None:
        """
        Returns:
            - torch.device if all tensors are on the same device
            - "MIX" if tensors are on multiple devices
            - None if no tensor found
        """
        devices = set()

        def collect_devices(arg, lookup):
            if isinstance(arg, GLBase):
                devices.add(arg.device)
            elif isinstance(arg, Symbol) and arg in lookup:
                tensor = lookup[arg]
                devices.add(tensor.device)

        for arg in self.args:
            collect_devices(arg, getattr(self, "lookup_table", {}))

        devices = {d for d in devices if d is not None}

        if not devices:
            return None
        if len(devices) == 1:
            return next(iter(devices))
        return "MIX"
  
    @property
    def paramtype(self) -> str | None:
        """
        Returns:
            - torch if all params are torch tensors
            - sympy if all params are sympy tuples
            - "MIX" if tensors are on multiple devices
            - None if no param found
        """
        paramtypes = set()

        def collect_paramtypes(arg, lookup):
            if isinstance(arg, GLBase):
                paramtypes.add(arg.paramtype)
            elif isinstance(arg, (SympyTuple, SympyInteger, SympyFloat)):
                paramtypes.add("sympy")
            elif isinstance(arg, Symbol) and arg in lookup:
                paramtypes.add("torch")
                # Others - Symbols not in lookup table are okay.

        for arg in self.args:
            collect_paramtypes(arg, getattr(self, "lookup_table", {}))

        paramtypes = {d for d in paramtypes if d is not None}

        if not paramtypes:
            return None
        if len(paramtypes) == 1:
            return next(iter(paramtypes))   
        return "MIX"


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
            if isinstance(arg, GLBase):
                str_args.append(arg.pretty_print(tabs=tabs + 1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    # Can be a tuple of tuple But can't be an empty tuple
                    if arg and len(arg) > 0 and isinstance(arg[0], SympyTuple):
                        item = [f"({', '.join([f'{y:.3f}' for y in x])})" for x in arg]
                    elif arg and len(arg) > 0:
                        item = [f"{x:.3f}" for x in arg]
                    else:
                        item = []
                    item = ", ".join(item)
                    str_args.append(f"({item})")
                elif isinstance(arg, th.Tensor):
                    # Handle tensor arguments
                    if arg.numel() == 1:
                        str_args.append(f"{arg.item():.3f}")
                    else:
                        str_args.append(f"tensor({list(arg.shape)})")
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
    
    def __str__(self, order="INFIX", with_lookup=True,*args, **kwargs):
        raise NotImplementedError

    def state(self):
        tensor_lookup = {}
        for arg in self.args:
            if isinstance(arg, GLBase):
                arg_state = arg.state()
                arg_lookup = arg_state["symbol_tensor_map"]
                tensor_lookup.update(arg_lookup)
            elif isinstance(arg, sp.Symbol):
                if arg in self.lookup_table:
                    tensor_lookup[arg] = self.lookup_table[arg]
        expr_str = self.__str__(with_lookup=False)
        state = {
            "expr_str": expr_str,
            "symbol_tensor_map": tensor_lookup,
        }
        return state
        
    @classmethod
    def from_state(cls, state):
        expr_str = state["expr_str"]
        tensor_lookup = state["symbol_tensor_map"]

        # Step 1: evaluate the symbolic expression (we assume all classes/functions are available)
        global_dict = {'sp': sp, 'th': th, 'torch': th}
        global_dict.update(SYMBOL_REGISTRY)
        expr = eval(expr_str, global_dict)

        # Step 2: recursively inject tensors wherever the symbol appears
        def inject(expr):
            if isinstance(expr, GLFunction):
                new_args = []
                for arg in expr.args:
                    if isinstance(arg, sp.Symbol) and arg in tensor_lookup:
                        new_args.append(tensor_lookup[arg])
                    elif isinstance(arg, GLBase):
                        new_args.append(inject(arg))
                    else:
                        new_args.append(arg)
                return type(expr)(*new_args)

            elif isinstance(expr, GLExpr):
                # Rebuild recursively
                args = [
                    inject(arg) if isinstance(arg, (GLBase, sp.Basic)) else arg
                    for arg in expr.expr.args
                ]
                new_expr = expr.expr.func(*args)
                return GLExpr(new_expr)

            elif isinstance(expr, sp.Basic):
                new_args = [
                    inject(arg) if isinstance(arg, (GLBase, sp.Basic)) else arg
                    for arg in expr.args
                ]
                return expr.func(*new_args)

            else:
                return expr

        rebuilt_expr = inject(expr)

        return rebuilt_expr

    def save_state(self, filename: str):
        # SAVE SAVE - Convert to cpu tensors
        self.to("cpu")
        state = self.state()
        with open(filename, "wb") as f:
            cPickle.dump(state, f)

    @classmethod
    def load_state(cls, filename: str):
        with open(filename, "rb") as f:
            state = cPickle.load(f)
        return cls.from_state(state)

    @classmethod
    def _should_evalf(cls, arg):
        return -1

    def is_zero(self):
        return False
        
    def __len__(self):
        length = 1
        for arg in self.args:
            if isinstance(arg, GLBase):
                length += len(arg)
            else:
                length += 0
        return length


@magic_method_decorator(base_class=Expr)
class GLExpr(GLBase):
    __sympy__ = True  # To avoid sympifying the expression which raises errors.

    def __init__(self, expr: Expr, lookup_table: Dict[sp.Symbol, th.Tensor] | None = None):
        self.expr = expr
        if lookup_table is None:
            self.lookup_table = {}
        else:
            self.lookup_table = lookup_table
        self.func = expr.func
        
    @property
    def args(self):
        return self.expr.args

    def __str__(self, order="INFIX", with_lookup: bool = True):
        arg_exprs = []
        for arg in self.args:
            if isinstance(arg, GLBase):
                arg_exprs.append(arg.__str__(order=order, with_lookup=with_lookup))  # type: ignore
            else:
                arg_exprs.append(str(arg))
        op = INFIX_OPERATORS[self.expr.func]
        if order == "INFIX":
            return f"({f' {op} '.join(arg_exprs)})"
        elif order == "PREFIX":
            return f"({op}, {', '.join(arg_exprs)})"
        elif order == "POSTFIX":
            return f"({', '.join(arg_exprs)}, {op})"
        else:
            raise ValueError(f"Invalid order: {order}")


@magic_method_decorator(base_class=Function)
class GLFunction(Function, GLBase):
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
        for arg in args:
            if isinstance(arg, th.Tensor):
                symbol = sp.Symbol(cls._generate_symbol_name(arg))
                new_args.append(symbol)
                merged_lookup_table[symbol] = arg
            else:
                new_args.append(arg)

        instance = super(GLFunction, cls).__new__(cls, *new_args, **kwargs)
        assert isinstance(instance, GLFunction), "Instance must be a GLFunction"
        instance.lookup_table = merged_lookup_table

        return instance

    def __str__(self, order="INFIX", with_lookup: bool = True):
        args = self.args
        if with_lookup:
            replaced_args = [self.lookup_table.get(arg, arg) if isinstance(
                arg, Symbol) else arg for arg in args]
        else:
            replaced_args = [arg for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, GLBase):
                str_args.append(arg.__str__(order=order, with_lookup=with_lookup))  # type: ignore
            else:
                if isinstance(arg, sp.Symbol):
                    str_args.append(f"'{arg.name}'")
                else:
                    str_args.append(str(arg))
        return f"{self.func.__name__}({', '.join(str_args)})"


    # TODO: In line resolution of parameters.
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

    ##: TODO: This is for type checking. TBD.
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

