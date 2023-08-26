# TODO: Remove reliance on sympy as it might be heavy.
class Expr(type):
    """The base class for all symbolic expressions.
    Functions are extensions of this class.
    """
    def __new__(cls, name, bases, attr):
        ...
        
    def __init__(self, *args, **kwargs):
        ...
        
    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.__class__.__name__, self.args, self.kwargs))
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               ", ".join([repr(x) for x in self.args]))

class Function(Expr):
    """The base class for all symbolic functions.
    """
    def __new__(cls, *args, **kwargs):
        cls._properties = [x for x in cls.__signature__.parameters.keys() if x != 'self']
        
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            for i, k in enumerate(args):
                setattr(self, self._properties[i], args[i])
            self.args = args
            
        for k, v in kwargs.items():
            if hasattr(self.__class__, k):
                setattr(self, k, v)

     # Use the eval method to type check the init of the function.
    @classmethod
    def eval(cls, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def args(self):
        return [getattr(self, x) for x in self._properties]
    
    @property
    def __signature__(self):
        """
        Allow Python 3's inspect.signature to give a useful signature for
        Function subclasses.
        """
        # Python 3 only, but backports (like the one in IPython) still might
        # call this.
        try:
            from inspect import signature
        except ImportError:
            return None

        # TODO: Look at nargs
        return signature(self.eval)
    