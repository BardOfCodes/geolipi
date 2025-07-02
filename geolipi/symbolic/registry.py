# global_registry.py
SYMBOL_REGISTRY = {}

def register_symbol(cls):
    SYMBOL_REGISTRY[cls.__name__] = cls
    return cls