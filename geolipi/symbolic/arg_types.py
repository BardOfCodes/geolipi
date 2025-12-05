ALIASES = {
  "vec2": "Vector[2]",
  "vec3": "Vector[3]",
  "vec4": "Vector[4]",
  "point2": "Vector[2]",
  "point3": "Vector[3]",
  "color3": "Vector[3]",
  "color4": "Vector[4]",
  "optional": "Optional",
  "list": "List",
  "tuple": "Tuple",
  "dict": "Dict",
  "union": "Union",
  "expr": "Expr",
  "node": "Node",
  "float2": "Vector[2]",
  "float3": "Vector[3]",
  "float4": "Vector[4]",
}


def normalize_type(type_str: str):
    """Simple normalizer: split on '|' and return a tuple of stripped strings."""
    if not isinstance(type_str, str):
        return (str(type_str),)
    parts = [p.strip() for p in type_str.split("|")]
    # drop empties
    parts = [p for p in parts if p]
    return tuple(parts)


def validate_spec(spec: dict) -> list:
    """Validate a default_spec dict structure. Returns list of error strings."""
    errors = []
    if not isinstance(spec, dict):
        return ["spec-not-dict"]
    for key, entry in spec.items():
        if not isinstance(key, str) or not key:
            errors.append(f"invalid-key:{key}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"invalid-entry:{key}")
            continue
        t = entry.get("type")
        if not isinstance(t, str) or not t:
            errors.append(f"missing-or-invalid-type:{key}")
            continue
        # basic normalize to ensure parsability
        _ = normalize_type(t)
        # flags must be booleans if present
        for flag in ("optional", "variadic"):
            if flag in entry and not isinstance(entry[flag], bool):
                errors.append(f"invalid-flag-{flag}:{key}")
    return errors


def validate_module_specs(modules: list) -> dict:
    """Validate all GLFunction.default_spec() in provided modules; returns {ClassName: [errors]}"""
    import inspect
    from geolipi.symbolic.base import GLFunction
    results: dict = {}
    for mod in modules:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, GLFunction):
                continue
            if not getattr(obj, "__module__", "").startswith(mod.__name__):
                continue
            try:
                spec = obj.default_spec()
            except Exception as e:
                results[obj.__name__] = [f"default-spec-error:{type(e).__name__}"]
                continue
            errs = validate_spec(spec)
            if errs:
                results[obj.__name__] = errs
    return results
