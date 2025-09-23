import inspect

def list_missing_default_specs(include_sysl: bool = True):
    missing = []
    try:
        import geolipi.symbolic as gls
        modules = [gls]
    except Exception:
        modules = []
    if include_sysl:
        try:
            import sysl.sysl.symbolic as sysl_sym
            modules.append(sysl_sym)
        except Exception:
            pass

    for mod in modules:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            # Only classes defined in these modules' packages
            if not getattr(obj, "__module__", "").startswith(mod.__name__):
                continue
            # Must be a GLFunction descendant
            try:
                from geolipi.symbolic.base import GLFunction
            except Exception:
                continue
            if not issubclass(obj, GLFunction):
                continue
            # Skip abstract bases from these modules (heuristic)
            if name in {"GLFunction", "GLExpr", "GLBase"}:
                continue
            # Check default_spec
            try:
                spec = obj.default_spec()
                if not isinstance(spec, dict):
                    missing.append((obj.__module__, name, "spec-not-dict"))
            except NotImplementedError:
                missing.append((obj.__module__, name, "missing-default-spec"))
            except Exception as e:
                missing.append((obj.__module__, name, f"error: {type(e).__name__}"))

    return missing


