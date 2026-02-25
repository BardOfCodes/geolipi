import importlib
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_geolipi_imports() -> None:
    """Basic import smoke test for the top-level package."""
    mod = importlib.import_module("geolipi")
    assert hasattr(mod, "__version__")


def test_symbolic_submodule_imports() -> None:
    """Ensure the symbolic submodule is importable."""
    try:
        symbolic = importlib.import_module("geolipi.symbolic")
    except OSError:
        # In constrained environments torch may not be fully usable; skip in that case.
        return

    assert hasattr(symbolic, "__all__") or hasattr(symbolic, "GLFunction")

