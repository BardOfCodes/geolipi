# torch_compute

This module provides PyTorch-based evaluation and compilation of GeoLIPI symbolic expressions. It serves as the execution engine that converts symbolic geometric expressions into concrete signed distance fields (SDFs) and colored canvases.

## Core Functions

### Primary Evaluation

#### `recursive_evaluate(expression, sketcher, **kwargs)`
**Main function for evaluating expressions** - Use this for most cases.

- **Purpose**: Recursively evaluates any GeoLIPI expression to generate SDFs or colored canvases
- **Features**: Full feature support including higher-order primitives, all modifiers, and color operations
- **Performance**: Slower but comprehensive - handles all GeoLIPI operations
- **Use when**: You need complete feature support or are working with complex expressions

```python
from geolipi.torch_compute import recursive_evaluate, Sketcher
import geolipi.symbolic as gls

# Create expression and sketcher
expression = gls.Union(gls.Circle2D((0.5,)), gls.Box2D((0.3, 0.4)))
sketcher = Sketcher(device="cuda", resolution=512, n_dims=2)

# Evaluate expression
result = recursive_evaluate(expression, sketcher)
```

### Compilation and Optimization

#### `unroll_expression(expression, sketcher, **kwargs)`
**For torch compilation and optimization** - Use with `torch.compile()`.

- **Purpose**: Unrolls expressions into a linear sequence of operations
- **Features**: Optimized for PyTorch's compilation process, generates efficient code
- **Performance**: Fastest when used with `torch.compile()` 
- **Use when**: You need maximum performance and are using PyTorch 2.0+ compilation

```python
from geolipi.torch_compute import unroll_expression

# Unroll for compilation
unrolled_fn, context = unroll_expression(expression, sketcher)

# Use with torch.compile for best performance
compiled_fn = torch.compile(unrolled_fn)
result = compiled_fn(sketcher.get_coords())
```

#### `create_compiled_expr(expression, sketcher, **kwargs)`
**For batch processing** - Use for training neural networks.

- **Purpose**: Compiles expressions for efficient batch evaluation
- **Features**: Optimized for large batches, memory efficient
- **Performance**: Best for batch processing scenarios
- **Use when**: Training neural networks or processing many expressions simultaneously

### Deprecated Functions

#### `expr_to_sdf()` and `expr_to_colored_canvas()` ⚠️ 
**Deprecated** - Use `recursive_evaluate()` instead.

- **Purpose**: Legacy stack-based evaluation (kept for backward compatibility)
- **Limitations**: Does not support higher-order primitives or certain modifiers
- **Status**: Maintained but not recommended for new code

## Module Structure

### Core Evaluation
- `evaluate_expression.py` - Main evaluation engine with `recursive_evaluate()`
- `unroll_expression.py` - Expression unrolling for torch compilation
- `compile_expression.py` - Expression compilation utilities
- `deprecated.py` - Legacy evaluation functions

### SDF Functions
- `sdf_functions_2d.py` - 2D signed distance field implementations
- `sdf_functions_3d.py` - 3D signed distance field implementations  
- `sdf_operators.py` - Boolean operations (union, intersection, difference)
- `sdf_functions_higher.py` - Higher-order primitive operations

### Utilities
- `sketcher.py` - Coordinate generation and canvas management
- `sphere_marcher.py` - 3D rendering via sphere marching
- `visualizer.py` - Visualization utilities
- `constants.py` - Mathematical constants used throughout the module
- `maps.py` - Function mappings between symbolic and computational representations

### Batch Processing
- `batch_compile.py` - Batch compilation utilities
- `batch_evaluate_sdf.py` - Efficient batch SDF evaluation

### Color and Rendering
- `color_functions.py` - Color blending and manipulation functions
- `colorspace_functions.py` - Color space conversion utilities

## Performance Guidelines

### Choose the Right Function

1. **For general use**: `recursive_evaluate()` - Full features, good performance
2. **For maximum speed**: `unroll_expression()` + `torch.compile()` - Fastest execution
3. **For batch training**: `create_compiled_expr()` - Memory efficient batching
4. **For legacy code**: `expr_to_sdf()` - Limited features, backward compatibility

### Performance Tips

- Use GPU (`device="cuda"`) for better performance with large resolutions
- Enable `torch.compile()` with unrolled expressions for maximum speed
- Use appropriate resolution - higher resolution = more computation
- Consider batch processing for multiple expressions

## Examples

### Basic 2D SDF Evaluation
```python
import geolipi.symbolic as gls
from geolipi.torch_compute import recursive_evaluate, Sketcher

# Create a circle
circle = gls.Circle2D((0.5,))
sketcher = Sketcher(device="cuda", resolution=256, n_dims=2)

# Evaluate to get SDF
sdf = recursive_evaluate(circle, sketcher)
```

### Complex Expression with Color
```python
# Create colored shapes
red_circle = gls.ApplyColor2D(gls.Circle2D((0.3,)), gls.Symbol("red"))
blue_box = gls.ApplyColor2D(gls.Box2D((0.4, 0.4)), gls.Symbol("blue"))

# Combine with blending
expression = gls.SourceOver(red_circle, blue_box)

# Evaluate to get colored canvas
canvas = recursive_evaluate(expression, sketcher)
```

### High-Performance Compilation
```python
from geolipi.torch_compute import unroll_expression
import torch

# Unroll and compile for maximum performance
unrolled_fn, _ = unroll_expression(expression, sketcher)
compiled_fn = torch.compile(unrolled_fn, mode="max-autotune")

# Fast evaluation
result = compiled_fn(sketcher.get_coords())
```

## Dependencies

- PyTorch (primary computation backend)
- NumPy (mathematical constants and utilities)
- SymPy (symbolic expression handling)
- GeoLIPI symbolic module (expression definitions)
