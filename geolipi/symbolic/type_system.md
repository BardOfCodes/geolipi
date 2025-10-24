Type System for GeoLIPI default_spec

This document defines the string-based type grammar used by GLFunction.default_spec() and provides common aliases and conventions.

Grammar

Type        := Simple | Param | Union | Optional | Enum | Literal
Simple      := "int" | "float" | "bool" | "str" | "None"
             | Vector | Matrix | Tensor
             | "Expr" | "Node" | "List" | "Tuple" | "Dict"

Param       := "Expr" "[" Type "]"
             | "Node" "[" Type "]"
             | "List" "[" Type "]"
             | "Tuple" "[" Type {"," Type} "]"
             | "Dict" "[" KeyType "," Type "]"
             | Vector | Matrix | Tensor

Vector      := "Vector" "[" Dim "]"
Matrix      := "Matrix" "[" Dim "," Dim "]"
Tensor      := "Tensor" "[" DType "," Shape "]"

Union       := "Union" "[" Type {"|" Type} "]"
Optional    := "Optional" "[" Type "]"
Enum        := "Enum" "[" Quoted {"|" Quoted} "]"
Literal     := "Literal" "[" (Number | Quoted) "]"

DType       := "float" | "int" | "bool"
Dim         := Number
Shape       := "(" DimOrSym {"," DimOrSym} ")"
DimOrSym    := Number | Ident
KeyType     := "str" | "int"
Quoted      := '"' {any char except '"'} '"'
Ident       := Letter {Letter | Digit | "_"}
Number      := Digit {Digit}

Aliases

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

Conventions

- Input naming:
  - Use "input" for a single child expression.
  - Use "inputs" for variadic child expressions (Union, SmoothUnion, etc.).
  - For color compositors, use "canvas" or "canvases" as appropriate.

- Categories (for documentation and code-gen grouping):
  - primitives_2d, primitives_3d, combinators, transforms_2d, transforms_3d, higher, color.

- Common parameter names:
  - "offset": Vector[2|3] translations
  - "scale": Vector[k] (canonical vector form)
  - "angles": Vector[3] (Euler radians)
  - "axis": Vector[3]
  - "angle": float (radians)
  - "matrix": Matrix[3,3] or Matrix[4,4] as appropriate
  - "shear": Tuple[...] or Vector[...] (implementation-defined)
  - "amount", "thickness", "radius", "k": float parameters with optional ranges

- Defaults and constraints:
  - Each entry is {"type": str, ...optional refinements...}
  - Supported refinements: default, min, max, exclusive_min, exclusive_max, min_len, max_len, desc
  - Shapes may include symbolic dimensions; shape symbols can be validated with an env mapping (e.g., {"N": 64}).

Examples

- Sphere(center, radius)
  {
    "center": {"type": "Vector[3]"},
    "radius": {"type": "float", "min": 0.0}
  }

- Polyline(points: List[Vector[2]], closed: bool=False)
  {
    "points": {"type": "List[Vector[2]]", "min_len": 2},
    "closed": {"type": "bool", "default": false}
  }

- TextureSample(uv: Expr[Vector[2]], filter: Enum["nearest"|"linear"]) 
  {
    "uv": {"type": "Expr[Vector[2]]"},
    "filter": {"type": "Enum[\"nearest\"|\"linear\"]", "default": "linear"}
  }

- Transform (canonical TRS or matrix)
  {
    "transform": {"type": "Union[Matrix[4,4]|Tuple[Vector[3],Vector[4],Vector[3]]]"}
  }

- Optional material color
  { "albedo": {"type": "Optional[Vector[3]]"} }

- Tensor with symbolic shape
  { "weights": {"type": "Tensor[float, (N, 3)]"} }




