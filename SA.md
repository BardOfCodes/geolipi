# Declarative ShapeAssembly

DSA -> 
Aligned Flag, and 3 attachments.

1) SACuboid(sizeVec, centerVec, AxisMat)
2) Attach(Shape_1, shape_2, Pt1, Pt2)
3) Reflect() and Translate()
4) Union
5) Squeeze -> 2 attach operations.

```python

Prog = Union(
    SACuboid(sizeVec, CenterVec, AxisMat)
    Attach(0, PtVec, PtVec)-> Translate(1, PtVec)
    Attach(0, PtVec, PtVec) -> Translate(1, PTVec) + Rotate() + resize()
    
)
```


# ProtoSA

1) Primitvies are defined with end points. Thats the only diff -> Cuboid(point1, point2, h, w, theta)