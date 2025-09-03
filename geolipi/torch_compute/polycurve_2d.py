import torch as th
from .constants import EPSILON, ACOS_EPSILON


########## POLYLINE SDF

def helper_polyline_sdf_arc(points, sc, ra):
    p = th.cat([th.abs(points[..., 0:1]), points[..., 1:2]], -1)
    # p[..., 0] = p[..., 0].abs()
    if sc[0, 0] < 0.0:
        d = -sc * ra
    else:
        d = sc * ra
        
    dist_1 = th.norm(p - d, dim=-1) 
    dist_2 = th.abs(th.norm(p, dim=-1) - ra)
    sdf = th.where(sc[0, 1] * p[..., 0] > sc[0, 0] * p[..., 1], dist_1, dist_2)
    return sdf

def helper_polyline_sc_from_ht(tan_half_x):
    tan_half_sq = tan_half_x * tan_half_x
    denom = 1.0 + tan_half_sq
    # sc = th.stack([1.0 - tan_half_sq, 2.0 * tan_half_x], -1) / denom
    sc = th.stack([(2.0 * tan_half_x), (1.0 - tan_half_sq)], -1) / denom
    return sc

def helper_polyline_bulge_arc(points, a, b, bulge):
    """
    points: (N, 2)
    a: (1, 2)
    b: (1, 2)
    bulge: (1,)
    """
    ba = b - a 
    l = th.norm(ba, dim=-1)
    ortho_ba = th.stack([ba[..., 1], -ba[..., 0]], -1)
    if l < EPSILON:
        l = l + EPSILON
    orth_ba = ortho_ba / (l)
    div = 1 - bulge * bulge
    div = th.where(div < EPSILON, div + EPSILON, div)
    tan_tby2 = 2.0 * bulge / (div)
    h = l / (2.0 * tan_tby2)
    center = (a + b) * 0.5 + orth_ba * h

    theta = -th.atan2(ba[..., 1], ba[..., 0])
    # theta = th.atan(ba[..., 1]/ba[..., 0])
    cos_theta = th.cos(-theta)
    sin_theta = th.sin(-theta)
    rot_mat = th.stack([cos_theta, sin_theta, -sin_theta, cos_theta], -1).view(-1, 2, 2)
    relative_p = points - center
    # relative_p is size (N, 2), rotmat is size (1, 2, 2)
    rot_points = th.einsum("bij,nj->ni", rot_mat, relative_p)

    radius = th.norm(a - center, dim=-1)
    sc = helper_polyline_sc_from_ht(bulge)

    d = helper_polyline_sdf_arc(rot_points, sc, radius)

    min_y = th.min(a[..., 1], b[..., 1])
    max_y = th.max(a[..., 1], b[..., 1])
    in_range =  (points[..., 1] > min_y) & (points[..., 1] < max_y)
    in_circ = th.norm(relative_p, dim=-1) <= radius

    # first part
    pa = points - a
    cond_1 = points[..., 1] >= a[0, 1]
    cond_2 = points[..., 1] < b[0, 1]
    cond_3 = ba[0, 0] * pa[..., 1] - ba[0, 1] * pa[..., 0] > 0
    left_ba = (cond_1 & cond_2 & cond_3) | (~cond_1 & ~cond_2 & ~cond_3)
    positive_arc = (b[0, 1] >= a[0, 1] and bulge > 0.0) or (b[0, 1] < a[0, 1] and bulge < 0.0)
    mul_1 = th.where(left_ba & (~in_circ), -1.0, 1.0)
    mul_2 = th.where(left_ba | in_circ, -1.0, 1.0)
    mul = th.where(positive_arc, mul_1, mul_2) 

    # next part
    consider = th.stack([a[0, 1], b[0, 1], a[0, 1], b[0, 1]], -1)
    maxy = center[..., 1] + radius
    miny = center[..., 1] - radius
    ac = a - center
    bc = b - center
    start = th.where(bulge >= 0.0, bc, ac)
    end = th.where(bulge >= 0.0, ac, bc)
    c = start[..., 0] * end[..., 1] - start[..., 1] * end[..., 0]
    c1 = -start[..., 0]
    c2 = -end[..., 0]
    consider[2] = th.where(c > 0.0, th.where(c1 < 0.0 and c2 > 0.0, maxy, a[0, 1]), th.where(c1 < 0.0 or c2 > 0.0, maxy, a[0, 1]))
    
    c1 = start[..., 0]
    c2 = end[..., 0]
    consider[3] = th.where(c > 0.0, th.where(c1 < 0.0 and c2 > 0.0, miny, a[0, 1]), th.where(c1 < 0.0 or c2 > 0.0, miny, a[0, 1]))
    arc_max_y = th.max(consider)
    arc_min_y = th.min(consider)
    in_arc_range = (points[..., 1] > arc_min_y) & (points[..., 1] < arc_max_y)
    mul_3 = th.where(in_circ & in_arc_range, -1.0, 1.0)

    final_mul = th.where(in_range, mul, mul_3)
    out = th.stack([d * d, final_mul], -1)
    return out


def helper_polyline_line(points, a, b):
    """
    points: (N, 2)
    a: (1, 2)
    b: (1, 2)
    """
    pa = points - a # (N, 2)
    ba = b - a # (1, 2)
    div = (ba * ba).sum(-1, keepdim=True)
    div = th.where(div < EPSILON, div + EPSILON, div)
    h = th.clamp((pa * ba).sum(-1, keepdim=True) / div, 0.0, 1.0)
    val = pa - ba * h
    sdf = (val * val).sum(-1)
    cond_1 = points[..., 1] >= a[0, 1]
    cond_2 = points[..., 1] < b[0, 1]
    # cond 3 - cross of ba, pa > 0
    cond_3 = ba[0, 0] * pa[..., 1] - ba[0, 1] * pa[..., 0] > 0
    cond = (cond_1 & cond_2 & cond_3) | (~cond_1 & ~cond_2 & ~cond_3)
    sign = th.where(cond, -1.0, 1.0)
    out = th.stack([sdf, sign], -1)
    return out


def helper_polyline_line_parallel(points, a_set, b_set):
    """
    TBD
    points: (N, 2)
    a_set: (k, 2)
    b_set: (k, 2)
    """
    points = points.unsqueeze(1)
    a_set = a_set.unsqueeze(0)
    b_set = b_set.unsqueeze(0)
    pa = points - a_set # (N, 2)
    ba = b_set - a_set # (1, 2)
    h = th.clamp((pa * ba).sum(-1, keepdim=True) / ((ba * ba).sum(-1, keepdim=True) + EPSILON), 0.0, 1.0)
    val = pa - ba * h
    sdf = (val * val).sum(-1)
    cond_1 = points[..., 1] >= a_set[..., 1]
    cond_2 = points[..., 1] < b_set[..., 1]
    # cond 3 - cross of ba, pa > 0
    cond_3 = ba[..., 0] * pa[..., 1] - ba[..., 1] * pa[..., 0] > 0
    cond = (cond_1 & cond_2 & cond_3) | (~cond_1 & ~cond_2 & ~cond_3)
    sign = th.where(cond, -1.0, 1.0)
    out = th.stack([sdf, sign], -1)
    return out

def helper_polyline_bulge_arc_parallel(points, a_set, b_set):
    """
    TBD
    points: (N, 2)
    a_set: (k, 2)
    b_set: (k, 2)
    """
    # This function appears to be incomplete in the original file
    # Placeholder implementation
    pass


def sdf2d_polyline(points: th.Tensor, vertices: th.Tensor) -> th.Tensor:
    """
    Parameters:
        points: Coordinates to evaluate, shape (N, 2)
        vertices: Polyline vertices (x, y, bulge), shape (k, 3)

    Returns:
        Tensor: SDF values for the polyline
    """
    # Ideally, gather all ther vertices where bulge = 0
    # and similarly compute all ther vertices where bulge != 0
    # then in parallel for all the points compute both the sign and distance
    # Final distance will be min of the distances, and final sign will be multiplication of all the signs. 
    n_points = vertices.shape[0]
    a = vertices[0:1, :2]
    b = vertices[1:2, :2]
    d = th.zeros_like(points[..., 0:1]) + 100.0
    s = th.ones_like(d)
    ds = th.cat([d, s], -1)
    for i in range(n_points):
        a = vertices[i:i+1, :2]
        next = (i + 1) % n_points
        b = vertices[next: next + 1, :2]
        if vertices[i, 2] == 0.0:
            ds = helper_polyline_line(points, a, b)
        else:
            ds = helper_polyline_bulge_arc(points, a, b, -vertices[i:i+1, 2])
        d = th.minimum(d, ds[..., 0:1])
        s = s * ds[..., 1:2]
    # Make gradients proper.
    sel_d =  d[..., 0]
    d_sim = th.where(sel_d > ACOS_EPSILON, sel_d, ACOS_EPSILON)
    d_sqrt = th.sqrt(d_sim)
    return s[..., 0] * d_sqrt


def nonsdf2d_polyline_smooth(points, vertices, smoothness):
    """
    Computes the SDF for a polyline in 2D.
    vertices is a list of (x, y, bulge) format of size (k, 3).
    points are (x, y) points on which SDF is to be computed, of size (N, 2).
    """
    # Ideally, gather all ther vertices where bulge = 0
    # and similarly compute all ther vertices where bulge != 0
    # then in parallel for all the points compute both the sign and distance
    # Final distance will be min of the distances, and final sign will be multiplication of all the signs. 
    d = points[..., 0] * 0.0 + 100.0
    vertices_a = vertices[:, :2]
    # roll
    vertices_b = th.roll(vertices_a, -1, 0)
    # Do the parallel computation. 
    ds = helper_polyline_line_parallel(points, vertices_a, vertices_b)
    # Ds should be N, k, 2
    distances = ds[..., 0]
    signs = ds[..., 1]
    # TO make it differentiable -> convert it into a temperature guided summation.
    d = th.amin(distances, 1)
    s = th.prod(signs, 1)
    
    d = th.where(d > EPSILON, d, EPSILON)
    return s * th.sqrt(d)
