import numpy as np
import torch as th
from .common import EPSILON

COS_30 = np.cos(np.pi / 6)
TAN_30 = np.tan(np.pi / 6)

def sdf3d_sphere(points, radius):
    # points shape [batch, num_points, 3]
    # radius shape [batch, 1]
    base_sdf = th.norm(points, dim=-1)
    base_sdf = base_sdf - radius
    return base_sdf

def sdf3d_box(points, size):
    # points shape [batch, num_points, 3]
    # size shape [batch, 3]
    q = th.abs(points) - size[..., None, :]
    term_1 = th.norm(th.clamp(q, min=0), dim=-1)
    term_2 = th.clamp(th.amax(q, -1), max=0)
    base_sdf = term_1 + term_2
    return base_sdf

def sdf3d_rounded_box(points, size, radius):
    # points shape [batch, num_points, 3]
    # size shape [batch, 3]
    # radius shape [batch, 1]
    q = th.abs(points) - size[..., None, :]
    term_1 = th.norm(th.clamp(q, min=0), dim=-1)
    term_2 = th.clamp(th.amax(q, -1), max=0) - radius
    base_sdf = term_1 + term_2
    return base_sdf

def sdf3d_box_frame(points, b, e):
    # points shape [batch, num_points, 3]
    # b shape [batch, 3]
    # e shape [batch, 1]
    points = th.abs(points) - b[..., None, :]
    q = th.abs(points + e[..., None, :]) - e[..., None, :]
    p1 = q.clone()
    p1[..., 0] = points[..., 0]
    p2 = q.clone()
    p2[..., 1] = points[..., 1]
    p3 = q.clone()
    p3[..., 2] = points[..., 2]
    stack_choices = th.stack([p1, p2, p3], dim=0)
    terms = th.norm(th.clamp(stack_choices, min=0), dim=-1) + th.clamp(th.amax(stack_choices, -1), max=0)
    sdf = th.amin(terms, dim=0)
    return sdf

def sdf3d_torus(points, t):
    # points shape [batch, num_points, 3]
    # t shape [batch, 2]
    q_x = th.norm(points[..., :2], dim=-1) - t[..., None, 0]
    q = th.stack([q_x, points[..., 2]], dim=-1)
    base_sdf = th.norm(q, dim=-1) - t[..., None, 1]
    return base_sdf

def sdf3d_capped_torus(points, angle, ra, rb):
    # points shape [batch, num_points, 3]
    # angle shape [batch, 1]
    # ra shape [batch, 1]
    # rb shape [batch, 1]
    sc = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    points[..., 0] = th.abs(points[..., 0])
    term_1 = (points[..., :2] * sc).sum(-1)
    term_2 = th.norm(points[..., :2], dim=-1)
    k = th.where(sc[..., 1] * points[..., 0] > sc[..., 0] * points[..., 1], term_1, term_2)
    term = (points * points).sum(-1) + ra ** 2 - 2 * ra * k
    base_sdf = th.sqrt(term + EPSILON) - rb
    return base_sdf

def sdf3d_link(points, le, r1, r2):
    # points shape [batch, num_points, 3]
    # le shape [batch, 1]
    # r1 shape [batch, 1]
    # r2 shape [batch, 1]
    q = points.clone()
    q[..., 1] = th.clamp(th.abs(q[..., 1]) - le, min=0)
    sdf_x = th.norm(q[..., :2], dim=-1) - r1
    sdf = th.norm(th.stack([sdf_x, q[..., 2]], dim=-1), dim=-1) - r2
    return sdf

def sdf3d_infinite_cylinder(points, c):
    # points shape [batch, num_points, 3]
    # c shape [batch, 3]
    base_sdf = th.norm(points[..., :2] - c[..., None, :2], dim=-1) - c[..., 2:3]
    return base_sdf

def sdf3d_cone(points, angle, h):
    # Alternatively pass q instead of (c,h),
    # which is the point at the base in 2D
    # points shape [batch, num_points, 3]
    # angle shape [batch, 1]
    # h shape [batch, 1]
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = h[..., None, :] * th.stack([c[..., 0] / c[..., 1], -th.ones_like(c[..., 0])], dim=-1)
    w = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    a = w - q * th.clamp((w * q).sum(-1, keepdim=True) / (q * q).sum(-1, keepdim=True), min=0.0, max=1.0)
    b = w - q * th.stack([th.clamp(w[..., 0] / q[..., 0], min=0.0, max=1.0), th.ones_like(w[..., 0])], dim=-1)
    k = th.sign(q[..., 1])
    d = th.minimum((a * a).sum(-1), (b * b).sum(-1))
    s = th.maximum(k * (w[..., 0] * q[..., 1] - w[..., 1] * q[..., 0]), k * (w[..., 1] - q[..., 1]))
    base_sdf = th.sqrt(d) * th.sign(s)
    return base_sdf

def sdf3d_inexact_cone(points, angle, h):
    # points shape [batch, num_points, 3]
    # angle shape [batch, 1]
    # h shape [batch, 1]
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = th.norm(points[..., :2], dim=-1)
    base_sdf = th.maximum((c * th.stack([q, points[..., 2]], dim=-1)).sum(-1), -h - points[..., 2])
    return base_sdf

def sdf3d_infinite_cone(points, angle):
    # points shape [batch, num_points, 3]
    # c shape [batch, 1]
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = th.norm(points[..., :2], dim=-1)
    q = th.stack([q, points[..., 2]], dim=-1)
    d = th.norm(q - c * th.clamp((q * c).sum(-1, keepdim=True), min=0.0), dim=-1)
    base_sdf = d * th.where((q[..., 0] * c[..., 1] - q[..., 1] * c[..., 0]) < 0.0, -1, 1)
    return base_sdf

# -------------------------------------------------------------------
def sdf3d_plane(points, n, h):
    # points shape [batch, num_points, 3]
    # n shape [batch, 3]
    # h shape [batch, 1]
    n = n / th.norm(n + EPSILON, dim=-1, keepdim=True)
    base_sdf = (points * n).sum(-1) + h
    return base_sdf

def sdf3d_hex_prism(points, h):
    # points shape [batch, num_points, 3]
    # h shape [batch, 2]
    k = th.tensor([-COS_30, 0.5, TAN_30], device=points.device, dtype=th.float32)
    k = k[None, None, :]
    points = th.abs(points)
    points[..., :2] = points[..., :2] - 2.0 * th.clamp((points[..., :2] * k[..., :2]).sum(-1, keepdim=True), max=0.0) * k[..., :2]
    lim = h[..., None, 0] * 0.5
    t1 = points[..., :2].clone()
    t1[..., 0] = t1[..., 0] - th.clamp(points[..., 0], min=-lim, max=lim)
    t1[..., 1] = t1[..., 1] - h[..., None, 0]
    term_1 = th.norm(t1, dim=-1) * th.sign(points[..., 1] - h[..., None, 0])
    term_2 = points[..., 2] - h[..., None, 1]
    base_sdf = th.clamp(th.maximum(term_1, term_2), max=0.0) + th.norm(th.clamp(th.stack([term_1, term_2], dim=-1), min=0.0), dim=-1)
    return base_sdf
    
def sdf3d_tri_prism(points, h):
    # points shape [batch, num_points, 3]
    # h shape [batch, 2]
    h = h[..., None, :]
    cos_30 = th.cos(th.tensor(np.pi / 6, device=points.device, dtype=th.float32))
    q = th.abs(points)
    term_1 = q[..., 2] - h[..., 1]
    
    term_2 = th.maximum(q[..., 0] * cos_30 + points[..., 1] * 0.5, -points[..., 1]) - h[..., 0] * 0.5
    sdf = th.maximum(term_1, term_2)
    return sdf

def sdf3d_capsule(points,  a, b, r):
    # points shape [batch, num_points, 3]
    # a shape [batch, 3]
    # b shape [batch, 3]
    # r shape [batch, 1]
    pa = points - a[..., None, :]
    ba = b[..., None, :] - a[..., None, :]
    h = th.clamp((pa * ba).sum(-1, keepdim=True) / (ba * ba).sum(-1, keepdim=True), min=0.0, max=1.0)
    sdf = th.norm(pa - ba * h, dim=-1) - r
    return sdf

def sdf3d_vertical_capsule(points, h, r):
    # points shape [batch, num_points, 3]
    # h shape [batch, 1]
    # r shape [batch, 1]
    points[..., 1] = points[..., 1] - th.clamp(th.clamp(points[..., 1], min=0.0), max=h)
    base_sdf = th.norm(points, dim=-1) - r
    return base_sdf

def sdf3d_vertical_capped_cylinder(points, h, r):
    # points shape [batch, num_points, 3]
    # h shape [batch, 1]
    # r shape [batch, 1]
    d = th.abs(th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)) - th.stack([r, h], dim=-1)
    base_sdf = th.clamp(th.amax(d[..., :2], dim=-1), max=0.0) + th.norm(th.clamp(d, min=0.0), dim=-1)
    return base_sdf

def sdf3d_capped_cylinder(points, h, r):
    # points shape [batch, num_points, 3]
    # h shape [batch, 1]
    # r shape [batch, 1]
    d = th.abs(th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)) - th.stack([r, h], dim=-1)
    base_sdf = th.clamp(th.amax(d[..., :2], dim=-1), max=0.0) + th.norm(th.clamp(d, min=0.0), dim=-1)
    return base_sdf

def sdf3d_arbitrary_capped_cylinder(points, a, b, r):
    # points shape [batch, num_points, 3]
    # a shape [batch, 3]
    # b shape [batch, 3]
    # r shape [batch, 1]
    ba = b - a
    ba = ba[..., None, :]
    pa = points - a[..., None, :]
    baba = (ba * ba).sum(-1, keepdim=True)
    paba = (pa * ba).sum(-1, keepdim=True)
    x = th.norm(pa * baba - ba * paba, dim=-1, keepdim=True) - r[..., None, :] * baba
    y = th.abs(paba - baba * 0.5) - baba * 0.5
    x2 = x * x
    y2 = y * y * baba
    cond = th.maximum(x, y) < 0.0
    term_1 = -th.minimum(x2, y2)
    term_2 = th.where(x > 0.0, x2, 0.0) + th.where(y > 0.0, y2, 0.0)
    d = th.where(cond, term_1, term_2)
    base_sdf = th.sign(d) * th.sqrt(th.abs(d)) / baba
    return base_sdf

def sdf3d_rounded_cylinder(points, ra, rb, h):
    # points shape [batch, num_points, 3]
    # ra shape [batch, 1]
    # rb shape [batch, 1]
    # h shape [batch, 1]
    d_x = th.norm(points[..., :2], dim=-1) - 2.0 * ra + rb
    d_y = th.abs(points[..., 2]) - h
    d = th.stack([d_x, d_y], dim=-1)
    base_sdf = th.clamp(th.amax(d, dim=-1), max=0.0) + th.norm(th.clamp(d, min=0.0), dim=-1) - rb
    return base_sdf

def sdf3d_capped_cone(points, r1, r2, h):
    # points shape [batch, num_points, 3]
    # h shape [batch, 1]
    # r1 shape [batch, 1]
    # r2 shape [batch, 1]
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    k1 = th.stack([r2, h], dim=-1)
    k2 = th.stack([r2 - r1, 2 * h], dim=-1)
    ca = th.stack([q[..., 0] - th.clamp(q[..., 0], min=th.where(q[..., 1] < 0.0, r1, r2)), th.abs(q[..., 1]) - h], dim=-1)
    cb = q - k1 + k2 * th.clamp(((k1 - q) * k2).sum(-1, keepdim=True) / (k2 * k2).sum(-1, keepdim=True), min=0.0, max=1.0)
    s = th.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    base_sdf = s * th.sqrt(th.minimum((ca * ca).sum(-1), (cb * cb).sum(-1)))
    return base_sdf

def sdf3d_arbitrary_capped_cone(points, a, b, ra, rb):
    # points shape [batch, num_points, 3]
    # a shape [batch, 3]
    # b shape [batch, 3]
    # ra shape [batch, 1]
    # rb shape [batch, 1]
    ra = ra[..., None, :]
    rb = rb[..., None, :]
    rba = (rb - ra)
    ba = b - a
    baba = (ba*ba).sum(-1, keepdim=True)[..., None, :]
    pa = points - a[..., None, :]
    papa = ((pa * pa)).sum(-1, keepdim=True)
    paba = (pa * ba[..., None, :]).sum(-1, keepdim=True) / baba
    x = th.sqrt(papa - paba * paba * baba)
    cax = th.clamp(x - th.where(paba < 0.5, ra, rb), min=0.0)
    cay = th.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = th.clamp((rba * (x - ra) + paba * baba) / k, min=0.0, max=1.0)
    cbx = x - ra - f * rba
    cby = paba - f
    s = th.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    base_sdf = s * th.sqrt(th.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))
    return base_sdf

def sdf3d_solid_angle(points, angle, ra):
    # points shape [batch, num_points, 3]
    # angle shape [batch, 1]
    # ra shape [batch, 1]
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    l = th.norm(q, dim=-1) - ra
    m = th.norm(q - c * th.clamp(th.clamp((q * c).sum(-1, keepdim=True), min=0.0), max=ra[..., None, :]), dim=-1)
    base_sdf = th.maximum(l, m * th.sign(c[..., 1] * q[..., 0] - c[..., 0] * q[..., 1]))
    return base_sdf

def sdf3d_cut_sphere(points, r, h):
    # points shape [batch, num_points, 3]
    # r shape [batch, 1]
    # h shape [batch, 1]
    w = th.sqrt(r * r - h * h)
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    term_1 = (h - r) * q[..., 0] * q[..., 0] + w * w * (h + r - 2.0 * q[..., 1])
    term_2 = h * q[..., 0] - w * q[..., 1]
    s = th.maximum(term_1, term_2)
    base_sdf = th.where(s < 0.0, th.norm(q, dim=-1) - r, 
                        th.where(q[..., 0] < w, 
                                 h - q[..., 1], 
                                 th.norm(q - th.stack([w, h], dim=-1), dim=-1)))
    return base_sdf

def sdf3d_cut_hollow_sphere(points, r, h, t):
    # points shape [batch, num_points, 3]
    # r shape [batch, 1]
    # h shape [batch, 1]
    # t shape [batch, 1]
    w = th.sqrt(r * r - h * h)
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    term_1 = th.norm(q - th.stack([w, h], dim=-1), dim=-1)
    term_2 = th.abs(th.norm(q, dim=-1) - r)
    base_sdf = th.where(q[..., 0]  * h < w * q[..., 1], term_1, term_2) - t
    return base_sdf

def sdf3d_death_star(points, ra, rb , d):
    # points shape [batch, num_points, 3]
    # ra shape [batch, 1]
    # rb shape [batch, 1]
    # d shape [batch, 1]
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = th.sqrt(th.clamp(ra * ra - a * a, min=0.0))
    p = th.stack([points[..., 0], th.norm(points[..., 1:], dim=-1)], dim=-1)
    cond = p[..., 0] * b - p[..., 1] * a > d * th.clamp(b - p[..., 1], min=0.0)
    term_1 = th.norm(p - th.stack([a, b], dim=-1), dim=-1)
    p2 = p.clone()
    p2[..., 0] = p2[..., 0] - d
    term_2 = th.maximum(th.norm(p, dim=-1) - ra, -th.norm(p2, dim=-1) + rb)
    base_sdf = th.where(cond, term_1, term_2)
    return base_sdf

# 10 functions remaining.
