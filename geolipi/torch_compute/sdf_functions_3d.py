import numpy as np
import torch as th
from .common import EPSILON, ACOS_EPSILON
from .sdf_functions_2d import ndot

COS_30 = np.cos(np.pi / 6)
TAN_30 = np.tan(np.pi / 6)


def sdf3d_sphere(points, radius):
    """
    Calculates the signed distance from 3D points to the surface of a sphere.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        radius (torch.Tensor): A tensor of shape [batch, 1] representing the radii of spheres.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the sphere surface.
    """

    # Calculate the Euclidean norm of each point and subtract the radius for signed distance
    base_sdf = th.norm(points, dim=-1) - radius
    return base_sdf


def sdf3d_box(points, size):
    """
    Calculates the signed distance from 3D points to the surface of an axis-aligned 3D box.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        size (torch.Tensor): A tensor of shape [batch, 3] representing the half extents of the box.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the box surface.
    """
    q = th.abs(points) - size[..., None, :]
    term_1 = th.norm(th.clamp(q, min=0), dim=-1)
    term_2 = th.clamp(th.amax(q, -1), max=0)
    base_sdf = term_1 + term_2
    return base_sdf


def sdf3d_rounded_box(points, size, radius):
    """
    Calculates the signed distance from 3D points to the surface of a rounded axis-aligned 3D box.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        size (torch.Tensor): A tensor of shape [batch, 3] representing the half extents of the box.
        radius (torch.Tensor): A tensor of shape [batch, 1] representing the radius of rounding.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the rounded box surface.
    """
    q = th.abs(points) - size[..., None, :]
    term_1 = th.norm(th.clamp(q, min=0), dim=-1)
    term_2 = th.clamp(th.amax(q, -1), max=0) - radius
    base_sdf = term_1 + term_2
    return base_sdf


def sdf3d_box_frame(points, b, e):
    """
    Calculates the signed distance from 3D points to the surface of a box frame.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        b (torch.Tensor): A tensor of shape [batch, 3] representing the half extents of the box frame.
        e (torch.Tensor): A tensor of shape [batch, 1] representing the frame thickness.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the box frame surface.
    """
    points = th.abs(points) - b[..., None, :]
    q = th.abs(points + e[..., None, :]) - e[..., None, :]
    p1 = q.clone()
    p1[..., 0] = points[..., 0]
    p2 = q.clone()
    p2[..., 1] = points[..., 1]
    p3 = q.clone()
    p3[..., 2] = points[..., 2]
    stack_choices = th.stack([p1, p2, p3], dim=0)
    terms = th.norm(th.clamp(stack_choices, min=0), dim=-1) + th.clamp(
        th.amax(stack_choices, -1), max=0
    )
    sdf = th.amin(terms, dim=0)
    return sdf


def sdf3d_torus(points, t):
    """
    Calculates the signed distance from 3D points to the surface of a torus.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        t (torch.Tensor): A tensor of shape [batch, 2] representing the major (first component) and minor (second component) radii of the torus.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the torus surface.
    """
    q_x = th.norm(points[..., :2], dim=-1) - t[..., None, 0]
    q = th.stack([q_x, points[..., 2]], dim=-1)
    base_sdf = th.norm(q, dim=-1) - t[..., None, 1]
    return base_sdf


def sdf3d_capped_torus(points, angle, ra, rb):
    """
    Calculates the signed distance from 3D points to the surface of a capped torus.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        angle (torch.Tensor): A tensor of shape [batch, 1] representing the angle of the cap.
        ra (torch.Tensor): A tensor of shape [batch, 1] representing the major radius of the torus.
        rb (torch.Tensor): A tensor of shape [batch, 1] representing the minor radius of the torus.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the capped torus surface.
    """
    sc = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    points[..., 0] = th.abs(points[..., 0])
    term_1 = (points[..., :2] * sc).sum(-1)
    term_2 = th.norm(points[..., :2], dim=-1)
    k = th.where(
        sc[..., 1] * points[..., 0] > sc[..., 0] * points[..., 1], term_1, term_2
    )
    term = (points * points).sum(-1) + ra**2 - 2 * ra * k
    base_sdf = th.sqrt(th.clamp(term, min=EPSILON)) - rb
    return base_sdf


def sdf3d_link(points, le, r1, r2):
    """
    Calculates the signed distance from 3D points to the surface of a link shape.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        le (torch.Tensor): A tensor of shape [batch, 1] representing the length of the link's straight section.
        r1 (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the link's first circular section.
        r2 (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the link's second circular section.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the link shape surface.
    """
    q = points.clone()
    q[..., 1] = th.clamp(th.abs(q[..., 1]) - le, min=0)
    sdf_x = th.norm(q[..., :2], dim=-1) - r1
    sdf = th.norm(th.stack([sdf_x, q[..., 2]], dim=-1), dim=-1) - r2
    return sdf


def sdf3d_infinite_cylinder(points, c):
    """
    Calculates the signed distance from 3D points to the surface of an infinite cylinder.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        c (torch.Tensor): A tensor of shape [batch, 3] representing the cylinder's center and radius (last component).

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the infinite cylinder surface.
    """
    base_sdf = th.norm(points[..., :2] - c[..., None, :2], dim=-1) - c[..., 2:3]
    return base_sdf


def sdf3d_cone(points, angle, h):
    """
    Calculates the signed distance from 3D points to the surface of a cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        angle (torch.Tensor): A tensor of shape [batch, 1] representing the cone's angle.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the cone's height.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the cone surface.
    """
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    c = th.where(c == 0, EPSILON, c)
    q = h[..., None, :] * th.stack(
        [c[..., 0] / c[..., 1], -th.ones_like(c[..., 0])], dim=-1
    )
    w = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    q = th.where(q == 0, EPSILON, q)
    a = w - q * th.clamp(
        (w * q).sum(-1, keepdim=True) / (q * q).sum(-1, keepdim=True), min=0.0, max=1.0
    )
    b = w - q * th.stack(
        [th.clamp(w[..., 0] / q[..., 0], min=0.0, max=1.0), th.ones_like(w[..., 0])],
        dim=-1,
    )
    k = th.sign(q[..., 1])
    d = th.minimum((a * a).sum(-1), (b * b).sum(-1))
    s = th.maximum(
        k * (w[..., 0] * q[..., 1] - w[..., 1] * q[..., 0]), k * (w[..., 1] - q[..., 1])
    )
    base_sdf = th.sqrt(d + EPSILON) * th.sign(s)
    return base_sdf


def sdf3d_inexact_cone(points, angle, h):
    """
    Calculates an approximate signed distance from 3D points to the surface of a cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        angle (torch.Tensor): A tensor of shape [batch, 1] representing the cone's angle.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the cone's height.

    Returns:
        torch.Tensor: A tensor containing the approximate signed distances of each point to the cone surface.
    """
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = th.norm(points[..., :2], dim=-1)
    base_sdf = th.maximum(
        (c * th.stack([q, points[..., 2]], dim=-1)).sum(-1), -h - points[..., 2]
    )
    return base_sdf


def sdf3d_infinite_cone(points, angle):
    """
    Calculates the signed distance from 3D points to the surface of an infinite cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        angle (torch.Tensor): A tensor of shape [batch, 1] representing the cone's angle.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the infinite cone surface.
    """
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = th.norm(points[..., :2], dim=-1)
    q = th.stack([q, points[..., 2]], dim=-1)
    d = th.norm(q - c * th.clamp((q * c).sum(-1, keepdim=True), min=0.0), dim=-1)
    base_sdf = d * th.where(
        (q[..., 0] * c[..., 1] - q[..., 1] * c[..., 0]) < 0.0, -1, 1
    )
    return base_sdf


def sdf3d_plane(points, n, h):
    """
    Calculates the signed distance from 3D points to a plane.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        n (torch.Tensor): A tensor of shape [batch, 3] representing the plane's normal vector.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the plane's offset from the origin.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the plane.
    """
    n = n / (th.norm(n, dim=-1, keepdim=True) + EPSILON)
    base_sdf = (points * n[..., None, :]).sum(-1) + h
    return base_sdf


def sdf3d_hex_prism(points, h):
    """
    Calculates the signed distance from 3D points to the surface of a hexagonal prism.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        h (torch.Tensor): A tensor of shape [batch, 2] representing the prism's height and width.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the hexagonal prism surface.
    """
    k = th.tensor([-COS_30, 0.5, TAN_30], device=points.device, dtype=th.float32)
    k = k[None, None, :]
    points = th.abs(points)
    points[..., :2] = (
        points[..., :2]
        - 2.0
        * th.clamp((points[..., :2] * k[..., :2]).sum(-1, keepdim=True), max=0.0)
        * k[..., :2]
    )
    lim = h[..., None, 0] * 0.5
    t1 = points[..., :2].clone()
    t1[..., 0] = t1[..., 0] - th.clamp(points[..., 0], min=-lim, max=lim)
    t1[..., 1] = t1[..., 1] - h[..., None, 0]
    term_1 = th.norm(t1, dim=-1) * th.sign(points[..., 1] - h[..., None, 0])
    term_2 = points[..., 2] - h[..., None, 1]
    base_sdf = th.clamp(th.maximum(term_1, term_2), max=0.0) + th.norm(
        th.clamp(th.stack([term_1, term_2], dim=-1), min=0.0), dim=-1
    )
    return base_sdf


def sdf3d_tri_prism(points, h):
    """
    Calculates the signed distance from 3D points to the surface of a triangular prism.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        h (torch.Tensor): A tensor of shape [batch, 2] representing the prism's height and base edge length.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the triangular prism surface.
    """
    h = h[..., None, :]
    cos_30 = th.cos(th.tensor(np.pi / 6, device=points.device, dtype=th.float32))
    q = th.abs(points)
    term_1 = q[..., 2] - h[..., 1]

    term_2 = (
        th.maximum(q[..., 0] * cos_30 + points[..., 1] * 0.5, -points[..., 1])
        - h[..., 0] * 0.5
    )
    sdf = th.maximum(term_1, term_2)
    return sdf


def sdf3d_capsule(points, a, b, r):
    """
    Calculates the signed distance from 3D points to the surface of a capsule.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor of shape [batch, 3] representing one end of the capsule's centerline.
        b (torch.Tensor): A tensor of shape [batch, 3] representing the other end of the capsule's centerline.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the capsule's radius.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the capsule surface.
    """
    pa = points - a[..., None, :]
    ba = b[..., None, :] - a[..., None, :]
    h = th.clamp(
        (pa * ba).sum(-1, keepdim=True) / ((ba * ba).sum(-1, keepdim=True) + EPSILON),
        min=0.0,
        max=1.0,
    )
    sdf = th.norm(pa - ba * h, dim=-1) - r
    return sdf


def sdf3d_vertical_capsule(points, h, r):
    """
    Calculates the signed distance from 3D points to the surface of a vertically-aligned capsule.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the capsule's height.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the capsule's radius.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the vertical capsule surface.
    """
    points[..., 1] = points[..., 1] - th.clamp(th.clamp(points[..., 1], min=0.0), max=h)
    base_sdf = th.norm(points, dim=-1) - r
    return base_sdf


def sdf3d_vertical_capped_cylinder(points, h, r):
    """
    Calculates the signed distance from 3D points to the surface of a vertically-oriented capped cylinder.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cylinder.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the cylinder.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the capped cylinder surface.
    """
    d = th.abs(
        th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    ) - th.stack([r, h], dim=-1)
    base_sdf = th.clamp(th.amax(d[..., :2], dim=-1), max=0.0) + th.norm(
        th.clamp(d, min=0.0), dim=-1
    )
    return base_sdf


def sdf3d_capped_cylinder(points, h, r):
    """
    Calculates the signed distance from 3D points to the surface of a capped cylinder (aligned along the z-axis).

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cylinder.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the cylinder.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the capped cylinder surface.
    """
    d = th.abs(
        th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    ) - th.stack([r, h], dim=-1)
    base_sdf = th.clamp(th.amax(d[..., :2], dim=-1), max=0.0) + th.norm(
        th.clamp(d, min=0.0), dim=-1
    )
    return base_sdf


def sdf3d_arbitrary_capped_cylinder(points, a, b, r):
    """
    Calculates the signed distance from 3D points to the surface of an arbitrarily oriented capped cylinder.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor of shape [batch, 3] representing one end of the cylinder.
        b (torch.Tensor): A tensor of shape [batch, 3] representing the other end of the cylinder.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the cylinder.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the arbitrarily oriented capped cylinder surface.
    """
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
    base_sdf = th.sign(d) * th.sqrt(th.abs(d) + EPSILON) / (baba + EPSILON)
    return base_sdf


def sdf3d_rounded_cylinder(points, ra, rb, h):
    """
    Calculates the signed distance from 3D points to the surface of a rounded cylinder.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        ra (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the cylinder's circular faces.
        rb (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the cylinder's rounded edges.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cylinder.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the rounded cylinder surface.
    """
    d_x = th.norm(points[..., :2], dim=-1) - 2.0 * ra + rb
    d_y = th.abs(points[..., 2]) - h
    d = th.stack([d_x, d_y], dim=-1)
    base_sdf = (
        th.clamp(th.amax(d, dim=-1), max=0.0)
        + th.norm(th.clamp(d, min=0.0), dim=-1)
        - rb
    )
    return base_sdf


def sdf3d_capped_cone(points, r1, r2, h):
    """
    Calculates the signed distance from 3D points to the surface of a capped cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        r1 (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the base of the cone.
        r2 (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the top of the cone.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cone.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the capped cone surface.
    """
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    k1 = th.stack([r2, h], dim=-1)
    k2 = th.stack([r2 - r1, 2 * h], dim=-1)
    ca = th.stack(
        [
            q[..., 0] - th.clamp(q[..., 0], min=th.where(q[..., 1] < 0.0, r1, r2)),
            th.abs(q[..., 1]) - h,
        ],
        dim=-1,
    )
    cb = (
        q
        - k1
        + k2
        * th.clamp(
            ((k1 - q) * k2).sum(-1, keepdim=True)
            / ((k2 * k2).sum(-1, keepdim=True) + EPSILON),
            min=0.0,
            max=1.0,
        )
    )
    s = th.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    base_sdf = s * th.sqrt(th.minimum((ca * ca).sum(-1), (cb * cb).sum(-1)) + EPSILON)
    return base_sdf


def sdf3d_arbitrary_capped_cone(points, a, b, ra, rb):
    """
    Calculates the signed distance from 3D points to the surface of an arbitrarily oriented capped cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor of shape [batch, 3] representing the apex of the cone.
        b (torch.Tensor): A tensor of shape [batch, 3] representing the base center of the cone.
        ra (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the apex of the cone.
        rb (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the base of the cone.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the arbitrarily oriented capped cone surface.
    """
    ra = ra[..., None, :]
    rb = rb[..., None, :]
    rba = rb - ra
    ba = b - a
    baba = (ba * ba).sum(-1, keepdim=True)[..., None, :]
    pa = points - a[..., None, :]
    papa = ((pa * pa)).sum(-1, keepdim=True)
    paba = (pa * ba[..., None, :]).sum(-1, keepdim=True) / (baba + EPSILON)
    x = th.sqrt(th.clamp(papa - paba * paba * baba, min=EPSILON))
    cax = th.clamp(x - th.where(paba < 0.5, ra, rb), min=0.0)
    cay = th.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = th.clamp((rba * (x - ra) + paba * baba) / (k + EPSILON), min=0.0, max=1.0)
    cbx = x - ra - f * rba
    cby = paba - f
    s = th.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    base_sdf = s * th.sqrt(
        th.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba) + EPSILON
    )
    return base_sdf


def sdf3d_solid_angle(points, angle, ra):
    """
    Calculates the signed distance from 3D points to a solid angle surface.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        angle (torch.Tensor): A tensor of shape [batch, 1] representing the angle of the solid angle surface.
        ra (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the solid angle surface.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the solid angle surface.
    """
    c = th.stack([th.cos(angle), th.sin(angle)], dim=-1)
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    l = th.norm(q, dim=-1) - ra
    m = th.norm(
        q
        - c
        * th.clamp(
            th.clamp((q * c).sum(-1, keepdim=True), min=0.0), max=ra[..., None, :]
        ),
        dim=-1,
    )
    base_sdf = th.maximum(l, m * th.sign(c[..., 1] * q[..., 0] - c[..., 0] * q[..., 1]))
    return base_sdf


def sdf3d_cut_sphere(points, r, h):
    """
    Calculates the signed distance from 3D points to a cut sphere (a sphere with a cap removed).

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the sphere.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cut from the sphere's base.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the cut sphere surface.
    """
    w = th.sqrt(th.clamp(r * r - h * h, min=EPSILON))
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    term_1 = (h - r) * q[..., 0] * q[..., 0] + w * w * (h + r - 2.0 * q[..., 1])
    term_2 = h * q[..., 0] - w * q[..., 1]
    s = th.maximum(term_1, term_2)
    base_sdf = th.where(
        s < 0.0,
        th.norm(q, dim=-1) - r,
        th.where(
            q[..., 0] < w, h - q[..., 1], th.norm(q - th.stack([w, h], dim=-1), dim=-1)
        ),
    )
    return base_sdf


def sdf3d_cut_hollow_sphere(points, r, h, t):
    """
    Calculates the signed distance from 3D points to the surface of a cut hollow sphere.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        r (torch.Tensor): A tensor of shape [batch, 1] representing the outer radius of the hollow sphere.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cut from the sphere's base.
        t (torch.Tensor): A tensor of shape [batch, 1] representing the thickness of the hollow sphere.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the cut hollow sphere surface.
    """
    w = th.sqrt(th.clamp(r * r - h * h, min=EPSILON))
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    term_1 = th.norm(q - th.stack([w, h], dim=-1), dim=-1)
    term_2 = th.abs(th.norm(q, dim=-1) - r)
    base_sdf = th.where(q[..., 0] * h < w * q[..., 1], term_1, term_2) - t
    return base_sdf


def sdf3d_death_star(points, ra, rb, d):
    """
    Calculates the signed distance from 3D points to the surface of a 'Death Star'-like shape.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        ra (torch.Tensor): A tensor of shape [batch, 1] representing the outer radius of the main sphere.
        rb (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the indented sphere (dish).
        d (torch.Tensor): A tensor of shape [batch, 1] representing the distance of the dish's center from the main sphere's center.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the 'Death Star' surface.
    """
    d = th.where(d == 0, EPSILON, d)
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = th.sqrt(th.clamp(ra * ra - a * a, min=EPSILON))
    p = th.stack([points[..., 0], th.norm(points[..., 1:], dim=-1)], dim=-1)
    cond = p[..., 0] * b - p[..., 1] * a > d * th.clamp(b - p[..., 1], min=0.0)
    term_1 = th.norm(p - th.stack([a, b], dim=-1), dim=-1)
    p2 = p.clone()
    p2[..., 0] = p2[..., 0] - d
    term_2 = th.maximum(th.norm(p, dim=-1) - ra, -th.norm(p2, dim=-1) + rb)
    base_sdf = th.where(cond, term_1, term_2)
    return base_sdf


def sdf3d_round_cone(points, r1, r2, h):
    """
    Calculates the signed distance from 3D points to the surface of a round cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        r1 (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the base of the cone.
        r2 (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the top of the cone.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the cone.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the round cone surface.
    """
    h = th.where(h == 0, EPSILON, h)
    b = (r1 - r2) / h
    a = th.sqrt(th.clamp(1.0 - b * b, min=EPSILON))
    q = th.stack([th.norm(points[..., :2], dim=-1), points[..., 2]], dim=-1)
    k = (q * th.stack([-b, a], dim=-1)).sum(-1)
    cond_1 = k < 0.0
    cond_2 = k > a * h
    term_1 = th.norm(q, dim=-1) - r1
    q2 = q.clone()
    q2[..., 1] = q2[..., 1] - h
    term_2 = th.norm(q2, dim=-1) - r2
    term_3 = (q * th.stack([a, b], dim=-1)).sum(-1) - r1
    base_sdf = th.where(cond_1, term_1, th.where(cond_2, term_2, term_3))
    return base_sdf


def sdf3d_arbitrary_round_cone(points, a, b, r1, r2):
    """
    Calculates the signed distance from 3D points to the surface of an arbitrarily oriented round cone.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor of shape [batch, 3] representing one end of the cone's axis.
        b (torch.Tensor): A tensor of shape [batch, 3] representing the other end of the cone's axis.
        r1 (torch.Tensor): A tensor of shape [batch, 1] representing the radius at one end of the cone.
        r2 (torch.Tensor): A tensor of shape [batch, 1] representing the radius at the other end of the cone.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the arbitrarily oriented round cone surface.
    """
    a = a[..., None, :]
    b = b[..., None, :]
    r1 = r1[..., None, :]
    r2 = r2[..., None, :]
    ba = b - a
    l2 = (ba * ba).sum(-1, keepdim=True)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / (l2 + EPSILON)
    pa = points - a
    y = (pa * ba).sum(-1, keepdim=True)
    z = y - l2
    x2 = pa * l2 - ba * y
    x2 = (x2 * x2).sum(-1, keepdim=True)
    y2 = y * y * l2
    z2 = z * z * l2
    k = th.sign(rr) * rr * rr * x2
    cond_1 = th.sign(z) * a2 * z2 > k
    cond_2 = th.sign(y) * a2 * y2 < k
    term_1 = th.sqrt(th.clamp(x2 + z2, min=EPSILON)) * il2 - r2
    term_2 = th.sqrt(th.clamp(x2 + y2, min=EPSILON)) * il2 - r1
    term_3 = (th.sqrt(th.clamp(x2 * a2 * il2, min=EPSILON)) + y * rr) * il2 - r1
    base_sdf = th.where(cond_1, term_1, th.where(cond_2, term_2, term_3))
    return base_sdf


def sdf3d_inexact_ellipsoid(points, r):
    """
    Calculates an approximate signed distance from 3D points to the surface of an ellipsoid.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        r (torch.Tensor): A tensor of shape [batch, 3] representing the radii of the ellipsoid along each axis.

    Returns:
        torch.Tensor: A tensor containing the approximate signed distances of each point to the ellipsoid surface.
    """
    r = th.where(r == 0, EPSILON, r)
    r = r[..., None, :]
    k0 = th.norm(points / r, dim=-1)
    k1 = th.norm(points / (r * r), dim=-1)
    base_sdf = k0 * (k0 - 1.0) / k1
    return base_sdf


def sdf3d_revolved_vesica(points, a, b, w):
    """
    Calculates the signed distance from 3D points to the surface of a revolved Vesica Piscis shape.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor of shape [batch, 3] representing the center of one sphere in the Vesica Piscis.
        b (torch.Tensor): A tensor of shape [batch, 3] representing the center of the other sphere in the Vesica Piscis.
        w (torch.Tensor): A tensor of shape [batch, 1] representing the radius of the spheres.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the revolved Vesica Piscis surface.
    """
    a = a[..., None, :]
    b = b[..., None, :]
    w = w[..., None, :]
    c = (a + b) * 0.5
    l = th.norm(b - a, dim=-1, keepdim=True)
    v = (b - a) / (l + EPSILON)
    y = ((points - c) * v).sum(-1, keepdim=True)
    q = th.stack([th.norm(points - c - y * v, dim=-1), th.abs(y)[..., 0]], dim=-1)
    r = 0.5 * l
    d = 0.5 * (r * r - w * w) / (w + EPSILON)
    q_1 = q.clone()
    r = r[..., 0]
    d = d[..., 0]
    w = w[..., 0]
    q_1[..., 1] = q_1[..., 1] - r
    term_1 = th.norm(q_1, dim=-1)
    q_2 = q.clone()
    q_2[..., 0] = q_2[..., 0] + d
    term_2 = th.norm(q_2, dim=-1) - (d + w)
    base_sdf = th.where(r * q[..., 0] < d * (q[..., 1] - r), term_1, term_2)
    return base_sdf


def sdf3d_rhombus(points, la, lb, h, ra):
    """
    Computes the signed distance field (SDF) for a rhombus shape in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        la (torch.Tensor): A tensor representing the length of the first diagonal of the rhombus.
        lb (torch.Tensor): A tensor representing the length of the second diagonal of the rhombus.
        h (torch.Tensor): A tensor representing the height of the rhombus from its base.
        ra (torch.Tensor): A tensor representing the rounding radius of the rhombus edges.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the rhombus surface.
    """
    p = th.abs(points)
    b = th.stack([la, lb], dim=-1)
    f = th.clamp(
        ndot(b, b - 2 * p[..., :2]) / ((b * b).sum(-1) + EPSILON), min=-1.0, max=1.0
    )
    f_factor = th.stack([1.0 - f, 1.0 + f], dim=-1)
    sign_term = th.sign(
        p[..., 0] * b[..., 1] + p[..., 1] * b[..., 0] - b[..., 0] * b[..., 1]
    )
    q_1 = th.norm(p[..., :2] - 0.5 * b * f_factor, dim=-1) * sign_term - ra
    q_2 = p[..., 2] - h
    q = th.stack(
        [
            th.norm(p[..., :2] - 0.5 * b * f_factor, dim=-1) * sign_term - ra,
            p[..., 2] - h,
        ],
        dim=-1,
    )
    base_sdf = th.clamp(th.amax(q, dim=-1), max=0.0) + th.norm(
        th.clamp(q, min=0.0), dim=-1
    )
    return base_sdf


def sdf3d_octahedron(points, s):
    """
    Computes the signed distance field (SDF) for an octahedron shape in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        s (torch.Tensor): A tensor of shape [batch, 1] representing the side length of the octahedron.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the octahedron surface.
    """
    p = th.abs(points)
    m = p[..., 0] + p[..., 1] + p[..., 2] - s
    m = m[..., :, None]
    q_1 = p.clone()
    q_2 = p.clone()[..., [1, 2, 0]]
    q_3 = p.clone()[..., [2, 0, 1]]
    cond_1 = 3.0 * p[..., 0:1] < m
    cond_2 = 3.0 * p[..., 1:2] < m
    cond_3 = 3.0 * p[..., 2:3] < m
    q = th.where(cond_1, q_1, th.where(cond_2, q_2, q_3))
    k = th.clamp(th.clamp(0.5 * (q[..., 2] - q[..., 1] + s), min=0.0), max=s)
    base_sdf = th.norm(
        th.stack([q[..., 0], q[..., 1] - s + k, q[..., 2] - k], dim=-1), dim=-1
    )
    sdf = th.where(
        cond_1[..., 0] | cond_2[..., 0] | cond_3[..., 0], base_sdf, m[..., 0] * COS_30
    )
    return sdf


def sdf3d_inexact_octahedron(points, s):
    """
    Computes an inexact signed distance field (SDF) for an octahedron shape in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        s (torch.Tensor): A tensor of shape [batch, 1] representing the side length of the octahedron.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the octahedron surface.
    """
    p = th.abs(points)
    base_sdf = (p[..., 0] + p[..., 1] + p[..., 2] - s) * TAN_30
    return base_sdf


def sdf3d_pyramid(points, h):
    """
    Computes the signed distance field (SDF) for a pyramid shape in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        h (torch.Tensor): A tensor of shape [batch, 1] representing the height of the pyramid.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the pyramid surface.
    """
    m2 = h * h + 0.25
    abs_points = th.abs(points[..., :2])
    cond = abs_points[..., 1:2] > abs_points[..., 0:1]
    swapped_points = th.where(cond, abs_points[..., [1, 0]], abs_points[..., :2])
    mod_points = swapped_points - 0.5
    z_coord = points[..., 2:3] + 0.5
    points = th.cat([mod_points, z_coord], dim=-1)
    q = th.stack(
        [
            points[..., 1],
            h * points[..., 2] - 0.5 * points[..., 0],
            h * points[..., 0] + 0.5 * points[..., 2],
        ],
        dim=-1,
    )
    s = th.clamp(-q[..., 0], min=0.0)
    t = th.clamp((q[..., 1] - 0.5 * points[..., 1]) / (m2 + 0.25), min=0.0, max=1.0)
    a = m2 * (q[..., 0] + s) * (q[..., 0] + s) + q[..., 1] * q[..., 1]
    b = m2 * (q[..., 0] + 0.5 * t) * (q[..., 0] + 0.5 * t) + (q[..., 1] - m2 * t) * (
        q[..., 1] - m2 * t
    )
    cond = th.minimum(q[..., 1], -q[..., 0] * m2 - 0.5 * q[..., 1]) > 0.0
    d2 = th.where(cond, 0.0, th.minimum(a, b))
    base_sdf = th.sqrt(
        th.clamp((d2 + q[..., 2] * q[..., 2]) / m2, min=EPSILON)
    ) * th.sign(th.maximum(q[..., 2], -points[..., 2]))
    return base_sdf


def sdf3d_triangle(points, a, b, c):
    """
    Computes the signed distance field (SDF) for a triangle shape in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor of shape [batch, 3] representing one vertex of the triangle.
        b (torch.Tensor): A tensor of shape [batch, 3] representing another vertex of the triangle.
        c (torch.Tensor): A tensor of shape [batch, 3] representing the third vertex of the triangle.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the triangle surface.
    """
    ba = b - a
    pa = points - a[..., None, :]
    cb = c - b
    pb = points - b[..., None, :]
    ac = a - c
    pc = points - c[..., None, :]
    nor = th.cross(ba, ac)
    term_1 = th.sign((th.cross(ba, nor)[..., None, :] * pa).sum(-1))
    term_2 = th.sign((th.cross(cb, nor)[..., None, :] * pb).sum(-1))
    term_3 = th.sign((th.cross(ac, nor)[..., None, :] * pc).sum(-1))
    cond = term_1 + term_2 + term_3 < 2.0
    term_1 = (ba[..., None, :] * pa).sum(-1) / (
        (ba * ba).sum(-1, keepdim=True) + EPSILON
    )
    term_2 = (cb[..., None, :] * pb).sum(-1) / (
        (cb * cb).sum(-1, keepdim=True) + EPSILON
    )
    term_3 = (ac[..., None, :] * pc).sum(-1) / (
        (ac * ac).sum(-1, keepdim=True) + EPSILON
    )
    sub_stack = th.stack([pa, pb, pc], dim=-1)
    mult_stack = th.stack([ba, cb, ac], dim=-1)[..., None, :, :]
    term_stack = th.stack([term_1, term_2, term_3], dim=-1)[..., None, :]
    out = mult_stack * th.clamp(term_stack, min=0.0, max=1.0) - sub_stack
    out = th.amin(out, dim=-1)
    out = (out * out).sum(-1)
    else_sdf = (nor[..., None, :] * pa).sum(-1) ** 2 / (
        (nor * nor).sum(-1, keepdim=True) + EPSILON
    )
    base_sdf = th.where(cond, out, else_sdf)
    base_sdf = th.sqrt(th.clamp(base_sdf, min=EPSILON))
    return base_sdf


def sdf3d_quadrilateral(points, a, b, c, d):
    """
    Computes the signed distance field (SDF) for a quadrilateral shape in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        a (torch.Tensor): A tensor representing one vertex of the quadrilateral.
        b (torch.Tensor): A tensor representing another vertex of the quadrilateral.
        c (torch.Tensor): A tensor representing the third vertex of the quadrilateral.
        d (torch.Tensor): A tensor representing the fourth vertex of the quadrilateral.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the quadrilateral surface.
    """
    papbpcpd = points[..., None] - th.stack([a, b, c, d], dim=-1)[..., None, :, :]
    pa, pb, pc, pd = th.unbind(papbpcpd, dim=-1)
    ba = b - a
    cb = c - b
    dc = d - c
    ad = a - d
    nor = th.cross(ba, ad)
    term_1 = th.sign((th.cross(ba, nor)[..., None, :] * pa).sum(-1))
    term_2 = th.sign((th.cross(cb, nor)[..., None, :] * pb).sum(-1))
    term_3 = th.sign((th.cross(dc, nor)[..., None, :] * pc).sum(-1))
    term_4 = th.sign((th.cross(ad, nor)[..., None, :] * pd).sum(-1))
    cond = term_1 + term_2 + term_3 + term_4 < 3.0
    term_1 = (ba[..., None, :] * pa).sum(-1) / (
        (ba * ba).sum(-1, keepdim=True) + EPSILON
    )
    term_2 = (cb[..., None, :] * pb).sum(-1) / (
        (cb * cb).sum(-1, keepdim=True) + EPSILON
    )
    term_3 = (dc[..., None, :] * pc).sum(-1) / (
        (dc * dc).sum(-1, keepdim=True) + EPSILON
    )
    term_4 = (ad[..., None, :] * pd).sum(-1) / (
        (ad * ad).sum(-1, keepdim=True) + EPSILON
    )
    sub_stack = th.stack([pa, pb, pc, pd], dim=-1)
    mult_stack = th.stack([ba, cb, dc, ad], dim=-1)[..., None, :, :]
    term_stack = th.stack([term_1, term_2, term_3, term_4], dim=-1)[..., None, :]
    out = mult_stack * th.clamp(term_stack, min=0.0, max=1.0) - sub_stack
    out = th.amin(out, dim=-1)
    out = (out * out).sum(-1)
    else_sdf = (nor[..., None, :] * pa).sum(-1) ** 2 / (
        (nor * nor).sum(-1, keepdim=True) + EPSILON
    )
    base_sdf = th.where(cond, out, else_sdf)
    base_sdf = th.sqrt(th.clamp(base_sdf, min=EPSILON))
    return base_sdf


def sdf3d_no_param_cuboid(points):
    """
    Computes the signed distance field (SDF) for a cuboid shape with no additional parameters in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the cuboid surface.
    """
    points = th.abs(points)
    points -= 0.5
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + th.clip(
        th.amax(points, -1), max=0
    )
    return base_sdf


def sdf3d_no_param_sphere(points):
    """
    Computes the signed distance field (SDF) for a sphere shape with no additional parameters in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the sphere surface.
    """
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - 0.5
    return base_sdf


def sdf3d_no_param_cylinder(points):
    """
    Computes the signed distance field (SDF) for a cylinder shape with no additional parameters in 3D.

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the cylinder surface.
    """
    r = 0.5
    h = 0.5
    xy_vec = th.norm(points[..., :2], dim=-1) - r
    height = th.abs(points[..., 2]) - h
    vec2 = th.stack([xy_vec, height], -1)
    base_sdf = th.amax(vec2, -1) + th.norm(th.clip(vec2, min=0.0) + EPSILON, -1)
    return base_sdf


def sdf3d_inexact_super_quadrics(points, skew_vec, epsilon_1, epsilon_2):
    """
    Computes an inexact signed distance field (SDF) for superquadric shapes in 3D.
    Reference: https://arxiv.org/pdf/2303.13190.pdf

    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        skew_vec (torch.Tensor): A tensor of shape [batch, 3] representing the skew parameters of the shape.
        epsilon_1 (torch.Tensor): A tensor of shape [batch, 1] representing the first epsilon parameter.
        epsilon_2 (torch.Tensor): A tensor of shape [batch, 1] representing the second epsilon parameter.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the superquadric surface.
    """
    points = th.abs(points)
    out_0 = (points[..., 0] / skew_vec[..., 0:1]) ** (2 / (epsilon_2 + EPSILON))
    out_1 = (points[..., 1] / skew_vec[..., 1:2]) ** (2 / (epsilon_2 + EPSILON))
    out_2 = (points[..., 2] / skew_vec[..., 2:3]) ** (2 / (epsilon_1 + EPSILON))
    inside_term = (th.abs(out_0 + out_1) + EPSILON) ** (epsilon_2 / (epsilon_1 + EPSILON))
    base_sdf = 1 - (th.abs(inside_term + out_2) + EPSILON) ** (-epsilon_1 / 2.0)
    return base_sdf


def sdf3d_inexact_anisotropic_gaussian(points, center, axial_radii, scale_constant):
    """
    Computes an inexact signed distance field (SDF) for an anisotropic Gaussian shape in 3D.
    Reference: https://arxiv.org/pdf/1904.06447.pdf
    Parameters:
        points (torch.Tensor): A tensor of shape [batch, num_points, 3] representing 3D points.
        center (torch.Tensor): A tensor of shape [batch, 3] representing the center of the Gaussian distribution.
        axial_radii (torch.Tensor): A tensor of shape [batch, 3] representing the axial radii of the Gaussian distribution.
        scale_constant (torch.Tensor): A tensor of shape [batch, 1] representing the scale constant of the Gaussian distribution.

    Returns:
        torch.Tensor: A tensor containing the signed distances of each point to the anisotropic Gaussian surface.
    """
    points = -((points - center[..., None, :]) ** 2) / (
        2 * axial_radii[..., None, :] ** 2
    )
    base_sdf = scale_constant[..., :] * th.exp(points.sum(-1))
    return base_sdf
