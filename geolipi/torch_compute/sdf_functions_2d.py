import numpy as np
import torch as th
from .common import EPSILON, ACOS_EPSILON

SQRT_3 = np.sqrt(3, dtype=np.float32)
SQRT_2 = np.sqrt(2, dtype=np.float32)
PENT_VEC = np.array([0.809016994, 0.587785252, 0.726542528], dtype=np.float32)
HEX_VEC = np.array([-0.866025404, 0.5, 0.577350269], dtype=np.float32)
OCT_VEC = np.array([-0.9238795325, 0.3826834323, 0.4142135623], dtype=np.float32)
HEXAGRAM_VEC = np.array(
    [-0.5, 0.8660254038, 0.5773502692, 1.7320508076], dtype=np.float32
)
STAR_VEC = np.array([0.809016994375, -0.587785252292], dtype=np.float32)


def ndot(vec_a, vec_b):
    output = vec_a[..., 0] * vec_b[..., 0] - vec_a[..., 1] * vec_b[..., 1]
    return output


def sdf2d_circle(points, radius):
    """
    Computes the signed distance field for a 2D circle.

    Parameters:
        points (Tensor): The coordinates at which to evaluate the SDF, shape [batch, num_points, 2].
        radius (Tensor): The radius of the circle, shape [batch, 1].

    Returns:
        Tensor: The calculated SDF values for each point.
    """
    base_sdf = th.norm(points, dim=-1)
    base_sdf = base_sdf - radius
    return base_sdf


def sdf2d_rounded_box(points, bounds, radius):
    """
    Computes the SDF for a 2D rounded box (rectangle with rounded corners).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        bounds (Tensor): Dimensions of the box, shape [batch, 2].
        radius (Tensor): Radii of the rounded corners, shape [batch, 4].

    Returns:
        Tensor: The SDF values for the rounded box.
    """
    # Select appropriate radii based on the sign of p
    radius = radius[..., None, :]
    bounds = bounds[..., None, :]
    radius_xy = th.where(points[..., 0:1] > 0, radius[..., 0:2], radius[..., 2:4])
    radius_x = th.where(points[..., 1:2] > 0, radius_xy[..., 0:1], radius_xy[..., 1:2])
    q = th.abs(points) - bounds + radius_x
    length_q = th.norm(th.clamp(q, min=0.0), dim=-1)
    sd = th.clamp(th.max(q[..., 0], q[..., 1]), max=0.0) + length_q - radius_x[..., 0]
    return sd.squeeze(-1)


def sdf2d_box(points, size):
    """
    Computes the SDF for a 2D axis-aligned rectangle.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        size (Tensor): Dimensions of the box, shape [batch, 2].

    Returns:
        Tensor: The SDF values for the rectangle.
    """
    points = th.abs(points)
    size = size[..., None, :] / 2.0
    points = points - size
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + th.clip(
        th.amax(points, -1), max=0
    )
    return base_sdf


def sdf2d_oriented_box(points, start_point, end_point, thickness):
    """
    Computes the SDF for a 2D oriented box (rotated rectangle).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        start_point (Tensor): One corner of the box, shape [batch, 2].
        end_point (Tensor): Opposite corner of the box, shape [batch, 2].
        thickness (Tensor): Thickness of the box, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the oriented box.
    """
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    diff = end_point - start_point
    length = diff.norm(dim=-1, keepdim=True)
    d = diff / (length + EPSILON)
    q = points - ((start_point + end_point)[..., None, :] * 0.5)
    mat = th.stack([d[..., 0], -d[..., 1], d[..., 1], d[..., 0]], dim=-1).view(-1, 2, 2)
    q = th.bmm(q, mat)
    q = th.abs(q)
    lt = th.stack([length, thickness], -1) * 0.5
    q = q - lt
    sdf = th.norm(th.clamp(q, min=0.0), dim=-1) + th.clamp(th.amax(q, dim=-1), max=0.0)
    return sdf


def sdf2d_rhombus(points, size):
    """
    Computes the SDF for a 2D rhombus (diamond shape).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        size (Tensor): Dimensions of the rhombus, shape [batch, 2].

    Returns:
        Tensor: The SDF values for the rhombus.
    """
    p = th.abs(points)
    size = size[..., None, :]
    h = th.clamp(ndot(size - 2.0 * p, size) / ((size * size).sum(-1) + EPSILON), -1, 1)
    d = th.norm(p - 0.5 * size * th.stack([1.0 - h, 1.0 + h], -1), dim=-1)
    sdf = d * th.sign(
        p[..., 0] * size[..., 1]
        + p[..., 1] * size[..., 0]
        - size[..., 0] * size[..., 1]
    )
    return sdf


def sdf2d_trapezoid(points, r1, r2, height):
    """
    Computes the SDF for a 2D trapezoid.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r1 (Tensor): Radius at one end, shape [batch, 1].
        r2 (Tensor): Radius at the other end, shape [batch, 1].
        height (Tensor): Height of the trapezoid, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the trapezoid.
    """
    k1 = th.stack([r2, height], -1)
    k2 = th.stack([r2 - r1, 2 * height], -1)
    points[..., 0] = th.abs(points[..., 0])
    ca = th.stack(
        [
            points[..., 0]
            - th.minimum(points[..., 0], th.where(points[..., 1] < 0, r1, r2)),
            th.abs(points[..., 1]) - height,
        ],
        -1,
    )
    cb = (
        points
        - k1
        + k2
        * (
            th.clamp(((k1 - points) * k2).sum(-1) / ((k2 * k2).sum(-1) + EPSILON), 0, 1)
        ).unsqueeze(-1)
    )
    s = th.where(th.logical_and(cb[..., 0] < 0, ca[..., 1] < 0), -1, 1)
    sdf = s * th.sqrt(th.minimum((cb * cb).sum(-1), (ca * ca).sum(-1)) + EPSILON)
    return sdf


def sdf2d_parallelogram(points, width, height, skew):
    """
    Computes the SDF for a 2D parallelogram.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        width (Tensor): Width of the parallelogram, shape [batch, 1].
        height (Tensor): Height of the parallelogram, shape [batch, 1].
        skew (Tensor): Skew factor, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the parallelogram.
    """
    e = th.stack([skew, height], -1)
    p = th.where(points[..., 1:2] < 0, -points, points)
    w = p - e
    w[..., 0] = w[..., 0] - th.clamp(w[..., 0].clone(), -width, width)
    d = th.stack([(w * w).sum(-1), -w[..., 1]], -1)
    s = p[..., 0:1] * e[..., 1:2] - p[..., 1:2] * e[..., 0:1]
    p = th.where(s < 0, -p, -p)
    v = p.clone()
    v[..., 0] = v[..., 0] - width
    const = th.clamp((v * e).sum(-1) / ((e * e).sum(-1) + EPSILON), -1.0, 1.0)
    v = v - (e * const[..., None])
    d = th.minimum(d, th.stack([(v * v).sum(-1), width * height - abs(s[..., 0])], -1))
    sdf = th.sqrt(d[..., 0] + EPSILON) * -th.sign(d[..., 1])
    return sdf


def sdf2d_equilateral_triangle(points, side_length):
    """
    Computes the SDF for a 2D equilateral triangle.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        side_length (Tensor): Length of each side of the triangle, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the equilateral triangle.
    """
    k = th.tensor(SQRT_3).to(points.device)
    points[..., 0] = th.abs(points[..., 0].clone()) - side_length
    points[..., 1] = points[..., 1] + side_length / k
    condition = points[..., 0] + (k * points[..., 1]) > 0
    new_points = (
        th.stack(
            [points[..., 0] - k * points[..., 1], -k * points[..., 0] - points[..., 1]],
            -1,
        )
        / 2.0
    )
    points = th.where(condition.unsqueeze(-1), new_points, points)
    points[..., 0] = points[..., 0] - th.clamp(
        th.clamp(points[..., 0].clone(), min=-2 * side_length), max=0
    )
    sdf = -th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf


def sdf2d_isosceles_triangle(points, wi_hi):
    """
    Computes the SDF for a 2D isosceles triangle.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        wi_hi (Tensor): Width and height of the triangle, shape [batch, 2].

    Returns:
        Tensor: The SDF values for the isosceles triangle.
    """
    points[..., 0] = th.abs(points[..., 0].clone())
    wi_hi_ex = wi_hi[..., None, :]
    points[..., 1] = points[..., 1] + wi_hi_ex[..., :, 1] / 2.0
    a = points - wi_hi_ex * th.clamp(
        (points * wi_hi_ex).sum(-1, keepdim=True) / ((wi_hi * wi_hi).sum(-1) + EPSILON),
        0,
        1,
    )
    b = points.clone()
    b[..., 0] = b[..., 0] - wi_hi_ex[..., :, 0] * th.clamp(
        points[..., 0] / (wi_hi_ex[..., :, 0] + EPSILON), 0, 1
    )
    b[..., 1] = b[..., 1] - wi_hi_ex[..., :, 1]
    s = -th.sign(wi_hi_ex[..., :, 1])
    d = th.minimum(
        th.stack(
            [
                (a * a).sum(-1),
                s
                * (
                    points[..., 0] * wi_hi_ex[..., :, 1]
                    - points[..., 1] * wi_hi_ex[..., :, 0]
                ),
            ],
            -1,
        ),
        th.stack([(b * b).sum(-1), s * (points[..., 1] - wi_hi_ex[..., :, 1])], -1),
    )
    sdf = th.sqrt(th.clamp(d[..., 0], min=EPSILON)) * -th.sign(d[..., 1])
    return sdf


def sdf2d_triangle(points, p0, p1, p2):
    """
    Computes the SDF for a general 2D triangle specified by three points.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        p0 (Tensor): First vertex of the triangle, shape [batch, 2].
        p1 (Tensor): Second vertex of the triangle, shape [batch, 2].
        p2 (Tensor): Third vertex of the triangle, shape [batch, 2].

    Returns:
        Tensor: The SDF values for the triangle.
    """
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    all_points = th.stack([p0, p1, p2], -1)[..., None, :, :]  # B, 2, 3
    all_vs = points.unsqueeze(-1) - all_points  # B, N, 2, 3
    all_es = th.stack([e0, e1, e2], -1)[..., None, :, :]  # B, 2, 3
    all_pqs = all_vs - all_es * th.clamp(
        (all_vs * all_es).sum(-2) / ((all_es * all_es).sum(-2) + EPSILON),
        min=0.0,
        max=1.0,
    ).unsqueeze(-2)
    s = th.sign(
        all_es[..., 0, 0] * all_es[..., 1, 2] - all_es[..., 1, 0] * all_es[..., 0, 2]
    )[..., None, :]
    d_0 = (all_pqs * all_pqs).sum(-2)
    d_1 = s * (
        all_vs[..., 0, :] * all_es[..., 1, :] - all_vs[..., 1, :] * all_es[..., 0, :]
    )
    d = th.stack([d_0, d_1], -1)
    d = th.amin(d, -2)
    sdf = -th.sqrt(d[..., 0] + EPSILON) * th.sign(d[..., 1])
    return sdf


def sdf2d_uneven_capsule(points, r1, r2, h):
    """
    Computes the SDF for a 2D uneven capsule (cylinder with hemispherical ends of different radii).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r1 (Tensor): Radius of one hemispherical end, shape [batch, 1].
        r2 (Tensor): Radius of the other hemispherical end, shape [batch, 1].
        h (Tensor): Height of the capsule, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the uneven capsule.
    """
    points[..., 0] = th.abs(points[..., 0])
    points[..., 1] = points[..., 1] + (h + r2 - r1) / 2
    b = (r1 - r2) / (h + EPSILON)
    a = th.sqrt(th.clamp(1 - b * b, min=EPSILON))
    k = points[..., 0] * (-b) + points[..., 1] * a
    pn = points.clone()
    pn[..., 1] = pn[..., 1] - h
    sdf = points[..., 0] * a + points[..., 1] * b - r1
    sdf = th.where(k > a * h, th.norm(pn, dim=-1) - r2, sdf)
    sdf = th.where(k < 0, th.norm(points, dim=-1) - r1, sdf)
    return sdf


def sdf2d_regular_pentagon(points, r):
    """
    Computes the SDF for a 2D regular pentagon.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of the circumscribing circle of the pentagon, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the regular pentagon.
    """
    k = th.tensor(PENT_VEC).to(points.device)
    points[..., 0] = th.abs(points[..., 0])
    k2 = k.clone()
    k2[0] *= -1
    points = points - 2 * (
        th.clamp((points * k2[:2]).sum(-1), max=0.0)[..., None] * k2[..., :2]
    )
    points = points - 2 * (
        th.clamp((points * k[:2]).sum(-1), max=0.0)[..., None] * k[..., :2]
    )
    points[..., 0] = points[..., 0] - th.clamp(
        points[..., 0].clone(), -r * k[2], r * k[2]
    )
    points[..., 1] = points[..., 1] - r
    sdf = th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf


def sdf2d_regular_hexagon(points, r):
    """
    Computes the SDF for a 2D regular hexagon.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of the circumscribing circle of the hexagon, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the regular hexagon.
    """
    k = th.tensor(HEX_VEC).to(points.device)
    points = th.abs(points)
    points = (
        points - 2 * th.clamp((points * k[:2]).sum(-1), max=0.0)[..., None] * k[..., :2]
    )
    points[..., 0] = points[..., 0] - th.clamp(
        points[..., 0].clone(), -r * k[2], r * k[2]
    )
    points[..., 1] = points[..., 1] - r
    sdf = th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf


def sdf2d_regular_octagon(points, r):
    """
    Computes the SDF for a 2D regular octagon.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of the circumscribing circle of the octagon, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the regular octagon.
    """
    k = th.tensor(OCT_VEC).to(points.device)
    points = th.abs(points)
    k2 = k.clone()
    k2[0] *= -1
    points = (
        points - 2 * th.clamp((points * k[:2]).sum(-1), max=0.0)[..., None] * k[..., :2]
    )
    points = (
        points
        - 2 * th.clamp((points * k2[:2]).sum(-1), max=0.0)[..., None] * k2[..., :2]
    )
    points[..., 0] = points[..., 0] - th.clamp(
        points[..., 0].clone(), -r * k[2], r * k[2]
    )
    points[..., 1] = points[..., 1] - r
    sdf = th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf


def sdf2d_hexagram(points, r):
    """
    Computes the SDF for a 2D hexagram (six-pointed star).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of the circumscribing circle of the hexagram, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the hexagram.
    """
    k = th.tensor(HEXAGRAM_VEC).to(points.device)
    points = th.abs(points)
    k2 = k.flip(0)
    points = (
        points - 2 * th.clamp((points * k[:2]).sum(-1), max=0.0)[..., None] * k[..., :2]
    )
    points = (
        points
        - 2 * th.clamp((points * k2[2:]).sum(-1), max=0.0)[..., None] * k2[..., 2:]
    )
    points[..., 0] = points[..., 0] - th.clamp(
        points[..., 0].clone(), r * k[2], r * k[3]
    )
    points[..., 1] = points[..., 1] - r
    sdf = th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf


def sdf2d_star_5(points, r, rf):
    """
    Computes the SDF for a 2D five-pointed star.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Outer radius of the star, shape [batch, 1].
        rf (Tensor): Factor to control the inner radius, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the five-pointed star.
    """
    k = th.tensor(STAR_VEC).to(points.device)
    k2 = k.clone()
    k2[0] = k2[0] * -1
    points[..., 0] = th.abs(points[..., 0])
    points = points - 2 * th.clamp((points * k).sum(-1), min=0.0)[..., None] * k[..., :]
    points = (
        points - 2 * th.clamp((points * k2).sum(-1), min=0.0)[..., None] * k2[..., :]
    )
    points[..., 0] = th.abs(points[..., 0])
    points[..., 1] = points[..., 1] - r
    ba = k2.clone() * -1
    ba = rf * ba.unsqueeze(0)
    ba[:, 1] = ba[:, 1] - 1
    ba = ba[..., None, :]
    h = th.clamp(
        th.clamp((points * ba).sum(-1) / ((ba * ba).sum(-1) + EPSILON), min=0.0), max=r
    )
    sdf = th.norm(points - ba * h[..., :, None], dim=-1) * th.sign(
        points[..., 1] * ba[..., :, 0] - points[..., 0] * ba[..., :, 1]
    )
    sdf = sdf[0, ...]
    return sdf


def sdf2d_regular_star(points, r, n, m):
    """
    Computes the SDF for a 2D regular star with specified points and symmetry.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of the star, shape [batch, 1].
        n (Tensor): Number of points of the star, shape [batch, 1].
        m (Tensor): Symmetry factor of the star, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the regular star.
    """
    an = np.pi / n
    en = np.pi / m
    acs = th.cat([th.cos(an), th.sin(an)], -1)
    ecs = th.cat([th.cos(en), th.sin(en)], -1)[..., None, :]
    bn = th.remainder(th.atan2(points[..., 0], points[..., 1]), (2 * an)) - an
    p = th.norm(points[..., :], dim=-1, keepdim=True) * th.stack(
        [th.cos(bn), th.abs(th.sin(bn))], -1
    )
    p = p - (r * acs)[..., None, :]
    max_val = r[..., 0] * acs[..., 1] / ecs[..., 0, 1]
    p = p + ecs * th.clamp(
        th.clamp(-(p * ecs).sum(-1, keepdim=True), min=0), max=max_val[..., None, None]
    )
    sdf = th.norm(p, dim=-1) * th.sign(p[..., 0])
    return sdf


def sdf2d_pie(points, c, r):
    """
    Computes the SDF for a 2D pie (circular sector).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        c (Tensor): Central angle of the sector, shape [batch, 2].
        r (Tensor): Radius of the pie, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the pie.
    """
    c = c[..., None, :]
    points[..., 0] = th.abs(points[..., 0])
    l = th.norm(points, dim=-1) - r
    z = th.clamp(th.clamp((points * c).sum(-1), min=0), max=r)
    m = th.norm(
        points - c * th.clamp(th.clamp((points * c).sum(-1), min=0), max=r)[..., None],
        dim=-1,
    )
    sign = th.sign(c[..., 1] * points[..., 0] - c[..., 0] * points[..., 1])
    sdf = th.maximum(l, m * sign)
    return sdf


def sdf2d_cut_disk(points, r, h):
    """
    Computes the SDF for a 2D cut disk (circular segment).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of the disk, shape [batch, 1].
        h (Tensor): Height of the cut segment, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the cut disk.
    """
    w = th.sqrt(th.clamp(r * r - h * h, min=EPSILON))
    points[..., 0] = th.abs(points[..., 0])
    s = th.maximum(
        (h - r) * points[..., 0] * points[..., 0]
        + w * w * (h + r - 2 * points[..., 1]),
        h * points[..., 0] - w * points[..., 1],
    )
    pn = points.clone()
    pn[..., 0] = pn[..., 0] - w
    pn[..., 1] = pn[..., 1] - h
    sdf = th.where(points[..., 0] < w, h - points[..., 1], th.norm(pn, dim=-1))
    sdf = th.where(s < 0, th.norm(points, dim=-1) - r, sdf)
    return sdf


def sdf2d_arc(points, angle, ra, rb):
    """
    Computes the SDF for a 2D arc.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        angle (Tensor): Central angle of the arc, shape [batch, 1].
        ra (Tensor): Radius of the arc, shape [batch, 1].
        rb (Tensor): Width of the arc, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the arc.
    """
    sc = th.cat([th.sin(angle), th.cos(angle)], -1)
    points[..., 0] = th.abs(points[..., 0])
    pointer = sc[..., 1:2] * points[..., 0] > sc[..., 0:1] * points[..., 1]
    sdf = th.where(
        pointer,
        th.norm(points - sc[..., None, :] * ra[..., None, :], dim=-1),
        th.abs(th.norm(points, dim=-1) - ra) - rb,
    )
    return sdf


def sdf2d_horse_shoe(points, angle, r, w):
    """
    Computes the SDF for a 2D horseshoe shape.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        angle (Tensor): Central angle of the horseshoe, shape [batch, 1].
        r (Tensor): Radius of the horseshoe, shape [batch, 1].
        w (Tensor): Width of the horseshoe, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the horseshoe shape.
    """
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    c = th.cat([th.cos(angle), th.sin(angle)], -1)
    points[..., 0] = th.abs(points[..., 0])
    l = th.norm(points, dim=-1)
    mat = th.stack([-c[..., 0], c[..., 1], c[..., 1], c[..., 0]], -1).view(-1, 2, 2)
    points = th.bmm(points, mat)
    out = points.clone()
    m = points[..., 0] > 0
    out[..., 0] = th.where(
        (points[..., 1] > 0) | m, points[..., 0], l * th.sign(-c[..., 0:1])
    )
    out[..., 1] = th.where(m, points[..., 1], l)
    out[..., 0] = out[..., 0] - w[..., 0:1]
    out[..., 1] = th.abs(out[..., 1] - r) - w[..., 1:2]
    sdf = th.norm(th.clamp(out, min=0.0), dim=-1) + th.clamp(
        th.amax(out, dim=-1), max=0.0
    )
    return sdf


def sdf2d_vesica(points, r, d):
    """
    Computes the SDF for a 2D vesica piscis shape (intersection of two circles).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        r (Tensor): Radius of each circle, shape [batch, 1].
        d (Tensor): Distance between the centers of the circles, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the vesica piscis.
    """
    points = th.abs(points)
    b = th.sqrt(th.clamp(r * r - d * d, min=EPSILON))
    p1 = th.clone(points)
    p2 = th.clone(points)
    p1[..., 1] = p1[..., 1] - b
    p2[..., 0] = p2[..., 0] + d
    sdf = th.where(
        (points[..., 1] - b) * d > points[..., 0] * b,
        th.norm(p1, dim=-1),
        th.norm(p2, dim=-1) - r,
    )
    return sdf


def sdf2d_oriented_vesica(points, a, b, w):
    """
    Computes the SDF for a 2D oriented vesica piscis.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        a (Tensor): One center of the vesica piscis, shape [batch, 2].
        b (Tensor): Other center of the vesica piscis, shape [batch, 2].
        w (Tensor): Width of the region connecting the circles, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the oriented vesica piscis.
    """
    r = 0.5 * th.norm(b - a, dim=-1, keepdim=True)
    d = 0.5 * (r**2 - w**2) / (w + EPSILON)
    v = (b - a) / (r + EPSILON)
    c = 0.5 * (b + a)
    mat = th.stack([v[..., 1], v[..., 0], -v[..., 0], v[..., 1]], dim=-1).reshape(
        -1, 2, 2
    )
    q = 0.5 * th.abs(th.matmul((points - c[..., None, :]), mat))
    cond = r * q[..., 0] < d * (q[..., 1] - r)
    h = th.where(
        cond.unsqueeze(-1),
        th.zeros_like(cond, dtype=th.float32).unsqueeze(-1),
        th.stack([-d, th.zeros_like(d, dtype=th.float32), d + w], dim=-1),
    )
    return th.norm(q - h[..., :2], dim=-1) - h[..., 2]


def sdf2d_moon(points, d, ra, rb):
    """
    Computes the SDF for a 2D moon shape (crescent).

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        d (Tensor): Distance between the centers of the circles forming the moon, shape [batch, 1].
        ra (Tensor): Radius of the first circle, shape [batch, 1].
        rb (Tensor): Radius of the second circle, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the moon shape.
    """
    points[..., 1] = th.abs(points[..., 1])
    a = (ra * ra - rb * rb + d * d) / (2.0 * d + EPSILON)
    b = th.sqrt(th.clamp(ra * ra - a * a, min=EPSILON))
    cond = d * (points[..., 0] * b - points[..., 1] * a) > d * d * th.clamp(
        b - points[..., 1], min=0.0
    )
    option_1 = th.norm(points - th.stack([a, b], -1), dim=-1)
    option_2 = th.norm(points, dim=-1) - ra
    points[..., 0] -= d
    option_3 = -(th.norm(points, dim=-1) - rb)
    sdf = th.where(cond, option_1, th.maximum(option_2, option_3))
    return sdf


def sdf2d_rounded_cross(points, h):
    """
    Computes the SDF for a 2D rounded cross.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        h (Tensor): Height of the arms of the cross, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the rounded cross.
    """
    k = 0.5 * (h + 1.0 / (h + EPSILON))
    points = th.abs(points)
    cond = (points[..., 0] < 1.0) & (points[..., 1] < points[..., 0] * (k - h) + h)
    # can remove three clones
    p1 = points.clone()
    p2 = points.clone()
    p3 = points.clone()
    p1[..., 0] = p1[..., 0] - 1.0
    p1[..., 1] = p1[..., 1] - k
    p2[..., 1] = p2[..., 1] - h
    p3[..., 0] = p3[..., 0] - 1.0
    all_p = th.stack([p1, p2, p3], -1)
    all_norm = th.norm(all_p, dim=-2)
    sdf = th.where(
        cond, k - all_norm[..., 0], th.minimum(all_norm[..., 1], all_norm[..., 2])
    )
    return sdf


def sdf2d_egg(points, ra, rb):
    """
    Computes the SDF for a 2D egg shape.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        ra (Tensor): Radius along one axis, shape [batch, 1].
        rb (Tensor): Radius along the other axis, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the egg shape.
    """
    k = th.tensor(SQRT_3).to(points.device)
    points[..., 0] = th.abs(points[..., 0])
    r = ra - rb
    cond_1 = points[..., 1] < 0.0
    cond_2 = k * (points[..., 0] + r) < points[..., 1]
    p2 = points.clone()
    p2[..., 1] = p2[..., 1] - (k * r)
    p3 = points.clone()
    p3[..., 0] = p3[..., 0] + r
    all_p = th.stack([points, p2, p3], -1)
    all_norm = th.norm(all_p, dim=-2)

    sdf = th.where(
        cond_1,
        all_norm[..., 0] - r,
        th.where(cond_2, all_norm[..., 1], all_norm[..., 2] - 2.0 * r),
    )
    sdf = sdf - rb
    return sdf


def sdf2d_heart(points):
    """
    Computes the SDF for a 2D heart shape.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].

    Returns:
        Tensor: The SDF values for the heart shape.
    """
    points[..., 1] += 0.5
    points[..., 0] = th.abs(points[..., 0])

    cond = points[..., 1] + points[..., 0] > 1.0
    sign = th.sign(points[..., 0] - points[..., 1])
    p2 = points.clone()
    p3 = points.clone()
    points[..., 0] = points[..., 0] - 0.25
    points[..., 1] = points[..., 1] - 0.75
    p2[..., 1] = p2[..., 1] - 1.0
    p3 = p3 - 0.5 * th.clamp(points.sum(-1, keepdim=True), min=0.0)
    all_p = th.stack([points, p2, p3], -1)
    all_norm = th.norm(all_p, dim=-2)
    sdf = th.where(
        cond,
        all_norm[..., 0] - SQRT_2 / 4.0,
        th.minimum(all_norm[..., 1], all_norm[..., 1]) * sign,
    )
    return sdf


def sdf2d_cross(points, b, r):
    """
    Computes the SDF for a 2D cross.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        b (Tensor): Size of the cross' arms, shape [batch, 2].
        r (Tensor): Radius of the rounding at the cross' center, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the cross.
    """
    points = th.abs(points)
    b = b[..., None, :]
    points = th.where(points[..., 1:2] > points[..., 0:1], points[..., [1, 0]], points)
    q = points - b
    k = th.amax(q, -1, keepdim=True)
    k_single = k[..., 0]
    w = th.where(k > 0.0, q, th.stack([b[..., 1] - points[..., 0], -k_single], -1))
    sdf = th.sign(k_single) * th.norm(th.clamp(w, min=0.0), dim=-1) + r
    return sdf


def sdf2d_rounded_x(points, w, r):
    """
    Computes the SDF for a 2D rounded 'X' shape.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        w (Tensor): Width of the arms of the 'X', shape [batch, 1].
        r (Tensor): Radius of the rounding at the intersections, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the rounded 'X'.
    """
    points = th.abs(points)
    sub = th.minimum(points[..., 0:1] + points[..., 1:2], w[..., None, :])
    p = points - sub * 0.5
    sdf = th.norm(p, dim=-1) - r
    return sdf


def sdf2d_polygon(points, verts):
    """
    Computes the SDF for a general 2D polygon.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        verts (Tensor): Vertices of the polygon, shape [batch, num_vertices, 2].

    Returns:
        Tensor: The SDF values for the polygon.
    """
    if len(points.shape) == 2:
        #     points = points.unsqueeze(0)
        n_verts = verts.shape[0]
    else:
        n_verts = verts.shape[1]
    d = th.norm(points - verts[..., 0:1, :], dim=-1)
    s = 1.0
    for i in range(n_verts):
        j = (i + 1) % n_verts
        v2 = verts[..., j : j + 1, :]
        v1 = verts[..., i : i + 1, :]
        e = v2 - v1
        e_size = th.norm(e, dim=-1)
        e_cond = e_size > EPSILON
        w = points - verts[..., i : i + 1, :]
        b = w - e * th.clamp(
            (w * e).sum(-1, keepdim=True) / ((e * e).sum(-1, keepdim=True) + EPSILON),
            min=0.0,
            max=1.0,
        )
        d2 = th.minimum(d, (b * b).sum(-1))
        d = th.where(e_cond, d2, d)
        c = th.stack(
            [
                points[..., 1] >= v2[..., 1],
                points[..., 1] < v1[..., 1],
                e[..., 0] * w[..., 1] > e[..., 1] * w[..., 0],
            ],
            -1,
        )
        cond = th.logical_or(th.all(c, -1), th.all(~c, -1))
        s = th.where(cond & e_cond, -s, s)
    return s * th.sqrt(d + EPSILON)


def sdf2d_ellipse(points, ab):
    """
    Computes the SDF for a 2D ellipse.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        ab (Tensor): Dimensions (semi-axes lengths) of the ellipse, shape [batch, 2].

    Returns:
        Tensor: The SDF values for the ellipse.
    """
    sqrt_3 = th.tensor(SQRT_3).to(points.device)
    points = th.abs(points)
    cond = points[..., 0:1] > points[..., 1:2]
    points_flipped = points.flip(-1)
    points = th.where(cond, points_flipped, points)
    ab_flipped = ab.flip(-1)
    ab = th.where(cond, ab_flipped[..., None, :], ab[..., None, :])
    l = ab[..., 1] * ab[..., 1] - ab[..., 0] * ab[..., 0]
    # l = l[..., None]
    m = ab[..., 0] * points[..., 0] / (l + EPSILON)
    n = ab[..., 1] * points[..., 1] / (l + EPSILON)
    m2 = m * m
    n2 = n * n
    c = (m2 + n2 - 1.0) / 3.0
    c3 = c * c * c
    q = c3 + m2 * n2 * 2.0
    d = c3 + m2 * n2
    g = m + m * n2
    # part 1
    c3 = th.where(th.abs(c3) < EPSILON, EPSILON, c3)
    z = q / (c3)
    acos_in = th.clamp(z, min=-1.0 + ACOS_EPSILON, max=1.0 - ACOS_EPSILON)
    h = th.acos(acos_in) / 3.0
    s = th.cos(h)
    t = th.sin(h) * sqrt_3
    rx = th.sqrt(th.clamp(-c * (s + t + 2.0) + m2, min=EPSILON))
    ry = th.sqrt(th.clamp(-c * (s - t + 2.0) + m2, min=EPSILON))
    co_1 = (ry + th.sign(l) * rx + th.abs(g) / (rx * ry) - m) / 2.0
    # part 2
    h = 2.0 * m * n * th.sqrt(th.clamp(d, min=EPSILON))
    s = th.sign(q + h) * th.pow(th.abs(q + h) + EPSILON, 1 / 3.0)
    u = th.sign(q - h) * th.pow(th.abs(q - h) + EPSILON, 1 / 3.0)
    rx = -s - u - c * 4.0 + 2.0 * m2
    ry = (s - u) * sqrt_3
    rm = th.sqrt(rx * rx + ry * ry + EPSILON)
    co_2 = (
        ry / th.sqrt(th.clamp(rm - rx, min=EPSILON)) + 2.0 * g / (rm + EPSILON) - m
    ) / 2.0
    co = th.where(d < 0.0, co_1, co_2)
    r = ab * th.stack([co, th.sqrt(th.clamp(1.0 - co * co, min=EPSILON))], -1)
    sdf = th.norm(r - points, dim=-1) * th.sign(points[..., 1] - r[..., 1])
    return sdf


def sdf2d_parabola(points, k):
    """
    Computes the SDF for a 2D parabola.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        k (Tensor): Parabola coefficient, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the parabola.
    """
    points[..., 0] = th.abs(points[..., 0])
    # k = k[..., None, :]
    ik = 1.0 / (k + EPSILON)
    p = ik * (points[..., 1] - 0.5 * ik) / 3.0
    q = 0.25 * ik * ik * points[..., 0]
    h = q * q - p * p * p
    r = th.sqrt(th.abs(h) + EPSILON)

    sol_1 = th.pow(th.abs(q + r) + EPSILON, 1.0 / 3.0) - th.pow(
        th.abs(q - r) + EPSILON, 1.0 / 3.0
    ) * th.sign(r - q)
    sol_2 = 2.0 * th.cos(th.atan2(r, q) / 3.0) * th.sqrt(th.clamp(p, min=EPSILON))
    x = th.where(h > 0.0, sol_1, sol_2)
    sdf = th.norm(points - th.stack([x, k * x * x], -1), dim=-1) * th.sign(
        points[..., 0] - x
    )
    return sdf


def sdf2d_parabola_segment(points, wi, he):
    """
    Computes the SDF for a segment of a 2D parabola.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        wi (Tensor): Width of the parabola segment, shape [batch, 1].
        he (Tensor): Height of the parabola segment, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the parabola segment.
    """
    points[..., 0] = th.abs(points[..., 0])
    ik = wi * wi / (he + EPSILON)
    p = ik * (he - points[..., 1] - 0.5 * ik) / 3.0
    q = points[..., 0] * ik * ik * 0.25
    h = q * q - p * p * p
    r = th.sqrt(th.abs(h) + EPSILON)
    sol_1 = th.pow(q + r, 1.0 / 3.0) - th.pow(th.abs(q - r), 1.0 / 3.0) * th.sign(r - q)
    sol_2 = 2.0 * th.cos(th.atan2(r, q) / 3.0) * th.sqrt(th.clamp(p, min=EPSILON))
    x = th.where(h > 0.0, sol_1, sol_2)
    x = th.minimum(x, wi)
    sdf = th.norm(points - th.stack([x, he - x * x / ik], -1), dim=-1) * th.sign(
        ik * (points[..., 1] - he) + points[..., 0] * points[..., 0]
    )
    return sdf


def sdf2d_blobby_cross(points, he):
    """
    Computes the SDF for a 2D 'blobby' cross shape.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        he (Tensor): Height parameter influencing the shape, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the blobby cross.
    """
    points = th.abs(points) * 2
    pos = (
        th.stack(
            [
                th.abs(points[..., 0] - points[..., 1]),
                1.0 - points[..., 0] - points[..., 1],
            ],
            -1,
        )
        / SQRT_2
    )
    p = (he - pos[..., 1] - 0.25 / (he + EPSILON)) / (6.0 * he + EPSILON)
    q = pos[..., 0] / (he * he * 16.0 + EPSILON)
    h = q * q - p * p * p
    r_1 = th.sqrt(th.clamp(h, min=EPSILON))
    x_1 = th.pow(th.abs(q + r_1) + EPSILON, 1.0 / 3.0) - th.pow(
        th.abs(q - r_1) + EPSILON, 1.0 / 3.0
    ) * th.sign(r_1 - q)
    r_2 = th.sqrt(th.clamp(p, min=EPSILON))
    acos_in = th.clamp(
        q / (p * r_2 + EPSILON), min=-1 + ACOS_EPSILON, max=1 - ACOS_EPSILON
    )
    x_2 = 2.0 * r_2 * th.cos(th.acos(acos_in) / 3.0)
    x = th.where(h > 0.0, x_1, x_2)
    x = th.clamp(x, max=SQRT_2 / 2.0)
    z = th.stack([x, he * (1.0 - 2.0 * x * x)], -1) - pos
    sdf = th.norm(z, dim=-1) * th.sign(z[..., 1])
    return sdf


def sdf2d_tunnel(points, wh):
    """
    Computes the SDF for a 2D tunnel shape.

    Parameters:
        points (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        wh (Tensor): Width and height of the tunnel, shape [batch, 2].

    Returns:
        Tensor: The SDF values for the tunnel.
    """
    points[..., 0] = th.abs(points[..., 0])
    points[..., 1] = -points[..., 1]
    q = points - wh[..., None, :]
    d1 = th.norm(
        th.stack([th.clamp(q[..., 0].clone(), min=0.0), q[..., 1]], -1), dim=-1
    )
    q[..., 0] = th.where(
        points[..., 1] > 0.0, q[..., 0], th.norm(points, dim=-1) - wh[..., None, 0]
    )
    d2 = th.norm(th.stack([q[..., 0], th.clamp(q[..., 1], min=0.0)], -1), dim=-1)
    d = th.sqrt(th.minimum(d1, d2) + EPSILON)
    sdf = th.where(th.amax(q, -1) < 0.0, -d, d)
    return sdf


def sdf2d_stairs(p, wh, n):
    """
    Computes the SDF for a 2D staircase.

    Parameters:
        p (Tensor): Coordinates for evaluation, shape [batch, num_points, 2].
        wh (Tensor): Width and height of each stair step, shape [batch, 2].
        n (Tensor): Number of steps, shape [batch, 1].

    Returns:
        Tensor: The SDF values for the staircase.
    """
    if len(p.shape) == 2:
        p = p.unsqueeze(0)
    ba = wh * n
    p += ba[..., None, :] / 2.0
    p1 = p.clone()
    p1[..., 0] = p1[..., 0] - th.clamp(
        th.clamp(p[..., 0], min=0.0), max=ba[..., None, 0]
    )
    d1 = (p1 * p1).sum(-1)
    p2 = p.clone()
    p2[..., 1] = p2[..., 1] - th.clamp(
        th.clamp(p[..., 1], min=0.0), max=ba[..., None, 1]
    )
    p2[..., 0] = p2[..., 0] - ba[..., None, 0]
    d2 = (p2 * p2).sum(-1)
    d = th.minimum(d1, d2)
    s = th.sign(th.maximum(-p[..., 1], p[..., 0] - ba[..., None, 0]))
    dia = th.norm(wh, dim=-1, keepdim=True) + EPSILON
    mat = th.stack([wh[..., 0], -wh[..., 1], wh[..., 1], wh[..., 0]], -1).view(-1, 2, 2)
    p = th.bmm(p, mat) / (dia[..., None])
    id = th.clamp(th.clamp(th.round(p[..., 0].clone() / dia), min=0.0), max=n - 1.0)
    p[..., 0] = p[..., 0] - id * dia
    mat = th.stack([wh[..., 0], wh[..., 1], -wh[..., 1], wh[..., 0]], -1).view(-1, 2, 2)
    p = th.bmm(p, mat) / dia[..., None]
    hh = wh[..., 1:2] / 2.0
    p[..., 1] = p[..., 1] - hh
    cond = p[..., 1] > hh * th.sign(p[..., 0])
    s = th.where(cond, 1.0, s)
    p = th.where((id[..., :, None] < 0.5) | (p[..., 0:1] > 0), p, -p)
    p1 = p.clone()
    p1[..., 1] = p1[..., 1] - th.clamp(p[..., 1], min=-hh, max=hh)
    p2 = p.clone()
    p2[..., 0] = p2[..., 0] - th.clamp(th.clamp(p[..., 0], min=0.0), max=wh[..., 0:1])
    p2[..., 1] = p2[..., 1] - hh
    d = th.minimum(d, (p1 * p1).sum(-1))
    d = th.minimum(d, (p2 * p2).sum(-1))
    sdf = th.sqrt(d + EPSILON) * s
    return sdf


def sdf2d_quadratic_circle(points):
    """
    Computes the signed distance function (SDF) of a quadratic circle in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].

    Returns:
        Tensor: The SDF values of the quadratic circle at the input points.
    """
    p = th.abs(points)
    p = th.where(p[..., 1:2] > p[..., 0:1], p[..., [1, 0]], p)
    a = p[..., 0] - p[..., 1]
    b = p[..., 0] + p[..., 1]
    c = (2.0 * b - 1.0) / 3.0
    h = a * a + c * c * c
    cond = h >= 0.0
    # solution 1
    h1 = th.sqrt(th.clamp(h, min=EPSILON))
    t1 = th.sign(h1 - a) * th.pow(th.abs(h1 - a), 1.0 / 3.0) - th.pow(h1 + a, 1.0 / 3.0)
    # solution 2
    z2 = th.sqrt(th.clamp(-c, min=EPSILON))
    acos_in = th.clamp(
        a / (c * z2 + EPSILON), min=-1 + ACOS_EPSILON, max=1 - ACOS_EPSILON
    )
    v2 = th.acos(acos_in) / 3.0
    t2 = -z2 * (th.cos(v2) + th.sin(v2) * SQRT_3)
    t = th.where(cond, t1, t2)
    t = t * 0.5
    w = th.stack([-t, t], -1) + 0.75 - (t * t)[..., None] - p
    sdf = th.norm(w, dim=-1) * th.sign(a * a * 0.5 + b - 1.5)
    return sdf


def sdf2d_cool_s(points):
    """
    Computes the SDF for a 'cool S' shape in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].

    Returns:
        Tensor: The SDF values of the 'cool S' shape at the input points.
    """
    six = th.where(points[..., 1] < 0.0, -points[..., 0], points[..., 0])
    points[..., 0] = th.abs(points[..., 0])
    points[..., 1] = th.abs(points[..., 1]) - 0.2
    rex = points[..., 0] - th.clamp(th.round(points[..., 0] / 0.4), max=0.4)
    aby = th.abs(points[..., 1] - 0.2) - 0.6
    vec2 = th.stack([six, -points[..., 1]], -1)
    vec2 = vec2 - th.clamp(0.5 * (six - points[..., 1]), min=0.0, max=0.2)[..., None]
    d = (vec2 * vec2).sum(-1)
    vec2 = th.stack([points[..., 0], -aby], -1)
    vec2 = vec2 - th.clamp(0.5 * (points[..., 0] - aby), min=0.0, max=0.4)[..., None]
    d = th.minimum(d, (vec2 * vec2).sum(-1))
    vec2 = th.stack(
        [rex, points[..., 1] - th.clamp(points[..., 1], min=0.0, max=0.4)], -1
    )
    d = th.minimum(d, (vec2 * vec2).sum(-1))
    s = 2.0 * points[..., 0] + aby + th.abs(aby + 0.4) - 0.4
    sdf = th.sqrt(d + EPSILON) * th.sign(s)
    return sdf


def sdf2d_circle_wave(points, tb, ra):
    """
    Computes the SDF for a circle wave pattern in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        tb (Tensor): Parameter tensor for the wave, shape [batch, 1].
        ra (Tensor): Radius of the circles, shape [batch, 1].

    Returns:
        Tensor: The SDF values of the circle wave pattern at the input points.
    """
    points[..., 0] += 1.0
    tb = th.clamp(tb, min=EPSILON)
    tb = np.pi * 5 / 6.0 * tb
    co = ra * th.cat([th.sin(tb), th.cos(tb)], -1)
    points[..., 0] = th.abs(
        th.fmod(points[..., 0].clone(), co[..., 0:1] * 4.0) - co[..., 0:1] * 2.0
    )
    p1 = points.clone()
    p2 = th.stack(
        [
            th.abs(points[..., 0] - 2.0 * co[..., 0:1]),
            -points[..., 1] + 2.0 * co[..., 1:2],
        ],
        -1,
    )
    cond = co[..., 1:2] * p1[..., 0] > co[..., 0:1] * p1[..., 1]
    d1 = th.where(
        cond, th.norm(p1 - co[..., None, :], dim=-1), th.abs(th.norm(p1, dim=-1) - ra)
    )
    cond = co[..., 1:2] * p2[..., 0] > co[..., 0:1] * p2[..., 1]
    d2 = th.where(
        cond, th.norm(p2 - co[..., None, :], dim=-1), th.abs(th.norm(p2, dim=-1) - ra)
    )
    sdf = th.minimum(d1, d2)
    return sdf


def sdf2d_hyperbola(points, k, he):
    """
    Computes the SDF for a hyperbola in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        k (Tensor): Scale factor for the hyperbola, shape [batch, 1].
        he (Tensor): Height parameter for the hyperbola, shape [batch, 1].

    Returns:
        Tensor: The SDF values of the hyperbola at the input points.
    """
    points = th.abs(points)
    points = (
        th.stack([points[..., 0] - points[..., 1], points[..., 0] + points[..., 1]], -1)
        / SQRT_2
    )
    x2 = points[..., 0] * points[..., 0] / 16.0
    y2 = points[..., 1] * points[..., 1] / 16.0
    r = k * (4.0 * k - points[..., 0] * points[..., 1]) / 12.0
    q = (x2 - y2) * k * k
    h = q * q + r * r * r
    cond = h < 0.0
    # solution1
    m1 = th.sqrt(th.clamp(-r, min=EPSILON))
    acos_in = th.clamp(
        q / (r * m1 + EPSILON), min=-1.0 + ACOS_EPSILON, max=1.0 - ACOS_EPSILON
    )
    u1 = m1 * th.cos(th.acos(acos_in) / 3.0)
    # solution2
    m2 = th.pow(th.sqrt(th.clamp(h, min=EPSILON)) - q, 1.0 / 3.0)
    u2 = (m2 - r / (m2 + EPSILON)) / 2.0
    u = th.where(cond, u1, u2)
    w = th.sqrt(th.clamp(u + x2, min=EPSILON))
    b = k * points[..., 1] - x2 * points[..., 0] * 2.0
    t = (
        points[..., 0] / 4.0
        - w
        + th.sqrt(th.clamp(2.0 * x2 - u + b / (w * 4.0 + EPSILON), min=EPSILON))
    )
    t = th.maximum(t, th.sqrt(th.clamp(he * he * 0.5 + k, min=EPSILON)) - he / SQRT_2)

    d = th.norm(points - th.stack([t, k / t], -1), dim=-1)
    sdf = th.where(points[..., 0] * points[..., 1] < k, d, -d)
    return sdf


def sdf2d_quadratic_bezier_curve(points, A, B, C):
    """
    Computes the SDF for a quadratic Bezier curve in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        A (Tensor): Start point of the Bezier curve, shape [batch, 2].
        B (Tensor): Control point of the Bezier curve, shape [batch, 2].
        C (Tensor): End point of the Bezier curve, shape [batch, 2].

    Returns:
        Tensor: The SDF values of the quadratic Bezier curve at the input points.
    """
    a = B - A
    b = A - 2.0 * B + C
    c = a * 2.0
    d = A[..., None, :] - points
    kk = 1.0 / ((b * b).sum(-1, keepdim=True) + EPSILON)
    kx = kk * (a * b).sum(-1, keepdim=True)
    ky = (
        kk * (2.0 * (a * a).sum(-1, keepdim=True) + (d * b[..., None, :]).sum(-1)) / 3.0
    )
    kz = kk * (d * a[..., None, :]).sum(-1)
    p = ky - kx * kx
    p3 = p * p * p
    q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    h = q * q + 4.0 * p3
    cond = h >= 0.0
    c = c[..., None, :]
    b = b[..., None, :]
    # sol 1
    h1 = th.sqrt(th.clamp(h, min=EPSILON))
    x1 = (th.stack([h1, -h1], -1) - q[..., None]) / 2.0
    uv1 = th.sign(x1) * th.pow(th.abs(x1) + EPSILON, 1.0 / 3.0)
    t1 = th.clamp(uv1[..., 0] + uv1[..., 1] - kx, min=0.0, max=1.0)
    t1 = t1[..., None]
    res_1 = th.norm(d + (c + b * t1) * t1, dim=-1)
    # sol 2
    z2 = th.sqrt(th.clamp(-p, min=EPSILON))
    acos_in = th.clamp(
        q / (p * z2 * 2.0 + EPSILON), min=-1.0 + ACOS_EPSILON, max=1.0 - ACOS_EPSILON
    )
    v2 = th.acos(acos_in) / 3.0
    m2 = th.cos(v2)
    n2 = th.sin(v2) * SQRT_3
    t2 = th.clamp(
        th.stack([m2 + m2, -n2 - m2, n2 - m2], -1) * z2[..., None] - kx[..., None],
        min=0.0,
        max=1.0,
    )
    res_2 = th.minimum(
        th.norm(d + (c + b * t2[..., 0:1]) * t2[..., 0:1], dim=-1),
        th.norm(d + (c + b * t2[..., 1:2]) * t2[..., 1:2], dim=-1),
    )
    res = th.where(cond, res_1, res_2)
    return th.sqrt(res + EPSILON)


def sdf2d_segment(points, start_point, end_point):
    """
    Computes the SDF for a line segment in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        start_point (Tensor): Start point of the line segment, shape [batch, 2].
        end_point (Tensor): End point of the line segment, shape [batch, 2].

    Returns:
        Tensor: The SDF values of the line segment at the input points.
    """
    pa = points - start_point[..., None, :]
    ba = (end_point - start_point)[..., None, :]
    h = th.clamp(
        (pa * ba).sum(-1, keepdim=True) / ((ba * ba).sum(-1, keepdim=True) + EPSILON),
        0.0,
        1.0,
    )
    sdf = th.norm(pa - ba * h, dim=-1)
    return sdf


# Simple parameter less primitives.
def sdf2d_no_param_rectangle(points):
    """
    Computes the SDF for a unit rectangle centered at the origin in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].

    Returns:
        Tensor: The SDF values of the unit rectangle at the input points.
    """
    points = th.abs(points)
    points = points - 0.5
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + th.clip(
        th.amax(points, -1), max=0
    )
    return base_sdf


def sdf2d_no_param_circle(points):
    """
    Computes the SDF for a unit circle centered at the origin in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].

    Returns:
        Tensor: The SDF values of the unit circle at the input points.
    """
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - 0.5
    return base_sdf


def sdf2d_no_param_triangle(points):
    """
    Computes the SDF for an equilateral triangle in 2D.

    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].

    Returns:
        Tensor: The SDF values of the equilateral triangle at the input points.
    """
    k = th.tensor(SQRT_3).to(points.device)
    r = 0.5
    points[..., 0] = th.abs(points[..., 0]) - r
    points[..., 1] = points[..., 1] + r / k
    condition = points[..., 0] + (k * points[..., 1]) > 0
    new_points = (
        th.stack(
            [points[..., 0] - k * points[..., 1], -k * points[..., 0] - points[..., 1]],
            -1,
        )
        / 2.0
    )
    points = th.where(condition.unsqueeze(-1), new_points, points)
    points[..., 0] = points[..., 0] - th.clamp(points[..., 0], -2 * r, 0)
    sdf = -th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf

def sdf2d_tile_uv(points, tile, height=None, width=None):
    """
    Splat the Tile over the points. 
    
    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        tile (Tensor): The tile to splat, shape [Batch, height, width, n_channel].
    """
    # using torch.nn.functional.grid_sample
    # rearrange points in B,H,W,2 format
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if len(tile.shape) == 3:
        tile = tile.unsqueeze(0)
    ps = points.shape
    bs = ps[0]
    if height is None:
        height = np.sqrt(ps[1]).astype(int)
        width = height
    # Doesn't handle the case where one is given    
    cur_points = points.reshape(bs, height, width, 2)
    cur_tile = tile.permute(0, 3, 1, 2)
    output = th.nn.functional.grid_sample(cur_tile, cur_points, align_corners=True)
    
    return output

def sdf2d_sin_x(points, freq, phase_shift):
    """
    Computes the SDF for a sin wave in 2D.
    
    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        freq (Tensor): Frequency of the sin wave, shape [batch, 1].
        phase_shift (Tensor): Phase shift of the sin wave, shape [batch, 1].
    """
    base_sdf = th.sin(2 * np.pi * freq * points[..., 0] + phase_shift)
    return base_sdf

def sdf2d_sin_y(points, freq, phase_shift):
    """
    Computes the SDF for a sin wave in 2D.
    
    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        freq (Tensor): Frequency of the sin wave, shape [batch, 1].
        phase_shift (Tensor): Phase shift of the sin wave, shape [batch, 1].
    """
    base_sdf = th.sin(2 * np.pi * freq * points[..., 1] + phase_shift)
    return base_sdf

def sdf2d_sinwave_y(points, freq, phase_shift, scale):
    """
    Computes the SDF for a sin wave in 2D.
    
    Parameters:
        points (Tensor): Input points to evaluate the SDF, shape [batch, num_points, 2].
        freq (Tensor): Frequency of the sin wave, shape [batch, 1].
        phase_shift (Tensor): Phase shift of the sin wave, shape [batch, 1].
    """
    
    target = (1 +  th.sin(2 * np.pi * freq * points[..., 1] + phase_shift)) * scale
    base_sdf = th.abs(points[..., 0]) - target 
    return base_sdf