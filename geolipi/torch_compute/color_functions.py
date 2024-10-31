import torch as th

# Color fetch
from .common import EPSILON
from .settings import Settings
import skfmm
import numpy as np

# Color boolean.
def destination_in(source, destination):
    """
    Computes the destination-in composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the destination-in composite operation.
    """
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., -1:]
    result = premult_d * alpha_s
    output = get_unmultiplied_form(result)
    return output


def destination_out(source, destination):
    """
    Computes the destination-out composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the destination-out composite operation.
    """
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., -1:]
    result = premult_d * (1 - alpha_s)
    output = get_unmultiplied_form(result)
    return output


def destination_over(source, destination):
    """
    Computes the destination-over composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the destination-over composite operation.
    """
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_d = premult_d[..., -1:]
    result = premult_s * (1 - alpha_d) + premult_d
    output = get_unmultiplied_form(result)
    return output


def destination_atop(source, destination):
    """
    Computes the destination-atop composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the destination-atop composite operation.
    """
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., -1:]
    alpha_d = premult_d[..., -1:]
    result = premult_s * (1 - alpha_d) + premult_d * alpha_s
    output = get_unmultiplied_form(result)
    return output


def svg_xor(source, destination):
    """
    Computes the XOR composite (as defined in SVG) of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the SVG XOR composite operation.
    """
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., -1:]
    alpha_d = premult_d[..., -1:]
    result = premult_s * (1 - alpha_d) + premult_d * (1 - alpha_s)
    output = get_unmultiplied_form(result)
    return output


def get_unmultiplied_form(result):
    """
    Converts a premultiplied color tensor to its unmultiplied form.

    Parameters:
        result (Tensor): Premultiplied color tensor.

    Returns:
        Tensor: Unmultiplied color tensor.
    """
    alpha_r = result[..., -1:]
    alpha_r = th.clamp(alpha_r, EPSILON, 1)
    color_r = result[..., :-1] / (alpha_r + EPSILON)
    if Settings.COLOR_CLAMP:
        color_r = th.clamp(color_r, 0, 1)
    output = th.cat([color_r, alpha_r], dim=-1)
    return output


def get_premultiplied_form(source, destination):
    """
    Converts source and destination color tensors to their premultiplied form.

    Parameters:
        source, destination (Tensor): Source and destination color tensors.

    Returns:
        Tuple[Tensor, Tensor]: Tensors in premultiplied form.
    """
    alpha_s = source[..., -1:]
    color_s = source[..., :-1]
    alpha_d = destination[..., -1:]
    color_d = destination[..., :-1]
    premult_s = color_s * alpha_s
    premult_s = th.cat([premult_s, alpha_s], dim=-1)
    premult_d = color_d * alpha_d
    premult_d = th.cat([premult_d, alpha_d], dim=-1)
    return premult_s, premult_d


def source_over_seq(*args):
    """
    Sequentially applies the source-over composite operation on a sequence of images.
    employed with executing macros in SVG expressions.

    Parameters:
        *args (Tensors): A sequence of image tensors.

    Returns:
        Tensor: The result of sequentially applying source-over compositing.
    """
    result = args[0]
    for i in range(1, len(args)):
        result = source_over(args[i], result)
    return result


def source_in(source, destination):
    """
    Computes the source-in composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the source-in composite operation.
    """
    return destination_in(destination, source)


def source_out(source, destination):
    """
    Computes the source-out composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the source-out composite operation.
    """
    return destination_out(destination, source)


def source_over(source, destination):
    """
    Computes the source-over composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the source-over composite operation.
    """
    return destination_over(destination, source)


def source_atop(source, destination):
    """
    Computes the source-atop composite of source and destination images.

    Parameters:
        source, destination (Tensor): Source and destination image tensors.

    Returns:
        Tensor: The result of applying the source-atop composite operation.
    """
    return destination_atop(destination, source)


def apply_color(occupancy, color):
    """
    Applies a color to an occupancy grid.

    Parameters:
        occupancy (Tensor): An occupancy grid tensor.
        color (Tensor): A color tensor.

    Returns:
        Tensor: A colored canvas tensor.
    """
    occ_expand = occupancy[..., None].float()
    alpha_a = occ_expand * color[..., -1:]
    color_a = occ_expand * color[..., :-1]
    # color_b = (1 - occ_expand) * 1.0
    # color_a = color_a + color_b
    canvas = th.cat([color_a, alpha_a], dim=-1)
    return canvas


def modify_opacity(color_canvas, opacity):
    """
    Modifies the opacity of a color canvas.

    Parameters:
        color_canvas (Tensor): A color canvas tensor.
        opacity (float): Desired opacity level.

    Returns:
        Tensor: A modified color canvas tensor.
    """
    color_canvas[..., 3] = color_canvas[..., 3] * opacity
    return color_canvas


def modify_color(color_canvas, new_color):
    """
    Modifies the color of a color canvas.

    Parameters:
        color_canvas (Tensor): A color canvas tensor.
        new_color (Tensor): New color tensor to apply.

    Returns:
        Tensor: A modified color canvas tensor.
    """
    old_alpha = color_canvas[..., -1:]
    new_alpha = new_color[..., -1:]
    result_alpha = new_alpha + old_alpha * (1 - new_alpha)
    color_canvas[..., :-1] = new_color[..., :-1] * new_alpha + color_canvas[..., :-1] * (
        1 - new_alpha
    )
    color_canvas[..., :-1] = color_canvas[..., :-1] / result_alpha
    return color_canvas


def depreciated_modify_color_tritone(points, mid_color, black=None, white=None, mid_point = 0.5):
    mid_color = mid_color[..., :-1]
    n = mid_color.shape[-1]
    if black is None:
        black = th.zeros_like(points[..., :n])
    else:
        black = black[..., :n]
    if white is None:
        white = th.ones_like(points[..., :n])
    else:
        white = white[..., :n]

    R, G, B, A = th.chunk(points, 4, dim=-1)
    # L = (R + G+ B) / 3# 
    L = R * 299/1000 + G * 587/1000 + B * 114/1000
    L = th.clamp(L, 0, 1)
    # Now using L we must give each a ratio factor
    factor_a = (L - mid_point) / (1 - mid_point)
    factor_a = th.clamp(factor_a, 0, 1)
    color_a = mid_color + (white - mid_color) * (factor_a)
    factor_b = (L / mid_point)
    factor_b = th.clamp(factor_b, 0, 1)
    color_b = black + (mid_color - black) * (factor_b)
    color = th.where(L >= mid_point, color_a, color_b)
    if n == 3:
        color = th.cat([color, A], dim=-1)

    return color

def alpha_mask(points):
    mask = 0.1 - points[..., -1:]
    return mask

def unopt_alpha_to_sdf(points, dx, canvas_shape=None):
    shape = points.shape
    if not len(shape) == 3:
        # its P
        # convert to 1, P, C
        points = points.unsqueeze(0)
    if isinstance(dx, th.Tensor):
        dx = dx.item()
    B, P, C = points.shape
    if canvas_shape is None:
        n = np.sqrt(P).astype(int)
        canvas_shape = (n, n)
    outputs = []
    
    for i in range(B):
        cur_inp = points[i] # P C
        cur_inp = cur_inp.reshape(canvas_shape[0], canvas_shape[1], -1)
        cur_inp = cur_inp.cpu().numpy()
        distances = skfmm.distance(cur_inp[..., 0], dx=dx)
        distances = th.from_numpy(distances.reshape(-1, 1)).to(points.device).to(points.dtype)
        outputs.append(distances)
    outputs = th.stack(outputs, dim=0)
    if B == 1:
        outputs = outputs.squeeze(0).squeeze(-1)
    return outputs

# To HSL

# Rotate HSL
# TO RGB
