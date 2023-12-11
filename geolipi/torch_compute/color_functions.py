import torch as th

# Color fetch
from .common import EPSILON


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
    alpha_s = premult_s[..., 3:4]
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
    alpha_s = premult_s[..., 3:4]
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
    alpha_d = premult_d[..., 3:4]
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
    alpha_s = premult_s[..., 3:4]
    alpha_d = premult_d[..., 3:4]
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
    alpha_s = premult_s[..., 3:4]
    alpha_d = premult_d[..., 3:4]
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
    alpha_r = result[..., 3:4]
    alpha_r = th.clamp(alpha_r, EPSILON, 1)
    color_r = result[..., :3] / (alpha_r + EPSILON)
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
    alpha_s = source[..., 3:4]
    color_s = source[..., :3]
    alpha_d = destination[..., 3:4]
    color_d = destination[..., :3]
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
    alpha_a = occ_expand * color[..., 3:4]
    color_a = occ_expand * color[..., :3]
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
    old_alpha = color_canvas[..., 3:4]
    new_alpha = new_color[..., 3:4]
    result_alpha = new_alpha + old_alpha * (1 - new_alpha)
    color_canvas[..., :3] = new_color[..., :3] * new_alpha + color_canvas[..., :3] * (
        1 - new_alpha
    )
    color_canvas[..., :3] = color_canvas[..., :3] / result_alpha
    return color_canvas
