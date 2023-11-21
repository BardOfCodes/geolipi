import torch as th
# Color fetch
from .common import EPSILON

# Use premultiplied? -> optimizing alpha seperately from color.

# Color boolean.
def destination_in(source, destination):
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., 3:4]
    result = premult_d * alpha_s
    output = get_unmultiplied_form(result)
    return output

def destination_out(source, destination):
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., 3:4]
    result = premult_d * (1 - alpha_s)
    output = get_unmultiplied_form(result)
    return output

def destination_over(source, destination):
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_d = premult_d[..., 3:4]
    result = premult_s * (1 - alpha_d) + premult_d
    output = get_unmultiplied_form(result)
    return output

def destination_atop(source, destination):
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., 3:4]
    alpha_d = premult_d[..., 3:4]
    result = premult_s * (1 - alpha_d) + premult_d * alpha_s
    output = get_unmultiplied_form(result)
    return output


def svg_xor(source, destination):
    premult_s, premult_d = get_premultiplied_form(source, destination)
    alpha_s = premult_s[..., 3:4]
    alpha_d = premult_d[..., 3:4]
    result = premult_s * (1 - alpha_d) + premult_d * (1 - alpha_s)
    output = get_unmultiplied_form(result)
    return output
    
def get_unmultiplied_form(result):
    alpha_r = result[..., 3:4]
    alpha_r = th.clamp(alpha_r, EPSILON, 1)
    color_r = result[..., :3] / (alpha_r + EPSILON)
    color_r = th.clamp(color_r, 0, 1)
    output = th.cat([color_r, alpha_r], dim=-1)
    return output

def get_premultiplied_form(source, destination):
    alpha_s = source[..., 3:4]
    color_s = source[..., :3]
    alpha_d = destination[..., 3:4]
    color_d = destination[..., :3]
    premult_s = color_s * alpha_s
    premult_s = th.cat([premult_s, alpha_s], dim=-1)
    premult_d = color_d * alpha_d
    premult_d = th.cat([premult_d, alpha_d], dim=-1)
    return premult_s, premult_d

def source_over_seq(sequence):
    result = sequence[0]
    for i in range(1, len(sequence)):
        result = source_over(sequence[i], result)
    return result

def source_in(source, destination):
    return destination_in(destination, source)

def source_out(source, destination):
    return destination_out(destination, source)

def source_over(source, destination):
    return destination_over(destination, source)

def source_atop(source, destination):
    return destination_atop(destination, source)

def apply_color(occupancy, color):
    occ_expand = occupancy[..., None].float()
    alpha_a = occ_expand * color[0, 3:4]
    color_a = occ_expand * color[..., :3]
    # color_b = (1 - occ_expand) * 1.0
    # color_a = color_a + color_b
    canvas = th.cat([color_a, alpha_a], dim=-1)
    return canvas

def modify_opacity(color_canvas, opacity):
    color_canvas[..., 3] = color_canvas[..., 3] * opacity
    return color_canvas

def modify_color(color_canvas, new_color):
    old_alpha = color_canvas[..., 3:4]
    new_alpha = new_color[..., 3:4]
    result_alpha = new_alpha + old_alpha * (1 - new_alpha)
    color_canvas[..., :3] = new_color[..., :3] * new_alpha + color_canvas[..., :3] * (1 - new_alpha)
    color_canvas[..., :3] = color_canvas[..., :3] / result_alpha
    return color_canvas