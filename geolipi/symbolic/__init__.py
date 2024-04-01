from .primitives_3d import *
from .primitives_2d import *
from .combinators import *
from .transforms_2d import *
from .transforms_3d import *
from .primitives_higher import *
from .color import *

import inspect


def get_cmd_mapper(include_torch=True):
    str_to_cmd_mapper = dict()
    import geolipi.symbolic as gls

    str_to_cmd_mapper.update(
        {x[0]: x[1] for x in inspect.getmembers(gls, inspect.isclass)}
    )
    if include_torch:
        import torch
        str_to_cmd_mapper['torch'] = torch
        str_to_cmd_mapper['tensor'] = torch.tensor
    return str_to_cmd_mapper
