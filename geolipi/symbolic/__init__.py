from .primitives_3d import *
from .primitives_2d import *
from .combinators import *
from .transforms_2d import *
from .transforms_3d import *
from .primitives_higher import *
from .color import *

import inspect


def get_cmd_mapper():
    str_to_cmd_mapper = dict()
    import geolipi.symbolic as gls

    str_to_cmd_mapper.update(
        {x[0]: x[1] for x in inspect.getmembers(gls, inspect.isclass)}
    )
    return str_to_cmd_mapper
