from typing import List
import re

from geolipi.symbolic import Primitive, Combinator
from geolipi.symbolic.primitives import Cuboid, Sphere
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.transforms import Translate, Scale

# Define the regular expression pattern (ChatGPT Help!)
pattern = r'(\w+|\$)(?:\(([-0-9.,\s]*)\))?|\$'

boolean_map = {
    'union': Union,
    'intersection': Intersection,
    'difference': Difference
}
primitive_map = {
    'cuboid': Cuboid,
    'sphere': Sphere
}
boolean_commands = boolean_map.keys()
primitive_commands = primitive_map.keys()


def parser(expression_string_list: List[str]):
    """Parse a list of expression strings into a list of expressions.
    """
    canvas_stack = []
    operator_stack = []

    for expression_string in expression_string_list:
        match = re.match(pattern, expression_string)
        cmd_name = match.group(1)
        if cmd_name in boolean_commands:
            operator_stack.append(boolean_map[cmd_name])
        elif cmd_name in primitive_commands:
            params = [float(x.strip()) for x in match.group(2).split(',')]
            cmd = Translate(Scale(primitive_map[cmd_name](), tuple(params[3:])),
                            tuple(params[:3]))
            canvas_stack.append(cmd)
        else:
            raise ValueError(f'Unknown command {expression_string}')

    while (operator_stack):
        cur_operator = operator_stack.pop()
        canvas_2 = canvas_stack.pop()
        canvas_1 = canvas_stack.pop()
        new_canvas = cur_operator(canvas_1, canvas_2)
        canvas_stack.append(new_canvas)

    assert len(canvas_stack) == 1, 'Error! Stack should have only one element'

    return canvas_stack[0]
