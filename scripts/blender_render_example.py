"""An example script for loading a GeoLIPI program in blender and rendering it. Note that only a few primitive types are supported currently."""
import os
import sys
import math
import random
from pathlib import Path
import _pickle as cPickle

import bpy
import mathutils

file_path = os.path.dirname(os.path.abspath(__file__))
GEOLIPI_PATH = os.path.join(file_path, "..")
print(GEOLIPI_PATH)
sys.path.insert(0, GEOLIPI_PATH)

# In case the script is rerun without restarting blender, we need to delete the geolipi modules from sys.modules
delete_key = []
for key, value in sys.modules.items():
    if "geolipi" in key:
        delete_key.append(key)

for key in delete_key:
    del sys.modules[key]
    
import geolipi.symbolic as gls
from geolipi.geometry_nodes.evaluate_graph import expr_to_geonode_graph
from geolipi.geometry_nodes.utils import clean_up, init_camera, init_lights, set_render_settings, BASE_COLORS



# Load a bunch of programs inferred by SIRI.
# Programs are saved in string format, and we use python's eval to convert them to a gls expressions
# NOTE - it can be dangerous to use eval on strings you don't trust.
str_to_cmd_mapper = gls.get_cmd_mapper()
# Load a program
selected_ind = 13
file_name = os.path.join(GEOLIPI_PATH, "assets", "example_inferred_programs.pkl")
with open(file_name, 'rb') as f:
    expressions = cPickle.load(f)
str_to_cmd_mapper = gls.get_cmd_mapper()
expressions = [eval(p, str_to_cmd_mapper) for p in expressions]
expr = expressions[selected_ind]

# Blender setup
clean_up()
init_camera()
init_lights()
set_render_settings(resolution=512, transparent_mode=True)

origin_point = mathutils.Vector((0, 0, 0))
bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, -0.0), scale=(1, 1, 1))
plane_obj = bpy.context.active_object

# set camera
theta = 45 
theta = theta * math.pi/180
size = math.sqrt(8)
camera_x = size * math.cos(theta)
camera_y = size * math.sin(theta)
camera = bpy.data.objects['Camera']
camera.location = (camera_x, camera_y, 1.5)
direction = origin_point - camera.location
rotQuat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rotQuat.to_euler()

# Generate Geometry Node Graph:
colors = random.shuffle(BASE_COLORS)
expr_to_geonode_graph(expr, plane_obj, "cuda", colors=colors)

#  Render
save_loc = os.path.join(GEOLIPI_PATH, "assets")
Path(save_loc).mkdir(parents=True, exist_ok=True)
save_template = f"{save_loc}/{selected_ind}"
save_file_name = f"{save_template}.png"
bpy.context.scene.render.filepath = save_file_name
bpy.ops.render.render(write_still = True)
