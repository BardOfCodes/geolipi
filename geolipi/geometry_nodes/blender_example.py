import sys
sys.path.insert(0, "/home/aditya/projects/iccv_23/repos/geolipi")

delete_key = []
for key, value in sys.modules.items():
    if "geolipi" in key:
        delete_key.append(key)

for key in delete_key:
    del sys.modules[key]
    
import geolipi
from geolipi.symbolic import *
import bpy
import mathutils
from geolipi.geometry_nodes.evaluate_graph import expr_to_geonode_graph
from geolipi.geometry_nodes.utils import clean_up, init_camera, init_lights, set_render_settings, get_obj_min_z
from geolipi.languages.macro_csg3d import str_to_expr
import _pickle as cPickle
import random
import math

clean_up()
init_camera()
init_lights()
set_render_settings(-1)

# check if the render of saved processed objs are good
#expression_file = "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_srt_2/beam_do_gs_cs_3_no_tax.pkl"
expression_file = "/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_srt_2/beam_do_gs_cs_3_no_tax.pkl"
program_list = cPickle.load(open(expression_file, "rb"))

sample_index = 2 # random.randint(0, len(program_list)/4)
program = program_list[sample_index]
expression = ['rotate(1.5707, 0, -1.57)', "intersection", "scale(2, 2, 2)", "cuboid"] + program['render_expr']
print("string expression")
print(program['render_expr'])
expr = str_to_expr(expression, to_cuda=True)
print("geolipi expression")
print(expr)

bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, -0.0), scale=(1, 1, 1))
plane_obj = bpy.context.active_object


expr_to_geonode_graph(expr, plane_obj, "cuda")





origin_point = mathutils.Vector((0, 0, 0))
save_loc = "/home/aditya/blender/outputs/new"
n = 64
for i in range(n):
    theta = 45 + 360/n * i
    theta = theta * math.pi/180
    size = math.sqrt(8)
    camera_x = size * math.cos(theta)
    camera_y = size * math.sin(theta)
    camera = bpy.data.objects['Camera']
    camera.location = (camera_x, camera_y, 1.5)
    direction = origin_point - camera.location
    rotQuat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rotQuat.to_euler()
    save_file_name = f"{save_loc}/{i}.png"
    bpy.context.scene.render.filepath = save_file_name
    bpy.ops.render.render(write_still = True)
    
     
    
        