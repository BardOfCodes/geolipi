import bpy
from mathutils import *
import mathutils
import math
import numpy as np
import _pickle as cPickle
import os
import sys

sys.path.insert(0, "/home/aditya/projects/rl/blender_app")
from iccv_render.utils import clean_up, render_settings
from iccv_render.csg_visualizer import CSGVisualizer
from iccv_render.bt_utils import setLight_ambient, setLight_sun

D = bpy.data
C = bpy.context

clean_up()

bpy.ops.object.camera_add(enter_editmode=False, align='VIEW',
                          location=(2.0, 2.0, 1.5), 
                          rotation=(1.015, 0.0105478, 2.356), 
                          scale=(1, 1, 1))
point = mathutils.Vector((0, 0, 0))

camera = bpy.data.objects['Camera']
direction = point - camera.location
rotQuat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rotQuat.to_euler()


bpy.context.scene.camera = camera

# Light Settings
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
light = bpy.data.objects['Sun']
light.data.angle = 25/180. * math.pi
light.data.energy = 3.5

# Shadow catcher
bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(5, 5, -5.0), scale=(1, 1, 1))
plane_obj = bpy.data.objects["Plane"]
#plane_obj.scale[0] = 100
#plane_obj.scale[1] = 100
#plane_obj.scale[2] = 100
plane_obj.is_shadow_catcher = False

# Render Settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.render.film_transparent = True
bpy.context.scene.cycles.samples = 32
bpy.context.scene.render.use_freestyle = True
bpy.context.scene.render.line_thickness = 0.25
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
freestyle_settings = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
lineset = freestyle_settings.linesets.active
lineset.select_silhouette = True
lineset.select_crease = True
lineset.select_border = True
lineset.select_material_boundary = True
lineset.select_external_contour = True

#lightAngle = (66, -30, -155) 
#strength = 2
#shadowSoftness = 0.3
#sun = setLight_sun(lightAngle, strength, shadowSoftness)
setLight_ambient(color=(0.1,0.1,0.1,1)) 

# COLOR PALETTE:
# https://stackoverflow.com/questions/56338701/generate-rgb-color-set-with-highest-diversity
# colors = np.load("/home/aditya/blender/diverse_colors_256.npy")
colors = np.load("/home/aditya/blender/diverse_colors_32.npy")
seed = 1
np.random.seed(seed)
np.random.shuffle(colors)

# for each expression, parse and create the node graph
#expression_file = "/home/aditya/projects/rl/weights/iccv/csgstump/fcsg3d_srt_2/beam_do_gs_cs_3.pkl"
expression_file = "/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_plad/beam_10.pkl"

save_dir = "/home/aditya/blender/outputs/iccv/pcsg3d_plad_beam_10"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

program_list = cPickle.load(open(expression_file, "rb"))
#program_list = [x for x in program_list if x['target_id'] == 0]

# Load the expressions
visualizer = CSGVisualizer(plane_obj, colors)

for ind, program in enumerate(program_list):
#   
    print(ind)
    slot_id = program['slot_id']
    target_id = program['target_id']
    save_file_name = os.path.join(save_dir, "%s_%d.png" % (slot_id, target_id))
    if not os.path.exists(save_file_name):
        # clean_up(avoid_keys=["Camera", "Sun", "Plane"])
        
        expression = ['rotate(-90, 0, 90)', "intersection", "scale(2, 2, 2)", "cuboid"] + program['render_expr']
    #    
        
        cmd = visualizer.parse(expression)

        bpy.ops.mesh.primitive_plane_add()
        dummy_obj = bpy.context.active_object
        bpy.ops.object.shade_smooth()

    #    dummy_obj.shade_smooth()
        dummy_obj.name = "DUMMY"
        visualizer.compile(cmd, dummy_obj)
    #    
        # Render:
        bpy.context.scene.render.filepath = save_file_name
        bpy.ops.render.render(write_still = True)
        
        bpy.data.objects.remove(dummy_obj, do_unlink = True)
        group = bpy.data.node_groups["CSG-Tree"]
        bpy.data.node_groups.remove(group, do_unlink=True)