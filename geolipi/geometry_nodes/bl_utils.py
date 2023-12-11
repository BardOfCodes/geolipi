import sys
import math
import numpy as np


# Unsafe?
def blender_importer(module_name):
    if module_name not in sys.modules:
        eval(f"import {module_name}")
    return sys.modules[module_name]


def import_bpy():
    module_name = "bpy"
    module = blender_importer(module_name)
    return module



def clean_up(avoid_keys=[]):
    """
    Cleans up the Blender scene by removing various data types except those specified in the avoid list.

    Args:
        avoid_keys (list, optional): A list of keys (names) to avoid deleting. Defaults to an empty list.
    """
    print("cleaning up")
    bpy = import_bpy()
    for key in bpy.data.node_groups.keys():
        if not key in avoid_keys:
            group = bpy.data.node_groups[key]
            bpy.data.node_groups.remove(group, do_unlink=True)
    for obj in bpy.data.objects:
        if not obj.name in avoid_keys:
            bpy.data.objects.remove(obj, do_unlink=True)
    for mat in bpy.data.materials:
        if not mat.name in avoid_keys:
            bpy.data.materials.remove(mat, do_unlink=True)
    for obj in bpy.data.meshes:
        if not obj.name in avoid_keys:
            bpy.data.meshes.remove(obj, do_unlink=True)
    for obj in bpy.data.cameras:
        if not obj.name in avoid_keys:
            bpy.data.cameras.remove(obj, do_unlink=True)
    if bpy.context.scene.use_nodes:
        tree = bpy.context.scene.node_tree
        for name, cur_node in tree.nodes.items():
            if not name in ("Render Layers", "Composite"):
                tree.nodes.remove(cur_node)
        inp = tree.nodes["Render Layers"]
        composite = tree.nodes["Composite"]
        tree.links.new(inp.outputs["Image"], composite.inputs["Image"])
        tree.links.new(inp.outputs["Alpha"], composite.inputs["Alpha"])


def init_camera():
    """
    Initializes and positions a camera in the Blender scene pointing towards the origin.
    """
    bpy = import_bpy()
    mathutils = blender_importer("mathutils")
    bpy.ops.object.camera_add(
        enter_editmode=False,
        align="VIEW",
        location=(2.0, 2.0, 1.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    point = mathutils.Vector((0, 0, 0))

    camera = bpy.data.objects["Camera"]
    direction = point - camera.location
    rotQuat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rotQuat.to_euler()

    bpy.context.scene.camera = camera


def init_lights(sun_energy=3.5):
    """
    Initializes lighting in the Blender scene with a SUN light type.

    Args:
        sun_energy (float, optional): The energy level of the sun light. Defaults to 3.5.
    """
    bpy = import_bpy()
    bpy.ops.object.light_add(
        type="SUN", radius=1, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    light = bpy.data.objects["Sun"]
    light.data.angle = 25 / 180.0 * math.pi
    light.data.energy = sun_energy

    bpy.data.scenes[0].world.use_nodes = True
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs[
        "Color"
    ].default_value = (0.1, 0.1, 0.1, 1)


# https://blender.stackexchange.com/questions/55702/getting-global-axis-value-of-the-vertex-with-the-lowest-value
def get_obj_min_z(obj):
    """
    Calculates the minimum Z-coordinate of an object in global space.

    Args:
        obj (bpy.types.Object): The Blender object to calculate the minimum Z-coordinate for.

    Returns:
        float: The minimum Z-coordinate of the object.
    """
    mw = obj.matrix_world  # Active object's world matrix
    glob_vertex_coordinates = [mw @ v.co for v in obj.data.vertices]  # Global
    # Find the lowest Z value amongst the object's verts
    minZ = min([co.z for co in glob_vertex_coordinates])
    return minZ


def set_render_settings(
    resolution=1024,
    shadow_catch=False,
    shadow_catcher_z=-1,
    transparent_mode=True,
    line_thickness=0.25,
):
    """
    Configures render settings in the Blender scene.

    Args:
        resolution (int, optional): The resolution for rendering. Defaults to 1024.
        shadow_catch (bool, optional): Flag to add a shadow catcher plane. Defaults to False.
        shadow_catcher_z (float, optional): Z-coordinate for the shadow catcher plane. Defaults to -1.
        transparent_mode (bool, optional): Enables or disables transparent mode. Defaults to True.
        line_thickness (float, optional): Thickness of the lines in the render. Defaults to 0.25.
    """
    bpy = import_bpy()
    if shadow_catch:
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, shadow_catcher_z),
            scale=(1, 1, 1),
        )
        plane_obj = bpy.context.active_object
        plane_obj.name = "shadow_catcher"
        plane_obj.scale[0] = 100
        plane_obj.scale[1] = 100
        plane_obj.scale[2] = 100
        plane_obj.is_shadow_catcher = True

    # Render Settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    if transparent_mode:
        bpy.context.scene.render.film_transparent = True
    else:
        bpy.context.scene.render.film_transparent = True
        # set background color to white:
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        inp = tree.nodes["Render Layers"]
        composite = tree.nodes["Composite"]
        alpha_node = tree.nodes.new(type="CompositorNodeAlphaOver")
        alpha_node.inputs[1].default_value = (16.19, 16.19, 16.19, 1)
        # denoise_node = tree.nodes.new(type="CompositorNodeDenoise")
        # tree.links.new(inp.outputs['Image'], denoise_node.inputs['Image'])
        tree.links.new(inp.outputs["Image"], alpha_node.inputs[2])
        tree.links.new(alpha_node.outputs["Image"], composite.inputs["Image"])
        # Remove link to alpha channel
        for link in inp.outputs["Alpha"].links:
            tree.links.remove(link)

    bpy.context.scene.cycles.samples = 512  # 32
    bpy.context.scene.render.use_freestyle = True
    bpy.context.scene.render.line_thickness = line_thickness  # 0.25
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    freestyle_settings = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
    lineset = freestyle_settings.linesets.active
    lineset.select_silhouette = True
    lineset.select_crease = True
    lineset.select_border = True
    lineset.select_material_boundary = True
    lineset.select_external_contour = True
    lineset.select_edge_mark = True


def render_with_rotation(xy_size, number_frames, save_loc):
    """
    Renders the scene with the camera rotating around the origin.

    Args:
        xy_size (float): The distance of the camera from the origin in the XY plane.
        number_frames (int): The number of frames to render, dictating the rotation increment.
        save_loc (str): The location to save the rendered images.
    """
    bpy = import_bpy()
    mathutils = blender_importer("mathutils")
    origin_point = mathutils.Vector((0, 0, 0))
    for i in range(number_frames):
        theta = (45 + 360 / number_frames * i) * math.pi / 180.0
        camera_x = xy_size * math.cos(theta)
        camera_y = xy_size * math.sin(theta)
        camera = bpy.data.objects["Camera"]
        camera.location = (camera_x, camera_y, camera.location[2])
        direction = origin_point - camera.location
        rotQuat = direction.to_track_quat("-Z", "Y")
        camera.rotation_euler = rotQuat.to_euler()
        save_file_name = f"{save_loc}/{i}.png"
        bpy.context.scene.render.filepath = save_file_name
        bpy.ops.render.render(write_still=True)
