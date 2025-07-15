# Tiny <300 line sphere marching renderer for GeoLIPI expressions with PyTorch
# Based on: https://github.com/skmhrk1209/th-Sphere-Tracer
import torch as th
import torch.nn as nn
import numpy as np
import geolipi.symbolic as gls
from .sketcher import Sketcher
from .depreciated_eval import expr_to_sdf
from .evaluate_expression import recursive_evaluate
from geolipi.torch_compute.batch_compile import create_compiled_expr
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate

NUM_ITERATIONS = 2000
CONVERGENCE_THRESHOLD = 1e-3

class Renderer:
    """
    The Renderer class is designed to render 3D scenes based on provided signed distance
    function (SDF) expressions. It supports various rendering settings including camera
    parameters, lighting settings, and material properties. The class can handle both 
    standard and recursive SDF evaluation and offers options for compiling expressions
    to reduce the rendering time.

    Attributes:
        dtype (th.dtype): The data type for PyTorch tensors.
        device (str): The device (e.g., "cuda", "cpu") used for tensor operations.
        resolution (tuple): The resolution of the rendered image.
        camera_params (dict): Parameters defining the camera settings.
        light_setting (dict): Settings for the lighting conditions in the scene.
        num_iterations (int): Number of iterations for the rendering loop.
        convergence_threshold (float): Threshold for convergence in the rendering process.
        sketcher (Sketcher): An instance of the Sketcher class for coordinate manipulation.
        recursive_evaluator (bool): Flag to use recursive evaluation of SDFs.
        secondary_sketcher (Sketcher or None): Secondary Sketcher instance for 2D coordinates.
        compile_expression (bool): Flag to compile the SDF expressions for performance.

    """
    def __init__(self, resolution=(512, 512), num_iterations=NUM_ITERATIONS,
                 convergence_threshold=CONVERGENCE_THRESHOLD,
                 camera_params=None, light_setting=None,
                 recursive_evaluator=True, compile_expression=False,
                 device="cuda", dtype=th.float32):
        self.dtype = th.float32
        self.device = device
        # Camera Matrix
        self.resolution = resolution
        if camera_params is None:
            # Default Camera params
            self.camera_params = dict(
                distance=6.0,
                azimuth=np.pi / 4.0,
                elevation=np.pi / 5.0,
                target_position=th.tensor([0.0, 0, 0.0], dtype=dtype, device=device),
                up_direction=th.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device),
                f=0.75,
                resolution=resolution)
        else:
            # convert lists to tensors:
            for key, value in camera_params.items():
                if isinstance(value, list):
                    camera_params[key] = th.tensor(value, dtype=dtype, device=device)
            self.camera_params = camera_params


        if light_setting is None:
            # Default Light
            self.light_setting = dict(
                light_ambient_color = th.ones(1, 1, 3, device=self.device),
                light_diffuse_color = th.ones(1, 1, 3, device=self.device),
                light_specular_color = th.ones(1, 1, 3, device=self.device),
                light_directions = th.tensor([1.0, -0.5, 0.0], dtype=dtype, device=device))
        else:
            for key, value in light_setting.items():
                if isinstance(value, list):
                    light_setting[key] = th.tensor(value, dtype=dtype, device=device)
            self.light_setting = light_setting

        self.num_iterations = num_iterations
        self.convergence_threshold = convergence_threshold
        # Sketchers:
        # does it matter?
        min_res = 32
        self.sketcher = Sketcher(resolution=min_res, device=device,
                                 dtype=dtype, n_dims=3)
        self.recursive_evaluator = recursive_evaluator
        if recursive_evaluator:
            self.eval_func = recursive_evaluate
            self.secondary_sketcher = Sketcher(resolution=min_res, device=device,
                                            dtype=dtype, n_dims=2)
        else:
            self.eval_func = expr_to_sdf
            self.secondary_sketcher = None
        self.compile_expression = compile_expression
        if self.compile_expression:
            assert not recursive_evaluator, "Cannot compile expressions which require recursive eval."

    def get_camera_matrix(self, resolution, f, *args, **kwargs):
        cx, cy = resolution[0], resolution[1]
        camera_matrix = th.tensor([[f, 0.0, 0], [0.0, f * cx / cy, 0], [0.0, 0.0, 1.0]],
                                  dtype=self.dtype, device=self.device)
        return camera_matrix

    def get_camera_position(self, distance, azimuth, elevation, *args, **kwargs):
        camera_position = th.tensor([
            +np.cos(elevation) * np.sin(azimuth),
            -np.sin(elevation),
            -np.cos(elevation) * np.cos(azimuth)
        ], dtype=self.dtype, device=self.device) * distance
        return camera_position

    def get_camera_rotation(self, camera_position, target_position, up_direction, *args, **kwargs):
        camera_z_axis = target_position - camera_position
        camera_x_axis = th.cross(up_direction, camera_z_axis, dim=-1)
        camera_y_axis = th.cross(camera_z_axis, camera_x_axis, dim=-1)
        camera_rotation = th.stack((camera_x_axis, camera_y_axis, camera_z_axis), dim=-1)
        camera_rotation = nn.functional.normalize(camera_rotation, dim=-2)
        return camera_rotation

    def get_rays(self, camera_position):

        camera_matrix = self.get_camera_matrix(**self.camera_params)
        camera_rotation = self.get_camera_rotation(camera_position, **self.camera_params)
        cx, cy = self.resolution[0], self.resolution[1]
        # better to do this as get -1 to 1 coords.
        y_positions = (th.arange(cy, dtype=self.dtype, device=self.device) / cy) - 0.5
        x_positions = (th.arange(cx, dtype=self.dtype, device=self.device) / cx) - 0.5
        y_positions, x_positions = th.meshgrid(y_positions, x_positions, indexing="ij")
        z_positions = th.ones_like(y_positions)
        ray_positions = th.stack((x_positions, y_positions, z_positions), dim=-1)
        ray_positions = ray_positions.reshape(-1, 3)
        # TODO: Can be refactored.
        ray_positions = th.einsum("mn,...n->...m", th.inverse(camera_matrix),  ray_positions)
        ray_positions = th.einsum("mn,...n->...m", camera_rotation, ray_positions) + camera_position
        ray_directions = nn.functional.normalize(ray_positions - camera_position, dim=-1)
        return ray_positions, ray_directions
    
    def render(self, sdf_expression, ground_expression=None, 
               material_setting=None, finite_difference_epsilon=None,
               convergence_threshold=None, num_iterations=None):
        """
        Renders a 3D scene based on the provided signed distance function (SDF) expressions.

        Args:
            sdf_expression: The primary SDF expression defining the shapes in the scene.
            ground_expression (optional): An additional SDF expression for the ground plane or similar.
            material_setting (optional): Settings for the material properties of the rendered objects.
            finite_difference_epsilon (optional): The epsilon value used in finite difference calculations for normals.
            convergence_threshold (optional): Threshold for convergence in the rendering loop. Defaults to class attribute if None.
            num_iterations (optional): Number of iterations for the rendering loop. Defaults to class attribute if None.

        Returns:
            A tensor representing the rendered image, with shape corresponding to the specified resolution.
        """
        if convergence_threshold is None:
            convergence_threshold = self.convergence_threshold
        if num_iterations is None:
            num_iterations = self.num_iterations
        camera_position = self.get_camera_position(**self.camera_params)
        ray_positions, ray_directions = self.get_rays(camera_position)
        if material_setting is None:
            material_setting = self.make_random_material()
        if ground_expression is None:
            ground_expression = self.create_ground_expr()

        overall_expression = gls.Union(sdf_expression, ground_expression)

        if self.compile_expression:
            compiled_obj = create_compiled_expr(overall_expression, self.sketcher, convert_to_cpu=False)
            expr_set = create_evaluation_batches([compiled_obj], convert_to_cuda=False)
            def sdf_eval_func(ray_points): return batch_evaluate(expr_set, sketcher=self.sketcher,
                                                                coords=ray_points).squeeze(0).unsqueeze(-1)
        else:
            def sdf_eval_func(ray_points): return self.eval_func(overall_expression, sketcher=self.sketcher,
                                                                secondary_sketcher=self.secondary_sketcher,
                                                                coords=ray_points).unsqueeze(-1)
            # We can use batching
        def ground_eval_func(ray_points): return self.eval_func(ground_expression, sketcher=self.sketcher,
                                                                secondary_sketcher=self.secondary_sketcher,
                                                                coords=ray_points).unsqueeze(-1)

        image = render_expression(sdf_eval_func, ground_eval_func,
                                  ray_positions, ray_directions,
                                  camera_position, 
                                  light_setting=self.light_setting,
                                  material_setting=material_setting,
                                  num_iterations=self.num_iterations,
                                  convergence_threshold=self.convergence_threshold,
                                  finite_difference_epsilon=finite_difference_epsilon)
        image = image.reshape(self.resolution[1], self.resolution[0], 3)
        return image

    def make_random_material(self):
        material_ambient_color = th.full((1, 1, 3), 0.2, device=self.device) + (th.rand(1, 1, 3, device=self.device) * 2 - 1) * 0.1
        material_diffuse_color = th.full((1, 1, 3), 0.7, device=self.device) + (th.rand(1, 1, 3, device=self.device) * 2 - 1) * 0.1
        material_specular_color = th.full((1, 1, 3), 0.1, device=self.device)
        material_emission_color = th.zeros(1, 1, 3, device=self.device)
        material_shininess = 64.0
        material_setting = dict(
            material_ambient_color=material_ambient_color,
            material_diffuse_color=material_diffuse_color,
            material_specular_color=material_specular_color,
            material_emission_color=material_emission_color,
            material_shininess=material_shininess,
        )
        return material_setting

    def create_ground_expr(self):
        n = th.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self.device)
        d = th.tensor([1.0], dtype=self.dtype, device=self.device)
        ground_expr = gls.Plane3D(n, d)
        return ground_expr


def sphere_tracing(sdf_eval_func, ray_positions, ray_directions,
                   num_iterations, convergence_threshold,
                   foreground_masks=None, bounding_radius=None):
    # vanilla sphere tracing
    if foreground_masks is None:
        foreground_masks = th.all(th.isfinite(ray_positions), dim=-1, keepdim=True)

    if bounding_radius:
        a = th.sum(ray_directions * ray_directions, dim=-1, keepdim=True)
        b = 2 * th.sum(ray_directions * ray_positions, dim=-1, keepdim=True)
        c = th.sum(ray_positions * ray_positions, dim=-1, keepdim=True) - bounding_radius ** 2
        d = b ** 2 - 4 * a * c
        t = (-b - th.sqrt(d)) / (2 * a)
        bounded = d >= 0
        ray_positions = th.where(bounded, ray_positions + ray_directions * t, ray_positions)
        foreground_masks = foreground_masks & bounded

    # vanilla sphere tracing
    with th.no_grad():
        for i in range(num_iterations):
            # TODO: make this work for only the unconverged points.
            signed_distances = sdf_eval_func(ray_positions)
            if i:
                ray_positions = th.where(foreground_masks & ~converged, ray_positions + ray_directions * signed_distances, ray_positions)
            else:
                ray_positions = th.where(foreground_masks, ray_positions + ray_directions * signed_distances, ray_positions)
            if bounding_radius:
                bounded = th.norm(ray_positions, dim=-1, keepdim=True) < bounding_radius
                foreground_masks = foreground_masks & bounded
            converged = th.abs(signed_distances) < convergence_threshold
            if th.all(~foreground_masks | converged):
                break

    return ray_positions, converged


def compute_shadows(sdf_eval_func, surface_positions, surface_normals, light_directions,
                    num_iterations, convergence_threshold,
                    foreground_masks=None, bounding_radius=None):
    surface_positions, converged = sphere_tracing(
        sdf_eval_func=sdf_eval_func,
        ray_positions=surface_positions + surface_normals * 1e-3,
        ray_directions=light_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
        foreground_masks=foreground_masks,
        bounding_radius=bounding_radius)
    return converged


def compute_normal(sdf_eval_func, surface_positions, finite_difference_epsilon=None):

    if finite_difference_epsilon:
        finite_difference_epsilon_x = surface_positions.new_tensor([finite_difference_epsilon, 0.0, 0.0])
        finite_difference_epsilon_y = surface_positions.new_tensor([0.0, finite_difference_epsilon, 0.0])
        finite_difference_epsilon_z = surface_positions.new_tensor([0.0, 0.0, finite_difference_epsilon])
        surface_normals_x = sdf_eval_func(surface_positions + finite_difference_epsilon_x) - sdf_eval_func(surface_positions - finite_difference_epsilon_x)
        surface_normals_y = sdf_eval_func(surface_positions + finite_difference_epsilon_y) - sdf_eval_func(surface_positions - finite_difference_epsilon_y)
        surface_normals_z = sdf_eval_func(surface_positions + finite_difference_epsilon_z) - sdf_eval_func(surface_positions - finite_difference_epsilon_z)
        surface_normals = th.cat((surface_normals_x, surface_normals_y, surface_normals_z), dim=-1)
    else:
        create_graph = surface_positions.requires_grad
        surface_positions.requires_grad_(True)
        with th.enable_grad():
            signed_distances = sdf_eval_func(surface_positions)
            surface_normals, = th.autograd.grad(
                outputs=signed_distances,
                inputs=surface_positions,
                grad_outputs=th.ones_like(signed_distances),
                create_graph=create_graph)

    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    
    return surface_normals


def phong_shading(surface_normals, view_directions,
        light_directions, light_ambient_color, light_diffuse_color, light_specular_color,
        material_ambient_color, material_diffuse_color, material_specular_color, material_emission_color, material_shininess):
    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    view_directions = nn.functional.normalize(view_directions, dim=-1)
    light_directions = nn.functional.normalize(light_directions, dim=-1)

    reflected_directions = 2 * surface_normals * th.sum(light_directions * surface_normals, dim=-1, keepdim=True) - light_directions

    diffuse_coefficients = nn.functional.relu(th.sum(light_directions * surface_normals, dim=-1, keepdim=True))
    specular_coefficients = nn.functional.relu(th.sum(reflected_directions * view_directions, dim=-1, keepdim=True)) ** material_shininess

    images = th.clamp(
        material_emission_color +
        material_ambient_color * light_ambient_color +
        material_diffuse_color * light_diffuse_color * diffuse_coefficients +
        material_specular_color * light_specular_color * specular_coefficients,
        0.0, 1.0)

    return images


def render_expression(sdf_eval_func, ground_eval_func,
                      ray_positions, ray_directions,
                      camera_position, light_setting,
                      material_setting,
                      num_iterations, convergence_threshold,
                      finite_difference_epsilon=None):
    """
    Renders a 3D scene using the provided sdf evaluation functions and rendering settings.

    This function implements the rendering pipeline, including sphere tracing for surface
    detection, normal computation, Phong shading for color calculation, ground and shadow
    handling, and convergence checks.

    Args:
        sdf_eval_func: A function to evaluate the shape's signed distance function (SDF).
        ground_eval_func: A function to evaluate the ground SDF.
        ray_positions: The starting positions of the rays.
        ray_directions: The directions of the rays.
        camera_position: The position of the camera.
        light_setting: Settings for the scene's lighting conditions.
        material_setting: Settings for the material properties of the objects.
        num_iterations (int): Number of iterations for sphere tracing.
        convergence_threshold (float): Threshold for convergence in the sphere tracing.
        finite_difference_epsilon (optional): Epsilon value for finite difference in normal computation.

    Returns:
        A tensor representing the rendered image, with shading and lighting effects applied.
    """

    surface_positions, converged = sphere_tracing(
        sdf_eval_func=sdf_eval_func,
        ray_positions=ray_positions,
        ray_directions=ray_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
    )
    surface_positions = th.where(converged, surface_positions, th.zeros_like(surface_positions))

    surface_normals = compute_normal(sdf_eval_func=sdf_eval_func, 
                                     surface_positions=surface_positions,
                                     finite_difference_epsilon=finite_difference_epsilon)

    surface_normals = th.where(converged, surface_normals, th.zeros_like(surface_normals))

    image = phong_shading(surface_normals=surface_normals,
                          view_directions=camera_position - surface_positions,
                          **light_setting, **material_setting)

    grounded_val = ground_eval_func(surface_positions)

    grounded = th.abs(grounded_val) < convergence_threshold
    image = th.where(grounded, th.full_like(image, 0.9), image)

    shadowed = compute_shadows(
        sdf_eval_func=sdf_eval_func,
        surface_positions=surface_positions,
        surface_normals=surface_normals,
        light_directions=light_setting['light_directions'],
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
        foreground_masks=converged)
    image = th.where(shadowed, image * 0.5, image)

    image = th.where(converged, image, th.zeros_like(image))
    return image
