import torch as th
import numpy as np
from .settings import Settings

class Sketcher:
    """
    The Sketcher class provides a framework for creating and manipulating a grid
    of coordinates in a specified number of dimensions. It supports operations 
    such as scaling, translating, and generating homogenous coordinates. This 
    class is designed to work with PyTorch tensors.

    Attributes:
        device (th.device): The PyTorch device (e.g., CUDA or CPU) used for tensor operations.
        dtype (th.dtype): The data type for PyTorch tensors (e.g., float32).
        resolution (int): The resolution of the coordinate grid.
        mode (str): The mode of operation (currently unused).
        n_dims (int): The number of dimensions for the coordinates.
        coord_scale (float): The scale factor applied to the coordinates.
        homogeneous_identity (th.Tensor): An identity matrix in homogeneous coordinates.
        zero_mat (th.Tensor): A zero matrix of size n_dims x n_dims.
        coords (th.Tensor): The coordinate grid tensor.
        scale_identity (th.Tensor): An identity scale tensor.
        translate_identity (th.Tensor): An identity translate tensor.

    """

    def __init__(self, device: str = "cuda", dtype: th.dtype = th.float32,
                 resolution: int = 64, mode: str = "direct", n_dims: int = 3,
                 coord_scale: float = 1.0):

        if dtype == "float32":
            dtype = th.float32
        if isinstance(device, str):
            self.device = th.device(device)
        else:
            self.device = device
        self.dtype = dtype
        self.resolution = resolution
        self.mode = mode
        self.n_dims = n_dims
        self.coord_scale = coord_scale
        self.homogeneous_identity = th.eye(n_dims + 1, device=self.device)
        self.zero_mat = th.zeros(n_dims, n_dims, device=self.device)
        if Settings.COORD_MODE == "bound":
            self.create_coords = self.create_bound_coords
        elif Settings.COORD_MODE == "centered":
            self.create_coords = self.create_centered_coords
        self.coords = self.create_coords()
        self.scale_identity = th.ones(n_dims, dtype=self.dtype, device=self.device)
        self.translate_identity = th.zeros(n_dims, dtype=self.dtype, device=self.device)

    def adapt_coords(self, scale, origin=None):
        coords = self.create_coords()
        
        if isinstance(scale, (int, float)):
            coords = coords * scale
        elif isinstance(scale, (tuple, list)):
            for i in range(self.n_dims):
                coords[:, i] = coords[:, i] * scale[i]
        else:
            raise ValueError("Invalid scale value.")
        
        if not origin is None:
            if isinstance(origin, (int, float)):
                coords = coords + origin
            elif isinstance(origin, (tuple, list)):
                for i in range(self.n_dims):
                    coords[:, i] = coords[:, i] + origin[i]
            else:
                raise ValueError("Invalid scale value.")
            
        self.coords = coords
    def adapt_coords_from_bounds(self, min_xy, max_xy):
        scale = tuple((max_xy[i] - min_xy[i]) / 2.0 for i in range(self.n_dims))
        origin = tuple((max_xy[i] + min_xy[i]) / 2.0 for i in range(self.n_dims))
        self.adapt_coords(scale=scale, origin=origin)
        
    def reset_coords(self):
        self.adapt_coords((1.0, 1.0), (0.0, 0.0))
        
    def get_scale_identity(self):
        """Return an identity scale matrix."""
        return self.scale_identity.clone().detach()

    def get_translate_identity(self):
        """Return an identity translate matrix."""
        return self.translate_identity.clone().detach()

    def get_affine_identity(self):
        """Return an identity affine matrix."""
        return self.homogeneous_identity.clone().detach()  # .detach()

    def get_color_canvas(self):
        canvas = th.ones_like(self.coords[..., :1]).repeat(1, 4)
        return canvas

    
    def create_bound_coords(self):
        res = self.resolution
        mesh_grid_inp = [range(res),] * self.n_dims
        points = np.stack(np.meshgrid(*mesh_grid_inp, indexing="ij"), axis=-1)
        points = points.astype(np.float32)
        points = (points / (res - 1) - 0.5) * 2
        points = points * self.coord_scale
        points = th.from_numpy(points)
        points = th.reshape(points, (-1, self.n_dims)).to(self.device)
        return points
    
    def create_centered_coords(self):
        res = self.resolution
        mesh_grid_inp = [range(res),] * self.n_dims
        points = np.stack(np.meshgrid(*mesh_grid_inp, indexing="ij"), axis=-1)
        points = points.astype(np.float32)
        points = (points / res - (res-1)/(2*res)) * 2
        points = points * self.coord_scale
        points = th.from_numpy(points)
        points = th.reshape(points, (-1, self.n_dims)).to(self.device)
        return points

    def get_coords(self, transform, points):
        if points is None:
            coords = self.get_base_coords()
        else:
            coords = points
        pad = th.ones_like(coords[:, :1])
        points_hom = th.cat([coords, pad], dim=1)
        rotated_points_hom = th.einsum("ij,mj->mi", transform, points_hom)
        rotated_points = rotated_points_hom[:, : self.n_dims]
        return rotated_points

    def get_base_coords(self):
        return self.coords.clone().detach()

    def get_homogenous_coords(self):
        coords = self.get_base_coords()
        coords = self.make_homogenous_coords(coords)
        return coords

    def make_homogenous_coords(self, coords):
        pad = th.ones_like(coords[:, :1])
        points_homog = th.cat([coords, pad], dim=1)
        return points_homog

    def empty_sdf(self):
        coords = self.get_base_coords()
        sdf = th.norm(coords, dim=-1)
        return sdf
