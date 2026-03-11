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
        self.homogeneous_identity = th.eye(n_dims + 1, device=self.device, dtype=self.dtype)
        self.zero_mat = th.zeros(n_dims, n_dims, device=self.device, dtype=self.dtype)
        if Settings.COORD_MODE == "bound":
            self.create_coords = self.create_bound_coords
        elif Settings.COORD_MODE == "centered":
            self.create_coords = self.create_centered_coords
        self.coords = self.create_coords()
        self.scale_identity = th.ones(n_dims, dtype=self.dtype, device=self.device)
        self.translate_identity = th.zeros(n_dims, dtype=self.dtype, device=self.device)
        self.frame_origin = th.zeros(n_dims, dtype=self.dtype, device=self.device)
        self.frame_scale = th.ones(n_dims, dtype=self.dtype, device=self.device)

    def adapt_coords(self, scale, origin=None):
        coords = self.create_coords()

        frame_scale = self.process_scale(scale)
        coords = coords * frame_scale

        frame_origin = self.process_origin(origin)
        if frame_origin is not None:
            coords = coords + frame_origin

        self.coords = coords
        self.frame_scale = frame_scale
        self.frame_origin = frame_origin


    def process_scale(self, scale):
        """
        Normalize `scale` into a 1D tensor of length `n_dims` on the correct device/dtype.

        Accepted types:
        - int, float, numpy scalar → isotropic scale (same in all dimensions)
        - tuple / list / 1D numpy array / 1D tensor of length `n_dims`
        - scalar tensor / 0D or 1-element numpy array → isotropic scale
        """
        # Handle numpy types early
        if isinstance(scale, np.ndarray):
            if scale.ndim == 0:
                scale = float(scale)
            else:
                scale = th.from_numpy(scale)
        elif np.isscalar(scale):
            scale = float(scale)

        # Python scalar → isotropic
        if isinstance(scale, (int, float)):
            return th.full(
                (self.n_dims,),
                float(scale),
                dtype=self.dtype,
                device=self.device,
            )

        # Sequence → tensor
        if isinstance(scale, (tuple, list)):
            scale = th.tensor(scale, dtype=self.dtype, device=self.device)

        # Tensor (from torch or converted from numpy/sequence)
        if isinstance(scale, th.Tensor):
            scale = scale.to(device=self.device, dtype=self.dtype)

            if scale.numel() == 1:
                # Broadcast scalar tensor
                return th.full(
                    (self.n_dims,),
                    float(scale.item()),
                    dtype=self.dtype,
                    device=self.device,
                )

            # Expect last dimension to match n_dims
            if scale.shape[-1] != self.n_dims:
                raise ValueError(
                    f"Scale tensor must have last dimension {self.n_dims}, "
                    f"got shape {tuple(scale.shape)}."
                )
            return scale

        raise ValueError(f"Invalid scale value: {scale!r}")


    def process_origin(self, origin):
        """
        Normalize `origin` into a 1D tensor of length `n_dims` on the correct device/dtype.

        Accepted types:
        - None → returns None
        - int, float, numpy scalar → same translation in all dimensions
        - tuple / list / 1D numpy array / 1D tensor of length `n_dims`
        - scalar tensor / 0D or 1-element numpy array → same translation in all dimensions
        """
        if origin is None:
            return None

        # Handle numpy types early
        if isinstance(origin, np.ndarray):
            if origin.ndim == 0:
                origin = float(origin)
            else:
                origin = th.from_numpy(origin)
        elif np.isscalar(origin):
            origin = float(origin)

        # Python scalar → isotropic translation
        if isinstance(origin, (int, float)):
            return th.full(
                (self.n_dims,),
                float(origin),
                dtype=self.dtype,
                device=self.device,
            )

        # Sequence → tensor
        if isinstance(origin, (tuple, list)):
            origin = th.tensor(origin, dtype=self.dtype, device=self.device)

        # Tensor (from torch or converted from numpy/sequence)
        if isinstance(origin, th.Tensor):
            origin = origin.to(device=self.device, dtype=self.dtype)

            if origin.numel() == 1:
                return th.full(
                    (self.n_dims,),
                    float(origin.item()),
                    dtype=self.dtype,
                    device=self.device,
                )

            # Expect last dimension to match n_dims
            if origin.shape[-1] != self.n_dims:
                raise ValueError(
                    f"Origin tensor must have last dimension {self.n_dims}, "
                    f"got shape {tuple(origin.shape)}."
                )
            return origin
        raise ValueError(f"Invalid origin value: {origin!r}")
        
    def reset_coords(self):
        self.frame_origin = th.zeros(self.n_dims, dtype=self.dtype, device=self.device)
        self.frame_scale = th.ones(self.n_dims, dtype=self.dtype, device=self.device)
        self.coords = self.create_coords()
        
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
        points = th.reshape(points, (-1, self.n_dims)).to(self.device).to(self.dtype)
        return points
    
    def create_centered_coords(self):
        res = self.resolution
        mesh_grid_inp = [range(res),] * self.n_dims
        points = np.stack(np.meshgrid(*mesh_grid_inp, indexing="ij"), axis=-1)
        points = points.astype(np.float32)
        points = (points / res - (res-1)/(2*res)) * 2
        points = points * self.coord_scale
        points = th.from_numpy(points)
        points = th.reshape(points, (-1, self.n_dims)).to(self.device).to(self.dtype)
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
        pad = th.ones_like(coords[..., -1:])
        points_homog = th.cat([coords, pad], dim=-1)
        return points_homog

    def empty_sdf(self):
        coords = self.get_base_coords()
        sdf = th.norm(coords, dim=-1)
        return sdf

    def set_non_square_coords(self, scale, origin):
        self.coords, _ = self.create_non_square_coords(scale, origin)

    def create_non_square_coords(self, scale, origin):
        res = self.resolution
        # resolution is a scalar; voxel size = 2 / resolution
        voxel_size = 2.0 / self.resolution     # with resolution=2 → voxel_size = 1.0

        # --- normalize scale & origin ---
        if isinstance(scale, (int, float)):
            scale = [scale] * self.n_dims
        scale = np.array(scale, dtype=np.float32)

        if isinstance(origin, (int, float)):
            origin = [origin] * self.n_dims
        origin = np.array(origin, dtype=np.float32)

        # number of points along each axis = scale / voxel_size
        dims = (scale / voxel_size).astype(int)   # e.g. (2,4)

        # build voxel index grid
        axes = [np.arange(n, dtype=np.float32) for n in dims]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)   # (2,4,2)

        # convert voxel indices to world coords:
        # center grid around origin, then multiply by voxel size
        # grid spans [-(dims[i]-1)/2 ... +(dims[i]-1)/2]
        pts = (grid - (dims - 1)/2) * voxel_size + origin

        # flatten to (N, n_dims)
        shape = pts.shape[:-1]
        pts = pts.reshape(-1, self.n_dims)

        return th.from_numpy(pts).to(self.device), shape