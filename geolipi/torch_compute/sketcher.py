import torch as th
import numpy as np


class Sketcher:

    def __init__(self, device="cuda", dtype=th.float32,
                 resolution=64, mode="direct", n_dims=3):
        if dtype == "float32":
            dtype = th.float32
        self.device = th.device(device)
        self.dtype = dtype
        self.resolution = resolution
        self.mode = mode
        self.n_dims = n_dims
        self.homogeneous_identity = th.eye(n_dims + 1, device=self.device)
        self.zero_mat = th.zeros(n_dims, n_dims, device=self.device)
        self.coords = self.create_coords()
        self.scale_identity = th.ones(
            n_dims, dtype=self.dtype, device=self.device)
        self.translate_identity = th.zeros(
            n_dims, dtype=self.dtype, device=self.device)

    def get_scale_identity(self):
        """Return an identity scale matrix.
        """
        return self.scale_identity.clone().detach()

    def get_translate_identity(self):
        """Return an identity translate matrix.
        """
        return self.translate_identity.clone().detach()

    def get_affine_identity(self):
        """Return an identity affine matrix.
        """
        return self.homogeneous_identity.clone().detach()  # .detach()

    def get_color_canvas(self):
        zeros = th.ones_like(self.coords[..., :1])
        canvas = th.cat(
            [zeros.clone(), zeros.clone(), zeros.clone()], dim=-1)
        return canvas

    def create_coords(self):
        res = self.resolution
        mesh_grid_inp = [range(res), ] * self.n_dims
        points = np.stack(np.meshgrid(*mesh_grid_inp, indexing="ij"), axis=-1)
        points = points.astype(np.float32)
        points = (points / (res-1) - 0.5) * 2
        points = th.from_numpy(points)
        points = th.reshape(points, (-1, self.n_dims)).to(self.device)
        return points

    def get_coords(self, transform):
        coords = self.coords.clone().detach()
        pad = th.ones_like(coords[:, :1])
        points_hom = th.cat([coords, pad], dim=1)
        rotated_points_hom = th.einsum('ij,mj->mi', transform, points_hom)
        rotated_points = rotated_points_hom[:, :self.n_dims]
        return rotated_points

    def get_base_coords(self):
        return self.coords.clone().detach()
