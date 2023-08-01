import torch as th
import numpy as np
from geolipi.symbolic.primitives import Cuboid, Sphere, Cylinder, Cone


class Sketch3D:

    def __init__(self, device="cuda", resolution=64, mode="direct"):
        self.device = th.device(device)
        self.resolution = resolution
        self.mode = mode
        self.homogeneous_identity = th.eye(4, device=self.device)
        self.zero_cube = th.zeros(3, 3, device=self.device)
        self.coords = self.create_coords()
        self.default_params = {
            Cuboid: th.tensor([1.0, 1.0, 1.0], device=self.device),
            Sphere: th.tensor([1.0], device=self.device),
        }

    def get_affine_identity(self):
        """Return an identity affine matrix.
        """
        return self.homogeneous_identity.detach()  # .detach()

    def create_coords(self):
        res = self.resolution
        points = np.stack(np.meshgrid(range(res), range(res),
                          range(res), indexing="ij"), axis=-1)
        points = points.astype(np.float32)
        points = ((points + 0.5) / res - 0.5) * 2
        points = th.from_numpy(points)
        points = th.reshape(points, (-1, 3)).to(self.device)
        return points

    def get_coords(self, transform):
        coords = self.coords.detach()
        pad = th.ones_like(coords[:, :1])
        points_hom = th.cat([coords, pad], dim=1)
        rotated_points_hom = th.einsum('ij,mj->mi', transform, points_hom)
        rotated_points = rotated_points_hom[:, :3]
        return rotated_points
