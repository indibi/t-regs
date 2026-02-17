from typing import Sequence
import math

import torch

from .manifold import Manifold

class Euclidean(Manifold):
    r"""Usual Euclidean Manifold

    Parameters
    ----------
        dims:
            Dimensions of the tensor or vector
    """
    def __init__(self,
                 dims: Sequence[int],
                 **kwargs):
        self._dims = dims
        dimension = math.prod(dims)
        name = f"Euclidean Manifold R^({tuple(dims)})"
        super().__init__(name, dimension, **kwargs)

    def inner_product(self,
                    point: torch.Tensor,
                    v1: torch.Tensor,
                    v2: torch.Tensor,
                    project = True) -> float:
        return torch.tensordot(v1, v2)

    def norm(self, point, v, project = True):
        return (v**2).sum()

    def random_point(self, generator=None):
        point = torch.randn(self._dims,
                           generator=generator,
                           dtype=self.dtype,
                           device=self.device)
        return point

    def random_tangent(self, point, generator=None):
        vector = torch.randn(self._dims,
                           generator=generator,
                           dtype=self.dtype,
                           device=self.device)
        return vector

    def retract(self,
                point: torch.Tensor,
                vector: torch.Tensor) -> torch.Tensor:
        return point + vector


    def project(self,
                X: torch.Tensor,
                Y: torch.Tensor) -> torch.Tensor:
        r"""Project `vector` :math:`Y` onto the tangent space on `point` :math:`X`
        
        Performs the projection operation of the vector :math:`Y`, i.e. 
        :math:`Proj_{T_X \mathcal{M}}(Y)`, onto the tangent space
        :math:`T_X \mathcal{M}` of the manifold at :math:`X`.
        """
        return Y

