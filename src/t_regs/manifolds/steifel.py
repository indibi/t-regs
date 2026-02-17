# TODO: Write documentation

import torch

from .manifold import Manifold


class Steifel(Manifold):
    r"""Steifel Manifold

    The Steifel Manifold :math:`St(n,p)` is the space of orthonormal
    ``n \times p`` matrices, i.e. :math:`St(n,p)=\{X \in \mathbb{R}^{n\times p}:
    X^TX = I_p \}.

    Parameters
    ----------
        n : int
            The number of rows
        p : int
            The number of columns
        retraction : str = 'qr'
            Method used to perform the retraction. Options include 'qr',
            'eig', 'polar'.
    """
    retractions = ['qr', 'polar', 'eig']

    def __init__(self,
                 n: int,
                 p: int,
                 retraction: str = 'qr',
                 **kwargs):
        self._n = n
        self._p = p

        if (n<p) or (p<1):
            raise ValueError((f"Invalid dimensions (n={n}, p={p}) for Steifel"
                              " Manifold."))

        if retraction not in self.retractions:
            raise ValueError((f"Invalid retraction type ({retraction}). "
                             f"Valid options are among {self.retractions}"))

        self._retraction = getattr(self, f"_retract_{retraction}")
        dimension = n*p - p*(p+1) /2
        name = f"Steifel Manifold St({n}, {p})"
        super().__init__(name, dimension, **kwargs)

    def inner_product(self,
                    point: torch.Tensor,
                    v1: torch.Tensor,
                    v2: torch.Tensor,
                    project: bool = True) -> float:
        if project:
            tv1= self.project(point, v1)
            tv2= self.project(point, v2)
            return torch.tensordot(tv1,tv2)
        else:
            return torch.tensordot(v1, v2)

    def norm(self, point, v, project: bool = True):
        if project:
            tv = self.project(point, v)
            return torch.linalg.norm(tv) # pylint: disable=not-callable
        else:
            return torch.linalg.norm(v)  # pylint: disable=not-callable


    def random_point(self, generator=None):
        point = torch.randn((self._n, self._p),
                           generator=generator,
                           dtype=self.dtype,
                           device=self.device)
        Q, _ = torch.linalg.qr(point) # pylint: disable=not-callable,invalid-name
        return Q                      # pylint: disable=invalid-name

    def random_tangent(self, point, generator=None):
        vector = torch.randn((self._n, self._p),
                           generator=generator,
                           dtype=self.dtype,
                           device=self.device)
        return self.retract(point, vector)

    def retract(self,
                point: torch.Tensor,
                vector: torch.Tensor) -> torch.Tensor:
        return self._retraction(point, vector)

    def _retract_qr(self, point, vector):
        x = point + vector
        Q, _ = torch.linalg.qr(x) # pylint: disable=not-callable,invalid-name
        return Q                  # pylint: disable=invalid-name

    def _retract_polar(self, point, vector):
        x = point + vector
        U, _, Vt= torch.linalg.svd(x, full_matrices = False) # pylint: disable=not-callable,invalid-name
        return U @ Vt                                        # pylint: disable=invalid-name

    def _retract_eig(self, point, vector):
        raise NotImplementedError("Retraction based on eig is not implemented")

    def project(self,
                X: torch.Tensor,
                Y: torch.Tensor) -> torch.Tensor:
        r"""Project `vector` :math:`Y` onto the tangent space on `point` :math:`X`
        
        Performs the projection operation of the vector :math:`Y`, i.e. 
        :math:`Proj_{T_X \mathcal{M}}(Y)`, onto the tangent space
        :math:`T_X \mathcal{M}` of the manifold at :math:`X`.
        """
        XTY = X.T@Y                                 # pylint: disable=invalid-name
        skewXTY = 0.5 * (XTY - XTY.T)               # pylint: disable=invalid-name
        return (Y - X@XTY) + X @ skewXTY            # pylint: disable=invalid-name
