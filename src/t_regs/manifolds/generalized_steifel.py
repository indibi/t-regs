"""

References
----------
..  [1] Sato, Hiroyuki, and Kensuke Aihara. "Cholesky QR-based retraction on
    the generalized Stiefel manifold." Computational Optimization and 
    Applications 72.2 (2019): 293-308.
    [2] Liu, Xin, Nachuan Xiao, and Ya-xiang Yuan. "A penalty-free infeasible
    approach for a class of nonsmooth optimization problems over the Stiefel
    manifold." Journal of Scientific Computing 99.2 (2024): 30.
"""

import torch

from .manifold import Manifold

class GeneralizedSteifel(Manifold):
    """Generalized Steifel manifold St_G(n, p) with G symmetric positive
    definite matrix.

    The generalized Stiefel manifold is defined as 

    .. math::
        St_G(n, p) = {X in R^{n x p} : X^T G X = I_p}

    where G is a symmetric positive definite matrix.

    Parameters
    ----------
        n : int
            Number of rows.
        p : int
            Number of columns.
        G : torch.Tensor
            Symmetric positive definite matrix defining the inner product.
        retraction : str
            Retraction method to use. Options are `'qr_with_inv_R'`, 
            `'qr_with_inv_sqrt_G'`, `'polar'`. Defaults to `'qr_with_inv_R'`.
                `qr_with_inv_R` : QR based retraction with complexity 
                    O(np^2 + p^3) as described in [1].
                `qr_with_inv_sqrt_G` : QR based retraction with complexity
                    O(np^2 + n^3) as described in [1]. This method calculates
                    the eigen decomposition of G to compute G^{-1/2}. It may
                    be more efficient when G is cyclic etc. NotImplemented yet.
                `polar` : Polar decomposition based retraction with complexity
                    NotImplemented yet.766666667
        **kwargs : dict, optional
            Additional keyword arguments for the Manifold base class.
    """
    retractions = ['qr_with_inv_R', 'qr_with_inv_sqrt_G', 'polar']
    def __init__(self,
                 n: int,
                 p: int,
                 G: torch.Tensor,
                 retraction: str = 'qr_with_inv_R',
                 **kwargs):
        self._n = n
        self._p = p
        self._G = G # pylint: disable=invalid-name

        if (n<p) or (p<1):
            raise ValueError((f"Invalid dimensions (n={n}, p={p}) for Steifel"
                              " Manifold."))

        if retraction not in self.retractions:
            raise ValueError((f"Invalid retraction type ({retraction}). "
                                f"Valid options are among {self.retractions}"))
        self._retraction = getattr(self, f"_retract_{retraction}")
        dimension = n*p - p*(p+1) /2
        name = f"Generalized Steifel Manifold St_G({n}, {p})"
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


    def norm(self, point, vector, project: bool = True):
        if project:
            tv = self.project(point, vector)
            return torch.sqrt(torch.tensordot(tv, tv))
        else:
            return torch.sqrt(torch.tensordot(vector, vector))


    def project(self,
                point: torch.Tensor,
                vector: torch.Tensor) -> torch.Tensor:
        X = point
        Z = vector
        XTGZ = X.T @ self._G @ Z
        sym_XTGZ = 0.5 * (XTGZ + XTGZ.T)
        projected_vector = Z - X @ sym_XTGZ
        return projected_vector


    def random_point(self, generator=None, iterations: int = 5,) -> torch.Tensor:
        Xk = torch.randn((self._n, self._p),
                               generator=generator,
                               dtype=self.dtype,
                               device=self.device)
        tangent = torch.randn((self._n, self._p),
                               generator=generator,
                               dtype=self.dtype,
                               device=self.device)
        for _ in range(iterations):
            tangent = self.project(Xk, tangent)
            Xk = self.retract(Xk, tangent)
        return Xk


    def random_tangent(self, point, generator=None):
        vector = torch.randn((self._n, self._p),
                           generator=generator,
                           dtype=self.dtype,
                           device=self.device)
        return self.project(point, vector)

    def retract(self, 
                point: torch.Tensor,
                vector: torch.Tensor) -> torch.Tensor:
        return self._retraction(point, vector)


    def _retract_qr_with_inv_R(self,    # pylint: disable=invalid-name
                             point: torch.Tensor,
                             vector: torch.Tensor) -> torch.Tensor:
        Y = point + vector
        Z = Y.T @ self._G @ Y
        R = torch.linalg.cholesky(Z, upper=True)    # pylint: disable=not-callable
        R_inv = torch.linalg.inv(R)    # pylint: disable=not-callable
        return Y @ R_inv
        # return Y


    def _retract_qr_with_inv_sqrt_G(self,  # pylint: disable=invalid-name
                             point: torch.Tensor,
                             vector: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(("Retraction based on QR with inv sqrt G"
                                  " is not yet implemented"))


    def tangent_infeasibility(self,
                                 point: torch.Tensor,
                                 vector: torch.Tensor) -> torch.Tensor:
        r"""Compute the relative error of the tangent vector `vector` at `point`
        
        Specifically, computes the error :math:`\frac{\| X^T G V + V^T G X \|_F}
        {\|X^T G V\|_F}`
        
        where :math:`X` is the input `point` and :math:`V` is the input `vector`
        """
        XTGV = point.T @ self._G @ vector
        sym_XTGV = XTGV + XTGV.T
        frob_norm_num = torch.linalg.norm(sym_XTGV)  # pylint: disable=not-callable
        frob_norm_denom = torch.linalg.norm(XTGV)  # pylint: disable=not-callable
        return frob_norm_num / frob_norm_denom


    def point_infeasibility(self, point: torch.Tensor) -> float:
        r"""Compute the error :math:`\frac{\| X^T G X - I_p \|_F}{\|I_p\|_F}`
        
        where :math:`X` is the input `point`.
        """
        XTGX = point.T @ self._G @ point
        I_p = torch.eye(self._p, dtype=self.dtype, device=self.device)
        # return XTGX - I_p
        frob_norm = torch.linalg.norm(XTGX - I_p, ord='fro')  # pylint: disable=not-callable
        denom = self._p
        return frob_norm / denom