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
                    NotImplemented yet.
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
        self.__retraction = retraction
        if (n<p) or (p<1):
            raise ValueError((f"Invalid dimensions (n={n}, p={p}) for Steifel"
                              " Manifold."))

        if retraction not in self.retractions:
            raise ValueError((f"Invalid retraction type ({retraction}). "
                                f"Valid options are among {self.retractions}"))
        self._retraction = getattr(self, f"_retract_{retraction}")
        dimension = n*p - p*(p+1) /2
        name = f"Generalized Steifel Manifold St_G({n}, {p})"
        size = torch.Size((n,p))
        super().__init__(name, dimension, size, **kwargs)

        self.__G_eigvals = None # pylint: disable=invalid-name
        self.__G_eigvecs = None # pylint: disable=invalid-name
        self.__sqrt_G = None # pylint: disable=invalid-name
        self.__inv_sqrt_G = None # pylint: disable=invalid-name


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
        X = point   # pylint: disable=invalid-name
        Z = vector  # pylint: disable=invalid-name
        XTGZ = X.T @ self._G @ Z    # pylint: disable=invalid-name
        sym_XTGZ = 0.5 * (XTGZ + XTGZ.T)    # pylint: disable=invalid-name
        projected_vector = Z - X @ sym_XTGZ # pylint: disable=invalid-name
        return projected_vector


    def random_point(
            self,
            generator:torch.Generator=None,
            iterations: int = 5) -> torch.Tensor:
        Xk = torch.randn((self._n, self._p),    # pylint: disable=invalid-name
                               generator=generator,
                               dtype=self.dtype,
                               device=self.device)
        tangent = torch.randn((self._n, self._p),
                               generator=generator,
                               dtype=self.dtype,
                               device=self.device)
        for _ in range(iterations):
            tangent = self.project(Xk, tangent)
            Xk = self.retract(Xk, tangent)  # pylint: disable=invalid-name
        return Xk


    def random_tangent(self,point: torch.Tensor, generator:torch.Generator=None):
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
        Y = point + vector  # pylint: disable=invalid-name
        Z = Y.T @ self._G @ Y   # pylint: disable=invalid-name
        R = torch.linalg.cholesky(Z, upper=True)    # pylint: disable=not-callable,invalid-name
        R_inv = torch.linalg.inv(R)    # pylint: disable=not-callable,invalid-name
        return Y @ R_inv
        # return Y


    def _retract_qr_with_inv_sqrt_G(self,  # pylint: disable=invalid-name
                             point: torch.Tensor,
                             vector: torch.Tensor) -> torch.Tensor:
        Y = point + vector  # pylint: disable=invalid-name
        Y = self.sqrt_G @ Y # pylint: disable=invalid-name
        Q, _ = torch.linalg.qr(Y) # pylint: disable=not-callable,invalid-name
        return self.inv_sqrt_G @ Q


    def tangent_infeasibility(self,
                                 point: torch.Tensor,
                                 vector: torch.Tensor) -> torch.Tensor:
        r"""Compute the relative error of the tangent vector `vector` at `point`
        
        Specifically, computes the error :math:`\frac{\| X^T G V + V^T G X \|_F}
        {\|X^T G V\|_F}`
        
        where :math:`X` is the input `point` and :math:`V` is the input `vector`
        """
        XTGV = point.T @ self._G @ vector   # pylint: disable=invalid-name
        sym_XTGV = XTGV + XTGV.T    # pylint: disable=invalid-name
        frob_norm_num = torch.linalg.norm(sym_XTGV)  # pylint: disable=not-callable
        frob_norm_denom = torch.linalg.norm(XTGV)  # pylint: disable=not-callable
        return frob_norm_num / frob_norm_denom


    def point_infeasibility(self, point: torch.Tensor) -> float:
        r"""Compute the error :math:`\frac{\| X^T G X - I_p \|_F}{\|I_p\|_F}`
        
        where :math:`X` is the input `point`.
        """
        XTGX = point.T @ self._G @ point    # pylint: disable=invalid-name
        I_p = torch.eye(self._p, dtype=self.dtype, device=self.device)  # pylint: disable=invalid-name
        # return XTGX - I_p
        frob_norm = torch.linalg.norm(XTGX - I_p, ord='fro')  # pylint: disable=not-callable
        denom = self._p
        return frob_norm / denom

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._G = self._G.to(*args, **kwargs)
        return self

    def get_properties(self):
        return {
            'n': self._n,
            'p': self._p,
            'G': self._G,
            'retraction': self.__retraction,
        }

    def __set_up_inv_G(self): # pylint: disable=invalid-name
        eigvals, eigvecs = torch.linalg.eigh(self._G) # pylint: disable=not-callable
        self.__G_eigvals = eigvals
        self.__G_eigvecs = eigvecs
        self.__sqrt_G = eigvecs @ torch.diag_embed(eigvecs**0.5) @ eigvecs.T
        self.__inv_sqrt_G = eigvecs@torch.diag_embed(1/eigvecs**0.5) @eigvecs.T

    @property
    def sqrt_G(self):   # pylint: disable=invalid-name
        if self.__sqrt_G is None:
            self.__set_up_inv_G()
        return self.__sqrt_G

    @property
    def inv_sqrt_G(self):   # pylint: disable=invalid-name
        if self.__inv_sqrt_G is None:
            self.__set_up_inv_G()
        return self.__inv_sqrt_G

    @property
    def G_eigvals(self):    # pylint: disable=invalid-name
        if self.__G_eigvals is None:
            self.__set_up_inv_G()
        return self.__G_eigvals

    @property
    def G_eigvecs(self):    # pylint: disable=invalid-name
        if self.__G_eigvecs is None:
            self.__set_up_inv_G()
        return self.__G_eigvecs
