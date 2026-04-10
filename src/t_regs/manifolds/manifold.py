"""Abstract Base Class for Manifold."""
import abc
from typing import Dict, Any

import torch

class Manifold(abc.ABC):
    """Riemannian manifold base class
    
    Abstract Base Class as a template for manifold classes

    Parameters
    ----------
        name : str
            String representation for the manifold
        dimension : int
            Dimension of the tangent vector space for the manifold
        size: torch.Size
            The size of the tensor object that holds the point.
    """
    # TODO: Find if we really need to specify the device and datatype and get
    # rid of them if it's possible. Probably working with the nn.Parameter
    # container and the nn.Module containers should allow me to get rid of them.
    def __init__(
            self,
            name : str,
            dimension : int,
            size: torch.Size,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            dtype: torch.dtype = torch.float64):
        if (not isinstance(dimension, int)) and (dimension <0):
            raise TypeError("Manifold dimension must be a non-negative integer")
        self._name = name
        self._dimension = dimension
        self._size = size
        self.device = device
        self.dtype = dtype

    def __str__(self):
        return self._name

    @property
    def size(self) -> torch.Size:
        """The size of the tensor that encodes the point."""
        return self._size

    @property
    def dim(self) -> int:
        """The dimension of the manifold"""
        return self._dimension

    @abc.abstractmethod
    def norm(self,
            point: torch.Tensor,
            vector: torch.Tensor,
            project: bool = True) -> float:
        """Calculate the norm associated to the riemannian metric of `vector`
        
        Parameters
        ----------
            point:
                Point lying on the manifold
            vector:
                Vector whose norm is to be evaluated
            project: bool
                When set to False, assumes that `vector` is already on the
                tangent space of the manifold at `point`, otherwise projects it
                to the tangent space before evaluating.
        """

    @abc.abstractmethod
    def inner_product(self, point: torch.Tensor,
                      v1: torch.Tensor,
                      v2: torch.Tensor,
                      project: bool = True) -> float:
        """Evaluate the riemmannian metric of tangent vectors `v1` and `v2` at `point`
        
        Parameters
        ----------
            point:
                Point lying on the manifold
            v1:
                First tangent vector
            v2:
                Secont tangent vector
            project: bool
                When set to False, assumes that `v1` and `v2 are already on the
                tangent space of the manifold at `point`, otherwise projects it
                to the tangent space before evaluating.
        """

    @abc.abstractmethod
    def random_point(self, generator:int) -> torch.Tensor:
        """Return a random point on the manifold"""

    @abc.abstractmethod
    def random_tangent(self, point: torch.Tensor, generator) -> torch.Tensor:
        """Return a random tangent vector at the `point` on the manifold"""

    @abc.abstractmethod
    def retract(self, point: torch.Tensor, vector: torch.Tensor)-> torch.Tensor:
        """Retract the `point` back onto the manifold."""

    @abc.abstractmethod
    def project(self,
                point: torch.Tensor,
                vector: torch.Tensor) -> torch.Tensor:
        """Project `vector` onto tangent space of the manifold at the `point`"""

    def to(self, *args, **kwargs):
        """Move the manifold to device if it is defined through a torch.Tensor"""
        
        dtargs = list(filter(lambda x: isinstance(x, torch.dtype),args))
        if len(dtargs) == 0:
            dtype = kwargs.get('dtype', None)
        else:
            dtype = dtargs[0]
        devargs = list(filter(lambda x: isinstance(x, torch.device),args))
        if len(dtargs) == 0:
            device = kwargs.get('device', None)
        else:
            device = devargs[0]
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        return self

    @abc.abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Get properties of the manifold class used to initialize it."""
