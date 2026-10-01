"""Manifold equipped PyTorch Tensors for auto riemannian gradient calculations.

Notes
-----
Implementation taken from McTorch library [1].

References
----------
.. [1] M. Meghawanshi, P. Jawanpuria, A. Kunchukuttan, H. Kasai, and B. Mishra, 
    McTorch, a manifold optimization library for deep learning.

Author: Mert Indibi (indibimert2@gmail.com)
"""
import weakref
import copy
from typing import Optional

import torch

def inherit_docstring(parent_method):
    def decorator(func):
        func.__doc__ = parent_method.__doc__
        return func
    return decorator


class ManifoldParameter(torch.nn.Parameter):
    r"""A Tensor coupled with a manifold that is considered a module parameter.
    
    Manifold Parameters are special :class:`torch.Tensor` subclasses equipped
    with a :class:`Manifold` that allows for easy riemannian optimization. They
    inherit the properties of :class:`torch.nn.Parameter`.

    Parameters
    ----------
        data: torch.Tensor
            Parameter tensor.
        requires_grad: Optional[bool]
            If the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
        manifold: t_regs.manifolds.Manifold | None
            Manifold object.
    """
    def __new__(
            cls,
            data=None,
            requires_grad=True,
            manifold=None,
            seed: Optional[torch.Generator | int] = None
            ):
        if data is None:
            if manifold is not None:
                if isinstance(seed, int):
                    rng = torch.Generator(device=manifold.device)
                    rng.manual_seed(seed)
                else:
                    rng = seed
                data = manifold.random_point(rng)
            else:
                data = torch.Tensor()
        return torch.nn.Parameter._make_subclass(cls, data, requires_grad)

    def __init__(
            self,
            data=None, # pylint: disable=unused-argument
            requires_grad=True, # pylint: disable=unused-argument
            manifold=None,
            seed: Optional[torch.Generator | int] = None
            ):
        self._manifold = manifold
        self._rgrad = None
        if manifold is not None:
            assert manifold.size == self.size()
            self.register_rgrad_hook()

    def __deepcopy__(self, memo):
        # If already copied in this session, return it
        if id(self) in memo:
            return memo[id(self)]
        # Clone data detached from the computation graph to preserve leaf status
        cloned_data = self.data.clone()
        # Instantiate a new object of your custom class
        result = ManifoldParameter(cloned_data, requires_grad=self.requires_grad)

        # Copy the gradient buffer if it exists
        if self.grad is not None:
            result.grad = self.grad.clone()
        result._manifold = copy.deepcopy(self._manifold)

        # Store in memo dictionary to handle circular references
        memo[id(self)] = result
        return result

    def register_rgrad_hook(self):
        """Register riemannian gradient hook"""
        weak_self = weakref.ref(self)

        def calculate_rgrad(grad):
            var = weak_self()
            if var is None or var._manifold is None:
                return
            var._rgrad = var._manifold.project(self.data, grad)

        self.register_hook(calculate_rgrad)

    @property
    def manifold(self):
        """Manifold accompanying the parameter."""
        return self._manifold

    @property
    def rgrad(self):
        """Riemannian gradient."""
        if self._manifold is not None:
            return self._rgrad

    #TODO: There is an issue with setting the datatype and device of the
    # Manifold class using this .to() method. Need fix.
    @inherit_docstring(torch.nn.Parameter.to)
    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        new_param = new_obj.as_subclass(type(self))
        if self._manifold is not None:
            if hasattr(self._manifold, 'to'):
                new_param._manifold = self._manifold.to(*args, **kwargs)
            else:
                new_param._manifold = self._manifold
        else:
            new_param._manifold = self._manifold
        return new_param
