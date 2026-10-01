"""Module defining the Problem class for manifold optimization.

Notes
-----
The implementation is influenced by the PyManopt [1] and McTorch [2] packages.

References
----------
..  [1] Townsend, James, Niklas Koep, and Sebastian Weichwald. "Pymanopt: A 
    python toolbox for optimization on manifolds using automatic 
    differentiation." Journal of Machine Learning Research 17, no. 137 
    (2016): 1-5.
..  [2] M. Meghawanshi, P. Jawanpuria, A. Kunchukuttan, H. Kasai, and B. 
    Mishra, McTorch, a manifold optimization library for deep learning.
"""
# TODO: Figure out a way to structure lipschitz constants for gradients and
# possibly implement estimators.

from typing import Callable, Optional

import torch

from ...manifolds import Manifold
from ...manifolds import ManifoldParameter


class Problem:
    r"""Problem class encapsulating a manifold constrained optimization problem.
    
    Parameters
    ----------
    manifold:
        The manifold on which the problem is defined over.
    objective: Callable[[torch.Tensor], torch.Tensor]
        Torch autograd enabled objective function. Meaning, it returns a
        torch.Tensor on which `.backward()` can be called.
    lipschitz_const: float | None = None,
        Lipschitz constant for a function.
    
    Examples
    --------
    >> manifold = Euclidean(2)
    >> x = torch.tensor([0,1])
    >> def objective(x):
    >>     return x.pow(2).sum()
    >> problem = Problem(manifold, objective)
    """
    # TODO: Currently, the cost function holds all of the data or the dataloader,
    # and handling the devices and the datatypes may be messy. Need to fix it.
    # may need to define a cost or objective function class to hold all of that.
    def __init__(self,
        manifold: Manifold,
        objective: Callable[[torch.Tensor], torch.Tensor],
        grad_f: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        lipschitz_const: float | None = None,
        ):
        self._objective = objective
        self.manifold = manifold
        self._grad_f = grad_f
        self._lipschitz_const = lipschitz_const

    def objective(self,
        point: torch.Tensor | ManifoldParameter,
        backward_pass: bool = False,
        ) -> torch.Tensor:
        """Evaluate objective value of the problem on the `point`.
        
        If `backward_pass` is `True`, reset `point.grad` and perform backward
        pass to calculate the gradient.
        """
        # TODO: Maybe clone and detach the tensor?
        if backward_pass:
            point.requires_grad = True
            point.grad = None
            obj = self._objective(point)
            obj.backward()
            point.requires_grad = False
            return float(obj.detach())
        else:
            with torch.no_grad():
                return float(self._objective(point).detach())

    def grad(self,
             point: torch.Tensor | ManifoldParameter,
             repeat_forward: bool = True,
             **kwargs,
             ) -> torch.Tensor:
        """Calculate the gradient of the objective function at `point`
        
        Parameters
        ----------
            point: torch.Tensor
                Point of evaluation.
            repeat_forward: bool = True
                If set to true, repeats the forward pass of the objective to
                calculate the gradient with backpropagation, else returns the
                gradient stored at the tensor `point`.
            **kwargs:
                Additional arguments passed onto the custom `_grad_f` function.
        """
        if self._grad_f is None:
            if repeat_forward or point.grad is None:
                required_grad = point.requires_grad
                if required_grad is False:
                    point.requires_grad = True
                self.objective(point, backward_pass=repeat_forward)
                point.requires_grad = required_grad
            grad = point.grad
        else:
            grad = self._grad_f(point,
                                repeat_forward=repeat_forward,
                                **kwargs)
        return grad

    @property
    def lipschitz_constant(self):
        """The lipschitz constant for lipcshitz continuous gradients."""
        return self._lipschitz_const
