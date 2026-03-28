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

from typing import Callable, Optional

import torch

from ...manifolds import Manifold
from .parameter import ManifoldParameter


class Problem:
    r"""Problem class encapsulating a manifold constrained optimization problem.
    
    Parameters
    ----------
        manifold:
            The manifold on which the problem is defined over.
        objective: Callable[[torch.Tensor], torch.Tensor]
            Torch autograd enabled objective function. Meaning, it returns a
            torch.Tensor on which `.backward()` can be called.
        init_point: Optional[torch.Tensor] = None
            Initial point for the parameter.
    """
    # TODO: Currently, the cost function holds all of the data or the dataloader,
    # and handling the devices and the datatypes may be messy. Need to fix it.
    # may need to define a cost or objective function class to hold all of that.
    def __init__(self,
        manifold: Manifold,
        objective: Callable[[torch.Tensor], torch.Tensor],
        grad_f: Optional[Callable[[torch.Tensor], torch.Tensor]] =None,
        ):
        self._objective = objective
        self.manifold = manifold
        self._grad_f = grad_f

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
            return obj
        else:
            with torch.no_grad():
                return self._objective(point)
    
    def grad(self,
             point: torch.Tensor | ManifoldParameter,
             repeat_forward: bool = True,
             ) -> torch.Tensor:
        """Calculate the gradient of the objective function at `point`
        
        Parameters
        ----------
            point: torch.Tensor
                Point of evaluation.
            repeat_forward: bool = False
                If set to true, repeats the forward pass of the objective to
                calculate the gradient with backpropagation, else returns the
                gradient stored at the tensor `point`.
        """
        if self._grad_f is None:
            if repeat_forward or point.grad is None:
                self.objective(point, backward_pass=repeat_forward)
            grad = point.grad
        else:
            grad = self._grad_f(point)
        return grad