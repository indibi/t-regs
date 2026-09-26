"""Module defining a Minimax Problem class for manifold optimization.

Many optimization problems whose objective functions are comprised of smooth + 
non-smooth proper convex terms can be turned into Minimax problems. 
"""

from typing import Callable, Optional, Union
from dataclasses import dataclass

import torch

from ...manifolds import Manifold
from ...manifolds import ManifoldParameter

@dataclass
class MinimaxProblemPoint:
    r"""Minimax Problem Point.

    Attributes
    ----------
    x: torch.Tensor
        Minimization variable.
    y: torch.Tensor
        Maximization variable
    objective: Optional[torch.Tensor | float]
        Objective value of the minimax problem at :math:`(x,y)` i.e. 
        :math:`F(x,y)`.
    f_x: Optional[torch.Tensor | float]
        Value of the function :math:`f` at :math:`x`, i.e. :math:`f(x)`.
    h_y: Optional[torch.Tensor | float]
        Value of the function :math:`h` at :math:`y`, i.e. :math:`f(y)`.
    in_prod: Optional[torch.Tensor | float]
        Value of the inner product :math:`\langle A(x), y \rangle`.
    grad_f: Optional[torch.Tensor]
        Euclidean gradient of :math:`f` at :math:`x`. i.e. :math:`\nabla f(x)`.
    """
    x: torch.Tensor
    y: torch.Tensor
    objective: Optional[float] = None
    f_x: Optional[torch.Tensor | float] = None
    h_y: Optional[torch.Tensor | float] = None
    in_prod: Optional[torch.Tensor | float] = None
    grad_f: Optional[torch.Tensor] = None
    grad_F_x: Optional[torch.Tensor] = None



class MinimaxProblem:
    r"""Class representing Riemannian nonconvex-linear (NC-L) minimax problems

    The problems have the form as in [1]
    .. math::
        \min_{x∈\mathcal{M}} \max_{y∈E} \{ F(x,y):= f(x) + <A(x), y> - h(y)\}

    where,
        - :math:`\mathcal{M}` is a riemannian manifold embedded in a finite 
            dimensional euclidean space :math:`\mathbb{E}_1`.
        - E is a finite dimensional euclidean space.
        - :math:`f: \mathbb{E}_1 \to (-\infty, +\infty]` is a continuously 
            differentiable function.
        - :math:`A: \mathbb{E}_1 \to \mathbb{E}_2` is a smooth mapping.
        - :math:`h: \mathbb{E}_2 \to (-\infty, +\infty]` is a proper closed 
            convex function with a compact domain.

    Parameters
    ----------
    manifold: Manifold
        Manifold object representing the riemannian manifold for variable `x`
    func_f: Callable[Union[torch.Tensor,ManifoldParameter], torch.Tensor]
        Smooth function :math:`f`. Should be compatible with `torch.autograd`
    func_h: Callable[[torch.Tensor], torch.Tensor]
        A proper closed convex function.
    prox_h: Callable[[torch.Tensor],torch.Tensor]
        Proximal operator of the function :math:`h`.
    mapping_A: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]
        Smooth mapping from :math:`x` to :math:`y`. Two ways to specify,
            1. Can be a python function taking and returning a torch tensor.
            2. Can be a matrix :math:`A` if the mapping is :math:`A(x) = A x`,
                i.e. linear and represented with the matrix :math:`A`.
    nabla_AT: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]
        Adjoint of the Jacobian of the mapping :math:`A`. If :math:`A` is a
        linear mapping, this corresponds to the adjoint of :math:`A`.
    grad_f: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        Function evaluating the gradient of the function :math:`f` at point
        :math:`x`. If not specified, it defaults to using `torch.autograd`

    References
    ----------
    ..  [1] Xu, Meng, et al. "A Riemannian Alternating Descent Ascent 
        Algorithmic Framework for Nonconvex-Linear Minimax Problems on 
        Riemannian Manifolds." arXiv preprint arXiv:2409.19588 (2024).
    """

    def __init__(self,
        manifold: Manifold,
        func_f: Callable[[torch.Tensor], torch.Tensor],
        func_h: Callable[[torch.Tensor], torch.Tensor],
        prox_h: Callable[[torch.Tensor], torch.Tensor],
        mapping_A: Union[Callable[[torch.Tensor],torch.Tensor],torch.Tensor], # pylint: disable=invalid-name
        func_g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        nabla_AT: Optional[ # pylint: disable=invalid-name
            Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]
            ] = None,
        grad_f: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ):
        self.manifold = manifold
        self._func_f = func_f
        self._func_h = func_h
        self._prox_h = prox_h
        self._mapping_A = mapping_A # pylint: disable=invalid-name
        self._grad_f = grad_f
        self._func_g = func_g

        if nabla_AT is None:
            if isinstance(mapping_A, torch.Tensor): # pylint: disable=invalid-name
                if mapping_A.ndim == 2:
                    self._nabla_AT = mapping_A.T # pylint: disable=invalid-name
                else:
                    raise ValueError((
                        "Linear mapping `mapping_A` is not a matrix and the "
                        "transposition for calculating the adjoint of the "
                        "jacobian is ambiguous.")
                        )
            elif isinstance(mapping_A, Callable):
                # TODO: Implement jacobian using autograd.
                raise NotImplementedError((
                    "Adjoint of the jacobian from mapping using torch.autograd "
                    "is not yet implemented"
                    ))
            else:
                raise TypeError("mapping_A is not a callable or a torch.Tensor")
        elif isinstance(nabla_AT, (torch.Tensor, Callable)):
            self._nabla_AT = nabla_AT   # pylint: disable=invalid-name
        else:
            raise TypeError(
                "Unrecognized nabla_AT when initiating Minimax problem." 
                )


    def func_f(self,
            x: torch.Tensor,
            backward_pass: bool = False,
            ) -> torch.Tensor:
        r"""Evaluate the value of the function :math:`f` on the point `x`.
        
        If `backward_pass` is `True`, reset `point.grad` and perform backward
        pass to calculate the gradient.
        """
        if backward_pass:
            x.requires_grad = True
            x.grad = None
            f_x = self._func_f(x)
            f_x.backward()
            x.requires_grad = False
            return float(f_x.detach())
        else:
            with torch.no_grad():
                return float(self._func_f(x).detach())

    def grad_f(self,
            x: Union[torch.Tensor, ManifoldParameter],
            repeat_forward: bool= True,
            **kwargs,
            ) -> torch.Tensor:
        """Calculate the euclidean gradient of the function :math:`f` at `x`.
        
        Parameters
        ----------
        x: Union[torch.Tensor, ManifoldParameter],
            Point of evaluation for function :math:`f`.
        repeat_forward: bool = True
            If set to `True`, repeats the forward pass of the objective to
            calculate the gradient with backpropagation, else returns the
            gradient stored at the tensor `x`.
        """
        if self._grad_f is None:
            if repeat_forward or x.grad is None:
                required_grad = x.requires_grad
                if required_grad is False:
                    x.requires_grad = True
                f_x = self.func_f(x, backward_pass=True)
            grad = x.grad
        else:
            grad = self._grad_f(
                x,
                repeat_forward=repeat_forward,
                **kwargs)
        return grad

    def func_h(self, y:torch.Tensor, *vargs, **kwargs) -> torch.Tensor:
        """Evaluate :math:`h` at :math:`y`."""
        return self._func_h(y, *vargs, **kwargs)

    def inner_product(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor: 
        """Calculate :math:`<A(x),y>`"""
        return (self.A(x)*y).sum()

    def A(self, x:torch.Tensor) -> torch.Tensor:    # pylint: disable=invalid-name
        r"""Map the point :math:`x` to :math:`y=A(x)`."""
        if isinstance(self._mapping_A, torch.Tensor):
            return self._mapping_A @ x
        else:
            return self._mapping_A(x)

    @torch.no_grad()
    def func_g(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate non-smooth g(x)"""
        if self._func_g is not None:
            # Ax = self.A(x)
            # return self._func_g(Ax)
            return self._func_g(x)
        else:
            raise AttributeError(
                "Minimax problem was not initialized with a `func_g`."
                )

    @torch.no_grad()
    def func_F(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate f(x) + g(x)"""
        fx = self.func_f(x, backward_pass = False)
        gx = self.func_g(x)
        return fx + gx

    def nabla_AT(self, x:Optional[torch.Tensor]=None) -> torch.Tensor: # pylint: disable=invalid-name
        r"""The adjoint of the Jacobian of the mapping :math:`A` at :math:`x`.
        """
        if isinstance(self._nabla_AT, torch.Tensor):
            return self._nabla_AT
        elif isinstance(self._nabla_AT, Callable):
            return self._nabla_AT(x)
        else:
            raise NotImplementedError(("The jacobian calculation using torch "
                                       "autograd is not yet implemented."))

    def prox_h(self, y, *vargs, **kwargs) -> torch.Tensor:
        """Evaluate the proximal operator of :math:`h` at :math:`y`."""
        return self._prox_h(y, *vargs, **kwargs)


    @torch.no_grad()
    def objective(self,
        x: torch.Tensor,
        y: torch.Tensor) -> torch.Tensor:
        """Evaluate :math:`F(x,y):= f(x) + <A(x),y> - h(y)`"""
        f_x = self.func_f(x, backward_pass=False)
        Ax = self.A(x)  # pylint: disable=invalid-name
        in_prod = (Ax*y).sum()
        hy = self.func_h(y)
        return f_x + in_prod - hy
