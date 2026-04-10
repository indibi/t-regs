"""
Docstring for t_regs.solvers.manifold.line_searcher

The implementation follows closely to the implementation of `Pymanopt` Townsend
et. al. (2016)

References
----------
..  Townsend, J., Koep, N., & Weichwald, S. (2016). Pymanopt: A python toolbox 
    for optimization on manifolds using automatic differentiation. Journal of 
    Machine Learning Research, 17(137), 1-5.
"""

from typing import Tuple, Optional
from functools import partial
from dataclasses import dataclass

import torch

from .problem import Problem
# from ...manifolds import Manifold

class LineSearcher:
    r"""Back-tracking line search algorithm for Riemannian Gradient Descent

    Parameters
    ----------
    step_size_strategy: str = 'backtracking'
        Step size selection strategy. Possible options are 'backtracking',
        'constant', 'adaptive'.
    tau: float = 0.5
        The attenuation factor for the step size search :math:`\tau \in (0,1)`
    c_1: float = 0.01
        The sufficient decrease factor :math:`c_1 \in (0,1)` as described in
        Algorithm 2 of [1].
    max_it: int = 25
        The maximum iterations the line search is allowed to run for.
    init_step_size: float = 1,
        Initial step size for the search.
    nu: float = 0
        Additional constant for the sufficient decrease condition to ensure
        convergence of nested optimization algorithms.
    optimism_factor: float = 2
        When searching with backtracking strategy, the step size is chosen to 
        initially search a little further than the initial guess based on previous
        call to the search.

    References
    ----------
    ..  [1] Boumal, N., Absil, P. A., & Cartis, C. (2019). Global rates of
        convergence for nonconvex optimization on manifolds. IMA Journal of 
        Numerical Analysis, 39(1), 1-33.
    """
    strategies = ['backtracking', 'adaptive']
    def __init__(self,
                 step_size_strategy: str = 'backtracking',
                 tau: float = 0.5,
                 c_1: float = 1e-2,
                 max_it: int = 25,
                 init_step_size: float = 1,
                 nu: float = 0.0,
                 optimism_factor: float = 2.0,
                 retain_old_f_x: bool = False,
    ):
        if tau >=1 or tau <=0:
            raise ValueError("Attenuation factor tau must be in (0,1)")
        if c_1 >=1 or c_1 <=0:
            raise ValueError("The sufficient decrease factor must be in (0,1)")
        if step_size_strategy not in self.strategies:
            raise ValueError(("Invalid choice of step size strategy for line "
                              f"searcher ({step_size_strategy}). "
                              f"Possible options are among {self.strategies}")
                            )
        self.step_size_strategy = step_size_strategy
        self.tau = tau
        self.c_1 = c_1
        self.init_step_size = init_step_size
        self.nu = nu
        self.max_it = max_it
        self.optimism_factor = optimism_factor
        self.retain_old_f_x = retain_old_f_x

        self.old_f_x = None
        self.old_tau = None

        self._search = getattr(self, f'_{step_size_strategy}')

    def get_parameters(self) -> dict: # pylint: disable=missing-function-docstring
        return {
            'step_size_strategy': self.step_size_strategy,
            'c_1': self.c_1,
            'nu': self.nu,
            'tau': self.tau,
            'max_it': self.max_it,
            'init_step_size': self.init_step_size,
            'optimism_factor': self.optimism_factor
        }


    def _backtracking(self, func_f, manifold, x, eta, f_x, df_x_eta):
        norm_eta = manifold.norm(x, eta)

        if (self.old_f_x is not None) and self.retain_old_f_x:
            t = 2* (f_x - self.old_f_x) / df_x_eta
            t *= self.optimism_factor
        else:
            t = self.init_step_size / norm_eta

        x_new = manifold.retract(x, t*eta)
        f_x_new = func_f(x_new)

        step_count = 1
        while (
            f_x_new > f_x - self.c_1 * t * df_x_eta + self.nu
            and step_count <= self.max_it
        ):
            t = self.tau * t

            x_new = manifold.retract(x, t*eta)
            f_x_new = func_f(x_new)

            step_count +=1

        if f_x_new > f_x:
            t = 0
            x_new = x

        step_size = t * norm_eta
        if self.retain_old_f_x:
            self.old_f_x = f_x
        return step_size, x_new, step_count


    def _adaptive(self, func_f, manifold, x, eta, f_x, df_x_eta):
        raise NotImplementedError("Adaptive Line Search is not implemented yet.")


    def search(self,
               problem: Problem,
               x: torch.Tensor,
               eta: torch.Tensor,
               f_x: float,
               df_x_eta: float):
        r"""Perform line search
        
        Parameters
        ----------
        problem:
            Problem object representing the manifold constrained optimization
            problem.
        x:
            Point on the manifold defining the tangent space
        eta:
            The descent direction tangent to the manifold at point :math:`x`
        f_x:
            The value of the function at the point :math:`x`
        df_x_eta:
            The riemannian directional derivative :math:`\eta`, i.e. :math:`
            \mathbf{D}f(x)[\eta] = \langle \mathrm{grad} f(x), \eta \rangle`.
        
        Returns
        -------
        step_size: float
            Norm of the vector retracted to reach the new point :math:`x_{new}`
        new_x:
            Next point in the iteration.
        """
        manifold = problem.manifold
        func_f = partial(problem.objective, backward_pass=False)
        return self._search(func_f, manifold, x, eta, f_x, df_x_eta)

@dataclass
class ArmijoPointResult:
    x_new: torch.Tensor
    f_x_new: float
    new_grad_f_x: torch.Tensor
    step_size: float
    step_count: int

# TODO: Retaining the step size from the previous call seems to be hurting 
# the algoritm especially in BCD tucker regression. It looks like it's better
# to just re-start the point search.
class ArmijoPointSearch:
    r"""Armijo Point Searcher

    For a cost function for a cost function :math:`f:\mathcal{M}\to\mathbb{R}`
    defined on a manifold :math:`\mathcal{M}`; find the Armijo
    point η^A = t^A η = β^m α η, of a tangent :math:`\eta \in T_x \mathcal{M}`
    where m is the smallest nonnegative integer satisfying,
    .. math::
        f(x) - f(R_x(β^m α η)) ≥ -σ 〈 \mathrm{grad} f(x), η 〉_x
    
    where β, σ ∈ (0,1) are the attenuation parameter and sufficient decrease
    parameter respectively and α > 0 is initial step size.

    Parameters
    ----------
    beta: float
        Attenuation coefficient β ∈ (0,1)
    alpha: float
        Initial step size guess.
    sigma: float
        sufficient_decrease coefficient.
    nu: float = 0
        Additional constant for the sufficient decrease condition to ensure
        convergence of nested optimization algorithms.
    optimism_factor: float = 2.0
        Optimism factor used to try larger step size when the initial step size
        is guessed from previous search calls.
    max_it: int = 25
        Maximum number of iterations for the search.
    retain_old_f_x: bool = False
        Retain the function value of the previous function evaluation to guess
        initial step size.
    retain_old_step_size: bool = True
        Retain the step size value of the previous line search to adjust the
        initial step size.

    References
    ----------
    ..  [1] Absil, P-A., Robert Mahony, and Rodolphe Sepulchre. Optimization 
        algorithms on matrix manifolds. Princeton University Press, 2008.
        pp. 62-68.
    """
    def __init__(
        self,
        beta: float = 0.5,
        alpha: float = 1.0,
        sigma: float = 0.01,
        nu: float = 0,
        optimism_factor: float = 2.0,
        max_it: int = 25,
        retain_old_f_x: bool = False,
        retain_old_step_size: bool = True,
        ):
        if beta >=1 or beta <=0:
            raise ValueError("Attenuation factor β must be in (0,1)")
        if sigma >=1 or sigma <=0:
            raise ValueError("The sufficient decrease factor σ must be in (0,1)")

        self.beta = beta
        self.sigma = sigma
        self.alpha = alpha
        self.nu = nu
        self.max_it = max_it
        self.optimism_factor = optimism_factor
        self.retain_old_f_x = retain_old_f_x
        self.retain_old_step_size = retain_old_step_size
        self.old_f_x = None
        self.old_alpha = alpha
        self.step_size_strategy = 'armijo'


    def search(self,
               problem: Problem,
               x: torch.Tensor,
               eta: torch.Tensor = None,
               f_x: Optional[float] = None,
               grad_f_x: Optional[torch.Tensor] = None,
               **kwargs
               ) -> Tuple[torch.Tensor, float, int]:
        r"""Search Armijo point.
        
        Parameters
        ----------
        problem: Problem
            Smooth optimization problem on manifold.
        x: torch.Tensor
            Point :math:`x \in \mathcal{M}`
        eta: torch.Tensor
            Descent direction in the tangent space of the manifold at `x`.
        f_x: Optional[float]
            Function value at `x`.
        grad_f_x: Optional[torch.Tensor]
            Riemannian gradient of :math:`f` at point `x`.
        
        Returns
        -------
        result: ArmijoPointResult
            
        """
        # eta_mag = problem.manifold.norm(x, eta)
        repeat_forward = True
        if f_x is None:
            f_x = problem.objective(x, backward_pass=True)
            repeat_forward = False
        if grad_f_x is None:
            grad_f_x = problem.grad(x, repeat_forward=repeat_forward)
            grad_f_x = problem.manifold.project(x, grad_f_x)
        if eta is None:
            eta = -grad_f_x

        # Directional derivative at `x` along `eta`
        d_f_x_eta = problem.manifold.inner_product(x, grad_f_x, eta)
        if self.retain_old_step_size:
            t = self.optimism_factor*self.old_alpha
        elif self.retain_old_f_x and (self.old_f_x is not None):
            t = 2*(f_x - self.old_f_x)/ d_f_x_eta
            t *= self.optimism_factor
        else:
            t = self.alpha

        x_new = problem.manifold.retract(x, t*eta)
        f_x_new = problem.objective(x_new)
        step_count = 1
        while (
            (f_x - f_x_new) < (- self.sigma*t*d_f_x_eta + self.nu)
            and step_count <= self.max_it
            ):
            t = t * self.beta
            x_new = problem.manifold.retract(x, t*eta)

            f_x_new = problem.objective(x_new)
            step_count +=1

        if f_x_new > f_x:
            step_size = 0
            x_new = x
            f_x_new = f_x
            new_grad_f_x = grad_f_x
        else:
            step_size = t
            new_grad_f_x = problem.grad(x_new)
            new_grad_f_x = problem.manifold.project(x_new, new_grad_f_x)

        self.old_f_x = f_x
        self.old_alpha = t

        return ArmijoPointResult(
            x_new = x_new,
            f_x_new = f_x_new,
            new_grad_f_x = new_grad_f_x,
            step_size = step_size,
            step_count = step_count,
            )

    @property
    def init_step_size(self):
        """Initial step size for the search."""
        return self.alpha
    
    @init_step_size.setter
    def init_step_size(self, step_size):
        self.alpha = step_size
        self.old_alpha = step_size

    def get_parameters(self) -> dict: # pylint: disable=missing-function-docstring
        return {
            'beta': self.beta,
            'sigma': self.sigma,
            'alpha': self.alpha,
            'nu': self.nu,
            'max_it': self.max_it,
            'optimism_factor': self.optimism_factor,
            'retain_old_f_x': self.retain_old_f_x,
            'retain_old_step_size': self.retain_old_step_size,
            }

    def __str__(self,) -> str:
        return type(self).__name__
