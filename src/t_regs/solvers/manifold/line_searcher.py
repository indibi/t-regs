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

from typing import Callable

import torch

from ...manifolds import Manifold

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
                 c_1: float = 0.85,
                 max_it: int = 25,
                 init_step_size: float = 1,
                 nu: float = 0.0,
                 optimism_factor: float = 2.0,
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

        self.old_f_x = None
        self.old_tau = None

        self._search = getattr(self, f'_{step_size_strategy}')

    def get_parameters(self) -> dict:
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

        if self.old_f_x is not None:
            t = 2* (f_x - self.old_f_x) / df_x_eta
            t *= self.optimism_factor
        else:
            t = self.init_step_size / norm_eta

        x_new = manifold.retract(x, t*eta)
        f_x_new = func_f(x_new)

        step_count = 1
        while (
            f_x_new > f_x + self.c_1 * t * df_x_eta + self.nu
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
        self.old_f_x = f_x
        return step_size, x_new, step_count


    def _adaptive(self, func_f, manifold, x, eta, f_x, df_x_eta):
        raise NotImplementedError("Adaptive Line Search is not implemented yet.")


    def search(self,
               func_f: Callable[[torch.Tensor], float],
               manifold: Manifold,
               x: torch.Tensor,
               eta: torch.Tensor,
               f_x: float,
               df_x_eta: float):
        r"""Perform line search
        
        Parameters
        ----------
        func_f:
            Objective function to minimize
        manifold:
            Manifold that the objective function is being minimized over
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
        return self._search(func_f, manifold, x, eta, f_x, df_x_eta)
