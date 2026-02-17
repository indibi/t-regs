"""Riemannian Alternating Descent Ascent Algorithm

Implementation is based on [1]

References
----------
..  [1] Xu, Meng, et al. "A Riemannian Alternating Descent Ascent Algorithmic Framework
    for Nonconvex-Linear Minimax Problems on Riemannian Manifolds." arXiv preprint
    arXiv:2409.19588 (2024).

Author:
Mert Indibi
1/17/2026
"""


import collections
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

import torch
import numpy as np

from ...manifolds import Manifold
from .line_searcher import LineSearcher
# from .gradient_descent import RiemmannianGradientDescent as RGD
from ...utils import printer

@dataclass
class RADA_RGD_Result:  # pylint: disable=invalid-name
    point: Any
    objective: float
    iterations: int
    stopping_criterion: str
    time: float
    function_evaluations: Optional[int] = None
    step_size: Optional[float] = None
    gradient_norm: Optional[float] = None
    dual_variable_change: Optional[float] = None
    log: Optional[Dict] = None


# class RADA_PGD:
#     pass

class RADA_RGD:   # pylint: disable=invalid-name
    r"""Riemannian Ascent Descent Algorithm with Riemannian Gradient Descent
    
    Parameters
    ----------
    line_searcher:
        Line search helper algorithm used to ensure the condition (4.2) in [1]
        is met. If not provided, defaults to the default of LineSearcher class
        initialization.
    step_size:
        If provided, this fixed step size will be used for the Riemannian
        gradient descent updates instead of performing a line search. Not
        recommended.
    max_it:
        Maximum iterations allowed for the algorithm.
    Tk:
        Number of inner iterations for the Phi_k(x) minimization task.
    beta1:
        Parameter :math:`\beta_1` used in the algorithm as described in [1].
        This is application dependent and should ideally be fine tuned.
    c_1:
        Sufficient decrease factor :math:`c_1 \in (0,1)`
    eta:
        Step size decrease factor :math:`\eta \in (0,1)` for the backtracking
        line search.
    rho:
        Dual variable proximal regularization parameter :math:`\beta_k`s 
        attenuation factor in :math:`\beta_{k+1} = \frac{\beta_1^{(k+1)}}{
        (k+1)^\rho}` with :math:`\rho>1`.
    tau_1:
        :math:`\beta_1^{(k)}` update threshold :math:`\tau_1 \in (0,1)`
    tau_2:
        :math:`\beta_1^{(k)}` attenuation factor :math:`\tau_2 \in (0,1)`
    eps:
        :math:`\epsilon`-Stationarity condition for the problem.
    R:
        Should be set to :math:`\mathrm{max}_{y \in dom(h)} \|y\|`
    zeta_max:
        Upper step size limit for BB step size.
    zeta_min:
        Lower limit for BB step size
    min_gradient_norm:
        Algorithm stopping criterion based on the minimum gradient norm.
    max_time:
        Maximum allowed time for the algorithm to run in seconds.
    max_function_evals:
        Maximum function evaluations the algorithm is allowed to perform
    min_step_size:
        Algorithm termination criteria based on the latest step size taken.
    verbosity:
        Verbosity level of the algorithm.
    log_verbosity:
        Verbosity level for logging.
    report_period:
        Period for reporting progress.
    logging_period:
        Period for logging details.
    """
    def __init__(self,
                 line_searcher: LineSearcher | None = None,
                 step_size: float | None = None,
                 max_it: int = 1000,
                 Tk: int = 10,  # py-lint: disable=invalid-name
                 beta1: float = 1.0,
                 c_1: float = 1e-4,
                 eta: float = 0.1,
                 rho: float = 1.5,
                 tau_1: float = 0.999,
                 tau_2: float = 0.9,
                 eps: float = 1e-6,
                 R: float = 1.0,    # py-lint: disable=invalid-name
                 # TODO: These were set arbitrarily. Find a more informed 
                 # way to choose them
                 zeta_max: float = 1e6,
                 zeta_min: float = 0,
                 min_gradient_norm : float = 1e-8,
                 max_time: float | None = None,
                 max_function_evals: int = 5000,
                 min_step_size:float = 0,
                 verbosity: int = 0,
                 log_verbosity: int = 1,
                 report_period: int = 1,
                 logging_period: int = 1,
                 ):
        # pylint: disable=invalid-name
        self.step_size = step_size
        self.line_searcher = None
        if step_size is None:
            if line_searcher is None:
                self.line_searcher = LineSearcher()
            elif isinstance(line_searcher, LineSearcher):
                self.line_searcher = line_searcher
            else:
                raise TypeError(("No step size is provided and the line "
                                 "search algorithm is not of type LineSearcher")
                )
        else:
            if step_size <= 0:
                raise ValueError(("Provided step size must be positive"))

        self.lda = eps/(2*R)
        self.nu = lambda beta_k: 2*Tk*(R**2)*beta_k
        self.beta1 = beta1
        self.Tk = Tk    # py-lint: disable=invalid-name
        self.c_1 = c_1
        self.eta = eta
        self.rho = rho
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.max_it = max_it
        self.zeta_max = zeta_max
        self.zeta_min = zeta_min
        self.min_gradient_norm = min_gradient_norm
        self.max_time = max_time
        self.max_function_evals = max_function_evals
        self.min_step_size = min_step_size
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = logging_period
        self.log = None


    def solve(self,
              func_f: Callable[[torch.Tensor], float | torch.Tensor],
              grad_f: Callable[[torch.Tensor], torch.Tensor],
              prox_h: Callable[[torch.Tensor, float], torch.Tensor],
              A: Callable[[torch.Tensor], torch.Tensor],    # py-lint: disable=invalid-name
              nabla_AT: Callable[[torch.Tensor], torch.Tensor],  # py-lint: disable=invalid-name
              manifold: Manifold,
              func_g: Callable[[torch.Tensor], float] | None = None,
              x0: torch.Tensor | None = None,
              y0: torch.Tensor | None = None,
              generator: torch.Generator = None
              ) -> RADA_RGD_Result:
        r"""Solve the optimization problem
        
        Parameters
        ----------
        func_f: 
            Smooth objective function :math:`f` to be minimized
        grad_f:
            Function evaluating the euclidean gradient of :math:`f` at a point.
        prox_h:
            Proximal operator of the function :math:`h`
        A:
            Smooth, possibly nonlinear mapping.
        nabla_AT:
            Adjoint of the Jacobian of the mapping :math:`A`. If :math:`A` is a
            linear mapping, this corresponds to the adjoint of :math:`A`.
        manifold:
            The manifold to minimize :math:`f` over
        x0:
            Initial point to start the optimization
        generator:
            Torch pseudo random generator used to initialize `x0` when it is not
            provided.
        
        Returns
        -------
            result: RADA_RGD_Result
        """
        # pylint: disable=invalid-name
        if self.verbosity >= 1:
            print("Optimizing...")
        if self.verbosity >= 2:
            iteration_format_length = int(np.log10(self.max_it)) + 1
            columns = [("Iteration", f"{iteration_format_length}d"),
                       ("f(x)", "+.12e")
                       ]
            columns += [("g(x)", "+.8e")] if func_g is not None else []
            columns += [("Phi_k(x)", "+.12e"),
                        ("Gradient norm", ".8e"),
                        ("del_k", ".8e"),
                        ]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        start_time = perf_counter()
        column_printer.print_header()

        if self.step_size is None:
            solver_params = self.line_searcher.get_parameters()
        else:
            step_size = self.step_size
            solver_params = {
                'step_size_strategy': 'constant',
                'step_size': step_size
            }

        self._initialize_log(solver_params=solver_params)

        k = 0
        if x0 is None:
            x = manifold.random_point(generator=generator)
        else:
            x = x0
        if y0 is None:
            y0 = torch.zeros_like( A(x) )

        y_k = y0
        beta1_k = self.beta1
        beta_k = beta1_k/(k+1)**self.rho
        del_k = torch.max(((self.lda + beta_k)*y_k - y0).abs())

        self.line_searcher.c_1 = self.c_1
        self.line_searcher.nu = self.nu(beta_k)/self.Tk
        self.line_searcher.init_step_size = 1.0
        Phi_k, grad_Phi_k, y_kp1 = self._initialize_sub_pgd_problem(
                                    x,
                                    func_f,
                                    grad_f,
                                    prox_h,
                                    A,
                                    nabla_AT,
                                    y_k,
                                    beta_k)

        f_x = func_f(x)
        g_x = func_g(x) if func_g is not None else None
        Phi_k_x = Phi_k(x)
        grad_Phi_k_x = grad_Phi_k(x)
        grad_Phi_k_x = -manifold.project(x, grad_Phi_k_x) # Descent direction
        grad_norm = manifold.norm(x, grad_Phi_k_x, project=False)
        row = [k, f_x]
        row += [g_x] if g_x is not None else []
        row += [Phi_k_x, grad_norm, del_k]
        column_printer.print_row(row)
        self._add_log_entry(k, x, Phi_k_x)
        func_evals = 2 # Because f(x) get evaluated again inside Phi_k(x)

        while True:
            k += 1
            x_kt = x.clone()
            line_searcher = self.line_searcher
            line_searcher.nu = self.nu(beta_k)/self.Tk

            for t in range(self.Tk):
                if self.step_size is None:
                    line_searcher.old_f_x = None
                    Phi_k_xkt = Phi_k(x_kt)
                    grad_Phi_k_xkt = grad_Phi_k(x_kt)
                    # Descent direction
                    grad_Phi_k_xkt = -manifold.project(x, grad_Phi_k_xkt)
                    grad_norm = manifold.norm(x, grad_Phi_k_xkt, project=False)

                    step_size, x_ktp1, step_count = line_searcher.search(
                                        Phi_k,
                                        manifold,
                                        x_kt,
                                        grad_Phi_k_xkt,
                                        Phi_k_xkt,
                                        -(grad_norm**2)
                                        )

                    func_evals += step_count
                    grad_Phi_k_xktp1 = grad_Phi_k(x_ktp1)
                    grad_Phi_k_xktp1 = - manifold.project(x, grad_Phi_k_xktp1)
                    # Previously `grad_Phi_k_xktp1` and `grad_Phi_k_xktp1` were set
                    # to descent direction. The minus sign below is a correction for
                    # that
                    grad_tp1_norm = manifold.norm(x_ktp1,
                                                grad_Phi_k_xktp1,
                                                project=False)
                    v_kt = -(grad_Phi_k_xktp1 - grad_Phi_k_xkt)

                    change = x_ktp1 - x_kt
                    inn_prod = v_kt.flatten().dot( change.flatten())
                    if t % 2 ==0:
                        zeta_BB_k_t = (change**2).sum() / inn_prod.abs()
                    else:
                        zeta_BB_k_t = inn_prod.abs() / (v_kt**2).sum()

                    next_step_size = max(
                        min(zeta_BB_k_t, self.zeta_max/grad_tp1_norm),
                        self.zeta_min)
                    # Here I multiplied `next_step_size` with `grad_tp1_norm`
                    # because the line searcher normalizes the step size with the
                    # gradient norm
                    line_searcher.init_step_size = next_step_size*grad_tp1_norm

                    x_kt = x_ktp1
                else:
                    Phi_k_xkt = Phi_k(x_kt)
                    grad_Phi_k_xkt = grad_Phi_k(x_kt)
                    # Descent direction
                    grad_Phi_k_xkt = -manifold.project(x, grad_Phi_k_xkt)
                    x_ktp1 = manifold.retract(x_kt, step_size*grad_Phi_k_xkt)
                    x_kt = x_ktp1

            x = x_ktp1
            Phi_k, grad_Phi_k, y_kp1 = self._initialize_sub_pgd_problem(
                                            x,
                                            func_f,
                                            grad_f,
                                            prox_h,
                                            A,
                                            nabla_AT,
                                            y_k,
                                            beta_k
                                            )

            del_kp1 = torch.max(((self.lda + beta_k)*y_kp1 - y_k).abs())

            beta1_kp1 = (self.tau_2*beta1_k if (del_kp1 >= self.tau_1*del_k)
                                            else beta1_k)
            beta_kp1 = beta1_kp1 / (k + 1)**self.rho

            y_k = y_kp1
            beta_k = beta_kp1
            beta1_k = beta1_kp1
            del_k = del_kp1

            f_x = func_f(x)
            g_x = func_g(x) if func_g is not None else None
            Phi_k_x = Phi_k(x)
            grad_Phi_k_x = grad_Phi_k(x)
            # Descent direction
            grad_Phi_k_x = -manifold.project(x, grad_Phi_k_x)
            grad_norm = manifold.norm(x, grad_Phi_k_x, project=False)
            row = [k, f_x]
            row += [g_x] if g_x is not None else []
            row += [Phi_k_x, grad_norm, del_k]
            column_printer.print_row(row)
            func_evals += 2 # Because f(x) get evaluated again in Phi_k(x)
            self._add_log_entry(k, x, Phi_k_x)

            stopping_criterion = self._check_stopping_criteria(start_time,
                                                               k,
                                                               grad_norm,
                                                               step_size,
                                                               func_evals)
            if stopping_criterion:
                if self.verbosity >=1:
                    print(stopping_criterion)
                    print("")
                break

        return self._return_result(
            start_time,
            point = x,
            objective = Phi_k_x,
            iterations = k,
            stopping_criterion = stopping_criterion,
            function_evaluations = func_evals,
            step_size = step_size,
            gradient_norm = grad_norm,
            dual_variable_change = del_k
        )


    def _initialize_sub_pgd_problem(self,
                                    x,
                                    func_f,
                                    grad_f,
                                    prox_h,
                                    A,
                                    nabla_AT,
                                    y_k,
                                    beta_k):
        z = (A(x) + beta_k * y_k)/(self.lda + beta_k)
        y_kp1 =prox_h( z, 1/(self.lda + beta_k) )
        Phi_k = lambda x: (func_f(x) 
                           + 0.5*((A(x) - y_k)**2).sum()/(self.lda + beta_k)
                           - 0.5*(self.lda+beta_k)*((y_kp1 - y_k)**2).sum()
                           - 0.5*beta_k*(y_k**2).sum()
                           )
        grad_Phi_k = lambda x: grad_f(x) + nabla_AT(y_kp1)
        return Phi_k, grad_Phi_k, y_kp1

    def _return_result(self, start_time, **kwargs) -> RADA_RGD_Result:
        return RADA_RGD_Result(
            time=perf_counter() - start_time,
            log=self.log,
            **kwargs
        )

    def _check_stopping_criteria(self,
                             start_time,
                             iteration,
                             gradient_norm,
                             step_size,
                             function_evaluations):
        run_time = perf_counter() - start_time
        reason = None
        if run_time >= self.max_time:
            reason = f"Terminated - max time reached after {iteration} iterations."
        elif iteration>= self.max_it:
            reason = ("Terminated - maximum number of iterations reached after "
                      f"{run_time:.3f} seconds.")
        elif gradient_norm <= self.min_gradient_norm:
            reason = (
                f"Terminated - min grad norm reached after {iteration} "
                f"iterations, {run_time:.3f} seconds."
            )
        elif (step_size < self.min_step_size) or (step_size ==0):
            reason = (
                f"Terminated - min step_size reached after {iteration} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif function_evaluations >= self.max_function_evals:
            reason = (
                "Terminated - max cost evals reached after "
                f"{run_time:.2f} seconds."
            )
        return reason


    def _initialize_log(self, *, solver_params=None):
        self.log = {
            'solver': str(self),
            'stopping_criteria': {
                'max_time': self.max_time,
                'max_it': self.max_it,
                'min_gradient_norm': self.min_gradient_norm,
                'max_function_evals': self.max_function_evals,
                'min_step_size': self.min_step_size
                },
            'solver_params': solver_params,
            'iterations': collections.defaultdict(list)
            }

    def _add_log_entry(self, iteration, point, objective, **kwargs):
        if self.log_verbosity <=0:
            return
        if (self.logging_period !=0) and (iteration % self.logging_period ==0):
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['time'].append(perf_counter())
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)

            if self.log_verbosity >1:
                self.log['iterations']['point'].append(point)


    def __str__(self):
        if self.line_searcher is None:
            name = type(self).__name__ + " with fixed step size"
        else:
            name = type(self).__name__ + (
                f" with {self.line_searcher.step_size_strategy} step size")
        return name

    def get_parameters(self) -> dict:
        params = {
            'step_size': self.step_size,
            'max_it': self.max_it,
            'min_gradient_norm': self.min_gradient_norm,
            'max_time': self.max_time,
            'max_function_evals': self.max_function_evals,
            'min_step_size': self.min_step_size,
            'verbosity': self.verbosity,
            'log_verbosity': self.log_verbosity,
            'report_period': self.report_period,
            'logging_period': self.logging_period
        }
        if self.line_searcher is not None:
            params['line_searcher'] = self.line_searcher.get_parameters()
        return params
