"""Riemannian Alternating Descent Ascent Algorithm

Implementation is based on [1]

References
----------
..  [1] Xu, Meng, et al. "A Riemannian Alternating Descent Ascent Algorithmic
    Framework for Nonconvex-Linear Minimax Problems on Riemannian Manifolds."
    arXiv preprint arXiv:2409.19588 (2024).

Author:
Mert Indibi
1/17/2026
"""


import collections
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Optional
import math

import torch

from ...manifolds import Manifold
from .problem import Problem
from .minimax_problem import MinimaxProblemPoint, MinimaxProblem
from .line_searcher import ArmijoPointSearch

from ...utils import printer

@dataclass
class RADA_RGD_Result:  # pylint: disable=invalid-name
    point: MinimaxProblemPoint
    phi_k_x: float
    iterations: int
    stopping_criterion: str
    time: float
    function_evaluations: Optional[int] = None
    step_size: Optional[float] = None
    gradient_norm: Optional[float] = None
    dual_variable_change: Optional[float] = None
    log: Optional[Dict] = None


class RADA_RGD:   # pylint: disable=invalid-name
    r"""Riemannian Ascent Descent Algorithm with Riemannian Gradient Descent

    Solves the following minimax problem defined in [1],
    .. math::
        \min_{x∈\mathcal{M}} \max_{y∈E} \{ F(x,y):= f(x) + <A(x), y> - h(y)\}

    Parameters
    ----------
    R:
        Should be set to :math:`\mathrm{max}_{y \in dom(h)} \|y\|`
    line_searcher: ArmijoPointSearch
        Line search helper algorithm used to ensure the condition (4.2) in [1]
        is met. If not provided, defaults to :class:`ArmijoPointSearch` with
        it's default initialization setting.
    step_size: Optional[float]
        If provided, this fixed step size will be used for the Riemannian
        gradient descent updates instead of performing a line search.
    max_it:
        Maximum iterations allowed for the algorithm.
    Tk:
        Number of inner iterations for the Phi_k(x) minimization task.
    beta1:
        Parameter :math:`\beta_1` used in the algorithm as described in [1].
        This is application dependent and should ideally be fine tuned.
    rho:
        Dual variable proximal regularization parameter :math:`\beta_k`s 
        attenuation factor in :math:`\beta_{k+1} = \frac{\beta_1^{(k+1)}}{
        (k+1)^\rho}` with :math:`\rho>1`.
    tau_1:
        :math:`\beta_1^{(k)}` update threshold :math:`\tau_1 \in (0,1)`
    tau_2:
        :math:`\beta_1^{(k)}` attenuation factor :math:`\tau_2 \in (0,1)`
    alpha_BB_max:
        Upper step size limit for BB step size.
    alpha_BB_min:
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
    
    References
    ----------
    ..  [1] Xu, Meng, et al. "A Riemannian Alternating Descent Ascent 
        Algorithmic Framework for Nonconvex-Linear Minimax Problems on 
        Riemannian Manifolds." arXiv preprint arXiv:2409.19588 (2024).
    """
    def __init__(self,
        R: float,    # py-lint: disable=invalid-name
        line_searcher: ArmijoPointSearch | None = None,
        step_size: float | None = None,
        max_it: int = 1000,
        Tk: int = 10,  # py-lint: disable=invalid-name
        beta1: float = 1.0,
        rho: float = 1.5,
        tau_1: float = 0.999,
        tau_2: float = 0.9,
        alpha_BB_max: float = 1e6,
        alpha_BB_min: float = 1e-8,
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
        self.line_searcher = line_searcher
        if step_size is None:
            if line_searcher is None:
                self.line_searcher = ArmijoPointSearch(
                    retain_old_step_size=False,
                )
            elif isinstance(line_searcher, ArmijoPointSearch):
                self.line_searcher = line_searcher
            else:
                raise TypeError((
                    "No step size is provided and the line search algorithm "
                    "is not of type `ArmijoPointSearch`"
                    ))
        else:
            if step_size <= 0:
                raise ValueError(("Provided step size must be positive"))
        eps = min_gradient_norm
        self.lda = eps/(2*R)
        self.nu = lambda beta_k: 2*Tk*(R**2)*beta_k
        self.beta1 = beta1
        self.Tk = Tk    # py-lint: disable=invalid-name
        self.rho = rho
        self.R = R
        self.eps = eps
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.max_it = max_it
        self.alpha_BB_max = alpha_BB_max
        self.alpha_BB_min = alpha_BB_min
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
        problem: MinimaxProblem,
        x0: Optional[torch.Tensor] = None,
        y0: Optional[torch.Tensor] = None,
        init_point: Optional[MinimaxProblemPoint] = None,
        seed: Optional[int | torch.Generator] = None,
        ) -> RADA_RGD_Result:
        r"""Solve the optimization problem

        Parameters
        ----------
        problem: :class:`MinimaxProblem`
            MinimaxProblem class defining the manifold optimization problem.
        init_point: Optional[MinimaxProblemPoint]
            Inital point to start the RADA optimization. Defaults to random
            initialization if not provided.
        x0:
            Initial :math:`x` value to begin the optimization. Defaults to
            random initialization with `seed`.
        y0:
            Initial :math:`y` value to begin the optimization. Defaults to
            random initialization with `seed`.
        seed:
            Integer or :class:`torch.Generator`. Defaults to None.

        Returns
        -------
            result: RADA_RGD_Result
        """
        # pylint: disable=invalid-name
        start_time = perf_counter()
        manifold: Manifold = problem.manifold
        line_searcher: ArmijoPointSearch = self.line_searcher
        column_printer = self._initialize_column_printer()
        column_printer.print_header()

        solver_params = self.get_parameters()
        if self.step_size is None:
            line_search_params = line_searcher.get_parameters()
        else:
            step_size = self.step_size
            line_search_params = {
                'step_size_strategy': 'constant',
                'step_size': step_size
            }
        solver_params['line_search'] = line_search_params
        self._initialize_log(solver_params=solver_params)

        k = 0
        t = 0
        ss = 1.0 if self.step_size is None else self.step_size
        # Initialize the point.
        if init_point is not None:
            x0 = init_point.x
            y0 = init_point.y

        if x0 is None:
            x = manifold.random_point() # TODO: Deal with the seed
            x.requires_grad=True
        else:
            x = x0
        if y0 is None:
            with torch.no_grad():
                y0 = problem.A(x)
                y0 = problem.prox_h(y0, 1/(self.lda+self.beta1))

        y_k = y0
        beta1_k = self.beta1
        beta_k = beta1_k/(k+1)**self.rho
        del_k = torch.max(((self.lda + beta_k)*y_k - y0).abs())

        nu_k = self.nu(beta_k)/self.Tk
        self.line_searcher.init_step_size = 1.0
        phi_k_problem = self._initialize_Phi_k_problem(
            x_k = x,
            y_k = y_k,
            mnmx_problem = problem,
            beta_k = beta_k,
            )

        with torch.no_grad():
            f_x = problem.func_f(x, no_grad=True)
            g_x = problem.inner_product(x, y_k)
        phi_k_x = phi_k_problem.objective(x, backward_pass=False)#True)
        grad_phi_k_x = phi_k_problem.grad(x, repeat_forward=True)#False)
        with torch.no_grad():
            grad_phi_k_x = manifold.project(x, grad_phi_k_x)
            desc_dir = -grad_phi_k_x # Descent direction
            grad_norm = manifold.norm(x, grad_phi_k_x, project=False)

        row = [k, t, f_x, g_x, phi_k_x, grad_norm, del_k, beta_k, nu_k, ss]
        # ("f(x_kt)", "+.12e"),
        # ("<A(x_kt), y_k>", "+.8e"),
        # ("Φ_k(x_kt)", "+.12e"),
        # ("||grad Φ_k(x_kt)||", ".6e"),
        # ("δ_k", ".4e"),
        # ("β_k", ".4e"),
        # ("ν_k", ".4e"),
        # ("α_kt+1", ".6e")
        # ("α_kt (Armijo)", ".6e"),
        column_printer.print_row(row)
        self._add_log_entry(k, x, phi_k_x)
        func_evals = 3 # Because f(x) get evaluated again inside Phi_k(x) & func_f
        x_kt = x.clone().detach()
        while True:
            k += 1
            for t in range(self.Tk):
                if self.step_size is None:
                    # If the step size is determined with line search and
                    # Barzilai-Borwein scheme.
                    with torch.no_grad():
                        f_x = problem.func_f(x_kt, no_grad=True)
                        g_x = problem.inner_product(x_kt, y_k)
                    phi_k_xkt = phi_k_problem.objective(
                        x_kt,
                        backward_pass=False#True
                        )
                    grad_phi_k_xkt = phi_k_problem.grad(
                        x_kt,
                        repeat_forward=True#False
                        )
                    with torch.no_grad():
                        grad_phi_k_xkt = manifold.project(x_kt, grad_phi_k_xkt)
                        desc_dir = -grad_phi_k_xkt # Descent direction
                        grad_norm = manifold.norm(
                            x_kt,
                            grad_phi_k_xkt,
                            project=False
                            )

                    search_result = line_searcher.search(
                        problem=phi_k_problem,
                        x=x_kt,
                        eta=desc_dir,
                        f_x=phi_k_xkt,
                        grad_f_x=grad_phi_k_xkt,
                        nu_k=nu_k,
                        )

                    with torch.no_grad():
                        x_ktp1 = search_result.x_new
                        grad_phi_k_xktp1 = search_result.new_grad_f_x
                        func_evals += 2 +  search_result.step_count
                        v_kt = grad_phi_k_xktp1 - grad_phi_k_xkt
                        change = x_ktp1 - x_kt
                        inn_prod = (v_kt*change).sum()
                        if t % 2 ==0:
                            alpha_BB_k_t = inn_prod.abs()/(v_kt.pow(2).sum()+1e-30)
                        else:
                            alpha_BB_k_t = (change.pow(2)).sum()/(inn_prod.abs()+1e-30)

                        next_step_size = max(
                            min(alpha_BB_k_t, self.alpha_BB_max),
                            self.alpha_BB_min
                            )
                        line_searcher.init_step_size = next_step_size
                        ss = next_step_size

                        x_kt.copy_(x_ktp1)
                    row = [k, t, f_x, g_x, phi_k_x, grad_norm, del_k,
                           beta_k, nu_k,
                           next_step_size, search_result.step_size]
                    column_printer.print_row(row)
                else: # If the step size is constant
                    phi_k_xkt = phi_k_problem.objective(
                        x_kt,
                        backward_pass=False#True
                        )
                    grad_phi_k_xkt = phi_k_problem.grad(
                        x_kt,
                        repeat_forward=True#False
                        )

                    with torch.no_grad():
                        grad_phi_k_xkt = manifold.project(x_kt, grad_phi_k_xkt)
                        desc_dir = -grad_phi_k_xkt # Descent direction
                        grad_norm = manifold.norm(
                            x_kt,
                            grad_phi_k_xkt,
                            project=False
                            )
                        x_ktp1 = manifold.retract(x_kt, ss*desc_dir)
                        x_kt.copy_(x_ktp1)
                        f_x = problem.func_f(x_kt, no_grad=True)
                        g_x = problem.inner_product(x_kt, y_k)
                    func_evals +=3
                    row = [k, t, f_x, g_x, phi_k_x, grad_norm, del_k,
                           beta_k, nu_k, ss]
                    column_printer.print_row(row)


            with torch.no_grad():
                x.copy_(x_kt)
                Ax = problem.A(x)
                z = (Ax + beta_k*y_k)/(self.lda+beta_k)
                y_kp1 = problem.prox_h(z, 1/(self.lda+self.beta1))
                del_kp1 = torch.max(((self.lda + beta_k)*y_kp1 - y_k).abs())
                beta1_kp1 = (self.tau_2*beta1_k if (del_kp1 >= self.tau_1*del_k)
                                                else beta1_k)
                beta_kp1 = beta1_kp1 / (k + 1)**self.rho

            y_k = y_kp1
            beta_k = beta_kp1
            beta1_k = beta1_kp1
            del_k = del_kp1
            nu_k = self.nu(beta_k)/self.Tk
            phi_k_problem = self._initialize_Phi_k_problem(
                x_k = x,
                y_k = y_k,
                mnmx_problem = problem,
                beta_k = beta_k,
                )

            with torch.no_grad():
                f_x = problem.func_f(x, no_grad=True)
                g_x = problem.inner_product(x, y_k)
            phi_k_x = phi_k_problem.objective(x, backward_pass=True)
            grad_phi_k_x = phi_k_problem.grad(x, repeat_forward=False)
            with torch.no_grad():
                grad_phi_k_x = manifold.project(x, grad_phi_k_x)
                grad_norm = manifold.norm(x, grad_phi_k_x, project=False)

            row = [k, t, f_x, g_x, phi_k_x, grad_norm, del_k, beta_k, nu_k, ss]
            column_printer.print_row(row)
            func_evals += 3
            # Because f(x) get evaluated again in Phi_k(x) and func_f.
            # This can be made more efficient.
            self._add_log_entry(k, x, phi_k_x)

            stopping_criterion = self._check_stopping_criteria(
                start_time,
                k,
                grad_norm,
                ss,
                func_evals
                )
            if stopping_criterion:
                if self.verbosity >=1:
                    print(stopping_criterion)
                    print("")
                break

        point = MinimaxProblemPoint(
            x=x,
            y=y_k,
            f_x=f_x,
            h_y= problem.func_h(y_k),
            in_prod=g_x
            )

        return self._return_result(
            start_time,
            point = point,
            phi_k_x = phi_k_x,
            iterations = k,
            stopping_criterion = stopping_criterion,
            function_evaluations = func_evals,
            step_size = ss,
            gradient_norm = grad_norm,
            dual_variable_change = del_k
        )


    def _initialize_Phi_k_problem(self,
            x_k: torch.Tensor,
            y_k: torch.Tensor,
            mnmx_problem: MinimaxProblem,
            beta_k: float
            ) -> Problem:
        with torch.no_grad():
            Ax_k = mnmx_problem.A(x_k)
            y_kp_half = mnmx_problem.prox_h(
                (Ax_k+beta_k*y_k)/(self.lda+beta_k),
                1/(self.lda + beta_k)
                )
            y_kp_half_energy = y_kp_half.pow(2).sum()*self.lda/2
            y_kp_half_minus_y_k_energy = (y_kp_half-y_k).pow(2).sum()
            y_kp_half_minus_y_k_energy = beta_k*y_kp_half_minus_y_k_energy/2
            h_y = mnmx_problem.func_h(y_kp_half)

        def Phi_k(x):
            f_x = mnmx_problem.func_f(x)
            with torch.no_grad():
                Ax = mnmx_problem.A(x)
                Axy = (Ax*y_kp_half).sum()
            return (f_x + Axy - h_y
                    - y_kp_half_energy - y_kp_half_minus_y_k_energy)

        def grad_Phi_k(x: torch.Tensor, repeat_forward=True, **kwargs):
            # if repeat_forward or x.grad is None:
            # required_grad = x.requires_grad
            # if required_grad is False:
            #     # x.requires_grad = True
            #     x.requires_grad_(True)
            #     # f_x = Phi_k(x)
            x = x.detach().requires_grad_(True)
            f_x = mnmx_problem.func_f(x, no_grad=False)
            f_x.backward()
            # x.requires_grad = required_grad
            grad_f = x.grad
            x.grad = None
            with torch.no_grad():
                Ax = mnmx_problem.A(x)
                z = (Ax + beta_k*y_k)/(self.lda+beta_k)
                prox_y = mnmx_problem.prox_h(z)
                nabla_AT = mnmx_problem.nabla_AT(x)
            grad = grad_f + nabla_AT@prox_y
            return grad
        return Problem(
            manifold = mnmx_problem.manifold,
            objective= Phi_k,
            grad_f= grad_Phi_k,
            )

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
            function_evaluations
            ) -> str:
        run_time = perf_counter() - start_time
        reason = None
        if self.max_time is not None and run_time >= self.max_time:
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

    def _initialize_column_printer(self) -> printer.VoidPrinter:
        if self.verbosity >= 1:
            print("RADA RGD Optimizing...")
        if self.verbosity >= 2:
            iteration_format_length = int(math.log(self.max_it, 10)) + 1
            iteration_format_length2 = int(math.log(self.Tk, 10)) + 1
            columns = [
                ("k", f"{iteration_format_length}d"),
                ("t", f"{iteration_format_length2}d"),
                ("f(x)", "+.12e"),
                ("<A(x), y>", "+.8e"),
                ("Φ_k(x)", "+.12e"),
                ("||grad Φ_k(x)||", ".6e"),
                ("δ_k", ".4e"),
                ("β_k", ".4e"),
                ("ν_k", ".4e"),
                ("α_kt+1", ".6e"),
                ]
            if self.step_size is None:
                columns.append(("α_kt (Armijo)", ".6e"))
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        return column_printer

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
        """Get the algorithm parameters as a dictionary"""
        params = {
            'lda': self.lda,
            'beta1': self.beta1,
            'Tk': self.Tk,    # py-lint: disable=invalid-name
            'rho': self.rho,
            'R': self.R,
            'eps': self.eps,
            'tau_1': self.tau_1,
            'tau_2': self.tau_2,
            'alpha_BB_max': self.alpha_BB_max,
            'alpha_BB_min': self.alpha_BB_min,
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
