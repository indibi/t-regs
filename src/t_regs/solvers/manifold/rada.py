"""Riemannian Alternating Descent Ascent Algorithm

Implementation is based on [1]

References
----------
..  [1] Xu, Meng, et al. "A Riemannian Alternating Descent Ascent Algorithmic
    Framework for Nonconvex-Linear Minimax Problems on Riemannian Manifolds."
    arXiv preprint arXiv:2409.19588 (2024).
    `https://github.com/XuMeng00124/RADAopt/tree/main`

Author:
Mert Indibi
1/17/2026
"""


import collections
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Optional, Any
import math
import pprint

import torch
import torch.linalg as LA

from ...manifolds import Manifold
# from .problem import Problem
from .minimax_problem import MinimaxProblemPoint, MinimaxProblem

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
    eta:
        Step size decrease factor :math:`eta \in (0,1)`.
    c1:
        Sufficient decrease factor :math:`c1 \in (0,1)`.
    max_it:
        Maximum iterations allowed for the algorithm.
    Tk:
        Number of inner iterations for the Phi_k(x) minimization task.
    max_line_search:
        Maximum number of steps to perform line search.
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
    zeta:
        Initial step size.
    zeta_BB_max:
        Upper step size limit for BB step size.
    zeta_BB_min:
        Lower limit for BB step size
    eps:
        Algorithm stopping criterion based on minimum gradient norm.
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
        eta: float = 0.1,
        c1: float = 1e-4,
        max_it: int = 1000,
        Tk: int = 10,  # py-lint: disable=invalid-name
        max_line_search: int= 10,
        beta1: float = 1.0,
        rho: float = 1.5,
        tau_1: float = 0.999,
        tau_2: float = 0.9,
        zeta: float = 1.0,
        zeta_BB_max: float = 1e6,
        zeta_BB_min: float = 1e-8,
        eps : float = 1e-8,
        max_time: float | None = None,
        max_function_evals: int = 5000,
        min_step_size:float = 0,
        verbosity: int = 0,
        log_verbosity: int = 1,
        report_period: int = 1,
        logging_period: int = None,
        ):
        self.R = R # pylint: disable=invalid-name
        self.eta = eta
        self.c1 = c1
        self.max_it = max_it #
        self.Tk = Tk    # py-lint: disable=invalid-name
        self.max_line_search = max_line_search
        self.beta1 = beta1
        self.lda = eps/(2*R) if self.R>0 else 0.0
        self.nu = lambda beta_k: 2*(self.R**2)*beta_k
        self.rho = rho
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.zeta = zeta
        self.zeta_BB_max = zeta_BB_max
        self.zeta_BB_min = zeta_BB_min
        self.eps = eps
        self.min_gradient_norm = eps
        self.max_time = max_time
        self.max_function_evals = max_function_evals
        self.min_step_size = min_step_size
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = max(
            1, max_it//100) if logging_period is None else logging_period
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
        x0:
            Initial :math:`x` value to begin the optimization. Defaults to
            random initialization with `seed`.
        y0:
            Initial :math:`y` value to begin the optimization. Defaults to
            random initialization with `seed`.
        init_point: Optional[MinimaxProblemPoint]
            Inital point to start the RADA optimization. Defaults to random
            initialization if not provided.
        seed: Optional[int | torch.Generator] = None
            Initialization seed for the algorithm.

        Returns
        -------
            result: RADA_RGD_Result
        """
        # pylint: disable=invalid-name
        start_time = perf_counter()
        manifold: Manifold = problem.manifold
        column_printer = self._initialize_column_printer()
        column_printer.print_header()


        self._initialize_log(run_params={'seed': seed})

        k = 0
        t = 0
        # ------ initialize points ---------
        if init_point is not None:
            x0 = init_point.x
            y0 = init_point.y
        if x0 is None:
            if isinstance(seed, int):
                rng = torch.Generator(device=manifold.device)
                rng.manual_seed(seed)
            elif isinstance(seed, torch.Generator):
                pass
            else:
                raise TypeError(
                    "`seed` must be an instance of `int` or a `torch.Generator`"
                    )
            x = manifold.random_point(generator=rng)
            x.requires_grad_(True)
        else:
            x = x0
        if y0 is None:
            with torch.no_grad():
                y0 = problem.A(x)
                y0 = problem.prox_h(y0, 1/(self.lda+self.beta1))
        x_kt = x.clone().detach()
        y_k = y0.clone().detach()
        # x_best = x.clone().detach()
        # y_best = y.clone().detach()
        # --------- initialize parameters ----
        with torch.no_grad():
            ss = self.zeta # Initial step size.
            beta1_k = self.beta1
            beta_k = beta1_k/(k+1)**self.rho
            nu_k = self.nu(beta_k)

        # Prefix `nabla_` is for euclidean gradients and `grad_` is for riem.
        f_xkt = problem.func_f(x_kt, backward_pass=True)
        nabla_f_xkt = problem.grad_f(x_kt, repeat_forward=False)
        with torch.no_grad():
            Ax = problem.A(x_kt)
            y_kp1 = problem.prox_h(
                (Ax + beta_k*y_k)/(self.lda+beta_k),
                1/(self.lda+beta_k)
                )
            # `1/(self.lda+beta_k)` is the for the h/(self.lda+beta_k) in
            # proximal operator. Has no effect if `h` is an indicator function.

            in_prod = (Ax*y_kp1).sum()
            h_y = problem.func_h(y_kp1)
            phi_k_xkt = (f_xkt + in_prod - h_y
                        - self.lda*LA.vector_norm(y_kp1)**2/2   # pylint: disable=not-callable
                        - beta_k*LA.vector_norm(y_kp1-y_k)**2/2 # pylint: disable=not-callable
                        )
            nabla_phi_k_xkt = nabla_f_xkt + problem.nabla_AT(x_kt)@y_kp1
            grad_phi_k_xkt = manifold.project(x_kt, nabla_phi_k_xkt)
            desc_dir = -grad_phi_k_xkt # Descent direction

            grad_norm = manifold.norm(x, grad_phi_k_xkt, project=False)
            del_k = torch.max(((self.lda + beta_k)*y_kp1 - beta_k*y_k).abs())
        c_val = phi_k_xkt
        row=[k, t, f_xkt, in_prod, phi_k_xkt, grad_norm, del_k, beta_k, nu_k, ss]
        column_printer.print_row(row)
        self._add_log_entry(k, x_kt, float(phi_k_xkt))
        # ("f(x_kt)", "+.12e"),
        # ("<A(x_kt), y_k>", "+.8e"),
        # ("Φ_k(x_kt)", "+.12e"),
        # ("||grad Φ_k(x_kt)||", ".6e"),
        # ("δ_k", ".4e"),
        # ("β_k", ".4e"),
        # ("ν_k", ".4e"),
        # ("ζ_kt+1", ".6e")
        # ("ζ_kt (Armijo)", ".6e"),
        func_evals = 1
        # ------------------- Main iteration start ------------------------
        while True:
            k += 1
            t = 0
            # ----------- Phi_k(x_kt) solution iteration (t) --------------
            while t < self.Tk:
                t +=1
                dir_derivative = manifold.inner_product(
                    x_kt,
                    desc_dir,
                    grad_phi_k_xkt,
                    project=False
                    )
                # ---------------- Line search start ----------------------
                step_count = 0
                while True:
                    x_new = manifold.retract(x_kt, ss*desc_dir)
                    f_x_new = problem.func_f(x_new, backward_pass=False)

                    Ax_new = problem.A(x_new)
                    y_new = problem.prox_h(
                        (Ax_new + beta_k*y_k)/(self.lda+beta_k),
                        1/(self.lda+beta_k)
                        )
                    h_y = problem.func_h(y_new, 1/(self.lda+beta_k))
                    in_prod_new = (Ax_new*y_new).sum()

                    phi_k_x_new = (
                        f_x_new + in_prod_new - h_y
                        - self.lda*LA.vector_norm(y_new)**2/2       # pylint: disable=not-callable
                        - beta_k*LA.vector_norm(y_new - y_k)**2/2   # pylint: disable=not-callable
                        )
                    step_count +=1
                    if ((phi_k_x_new - c_val <= self.c1*ss*dir_derivative)
                        or (step_count >= self.max_line_search)):
                        break

                    ss = self.eta*ss
                armijo_ss = ss
                func_evals += step_count
                # ----------------- Line search end -----------------------

                # ------ Update beta_k and y_k ------
                if t == self.Tk:
                    del_kp1 = torch.max(
                        ((self.lda + beta_k)*y_new - y_k).abs()
                        )
                    beta1_kp1 = (
                        self.tau_2*beta1_k if (del_kp1 >= self.tau_1*del_k)
                                            else beta1_k
                        )
                    beta_kp1 = beta1_kp1 / (k + 1)**self.rho

                    beta_k = beta_kp1

                    y_k = y_new.clone().detach()
                    y_new = problem.prox_h(
                        (Ax_new + beta_k*y_k)/(self.lda+beta_k),
                        1/(self.lda+beta_k)
                        )

                # ------------ Calculate ∇Φ_k(x_kt) -----------------
                with torch.no_grad():
                    old_x = x_kt.clone()
                    old_grad = grad_phi_k_xkt.clone()
                x_kt.copy_(x_new)
                func_evals +=1
                f_xkt = problem.func_f(x_kt, backward_pass=True)
                nabla_f_xkt = problem.grad_f(x_kt, repeat_forward=False)
                with torch.no_grad():
                    Ax = problem.A(x_kt)
                    y_kp1 = problem.prox_h(
                        (Ax + beta_k*y_k)/(self.lda+beta_k),
                        1/(self.lda+beta_k)
                        )
                    in_prod = (Ax*y_kp1).sum()
                    h_y = problem.func_h(y_kp1)
                    phi_k_xkt = (f_xkt + in_prod - h_y
                        - self.lda*LA.vector_norm(y_kp1)**2/2       # pylint: disable=not-callable
                        - beta_k*LA.vector_norm(y_kp1-y_k)**2/2     # pylint: disable=not-callable
                        )
                    nabla_phi_k_xkt = nabla_f_xkt + problem.nabla_AT(x_kt)@y_kp1
                    grad_phi_k_xkt = manifold.project(x_kt, nabla_phi_k_xkt)
                    desc_dir = -grad_phi_k_xkt # Descent direction

                    grad_norm = manifold.norm(x, grad_phi_k_xkt, project=False)
                    del_norm = LA.vector_norm(y_kp1 - problem.prox_h(y_kp1 + Ax)) # pylint: disable=not-callable
                    del_k = torch.max(
                        ((self.lda + beta_k)*y_kp1 - beta_k*y_k).abs()
                        )
                    c_val = phi_k_xkt + 2*beta_k*self.R**2

                    # ---- Calculate BB step size -------
                    v_kt = grad_phi_k_xkt - old_grad
                    change = x_kt - old_x
                    inn_prod = (v_kt*change).sum()
                    if t % 2 ==0:
                        zeta_BB_k_t = inn_prod.abs()/(v_kt.pow(2).sum()+1e-30)
                    else:
                        zeta_BB_k_t=(change.pow(2)).sum()/(inn_prod.abs()+1e-30)
                    next_step_size = max(
                        min(zeta_BB_k_t, self.zeta_BB_max),
                        self.zeta_BB_min
                        )
                    ss = next_step_size

                row=[k, t, f_xkt, in_prod, phi_k_xkt,
                    grad_norm, del_norm,
                    del_k, beta_k, nu_k, ss, armijo_ss, step_count]
                if (self.verbosity >= 2 and t==self.Tk) or (self.verbosity>=3):
                    column_printer.print_row(row)
                self._add_log_entry(k, x_kt, float(phi_k_xkt))

            # TODO: Add functionality to track and return the best result
            # based on minimization formulation of the optimization.
            stopping_criterion = self._check_stopping_criteria(
                start_time,
                k,
                grad_norm,
                del_norm,
                ss,
                func_evals
                )
            if stopping_criterion:
                if self.verbosity >=1:
                    print(stopping_criterion)
                    print("")
                break

        point = MinimaxProblemPoint(
            x=x_kt,
            y=y_kp1,
            f_x=f_xkt,
            h_y= h_y,
            in_prod=in_prod
            )

        return self._return_result(
            start_time,
            point = point,
            phi_k_x = phi_k_xkt,
            iterations = k,
            stopping_criterion = stopping_criterion,
            function_evaluations = func_evals,
            step_size = ss,
            gradient_norm = grad_norm,
            dual_variable_change = del_k
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
            del_norm,
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
        elif max([del_norm,gradient_norm]) <= self.min_gradient_norm:
            reason = (
                f"Terminated - eps-RGS point reached after {iteration} "
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


    def _initialize_log(self, *, run_params=None):
        self.log = {
            'solver': str(self),
            'stopping_criteria': {
                'max_time': self.max_time,
                'max_it': self.max_it,
                'min_gradient_norm': self.min_gradient_norm,
                'max_function_evals': self.max_function_evals,
                'min_step_size': self.min_step_size,
                },
            'parameters': self.get_parameters(),
            'run_params': run_params,
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
                ("||y - prox_h(y + Ax)||", ".6e"),
                ("δ_k", ".5e"),
                ("β_k", ".5e"),
                ("ν_k", ".5e"),
                ("ζ_kt (Step Size)", ".5e"),
                ("ζ_kt (Armijo)", ".5e"),
                ("Step count", f"{int(math.log(self.max_line_search, 10))+1}d"),
                ]
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
        name = type(self).__name__
        if self.verbosity >0:
            parameters_str = pprint.pformat(self.get_parameters())
            name += "\n\t"+"-"*5+" with Parameters "+"-"*5 + parameters_str
        return name

    def get_parameters(self) -> dict[str, Any]:
        """Get the algorithm parameters as a dictionary"""
        return {
            'lda': self.lda,
            'eta': self.eta,
            'beta1': self.beta1,
            'Tk': self.Tk,    # py-lint: disable=invalid-name
            'rho': self.rho,
            'R': self.R,
            'eps': self.eps,
            'tau_1': self.tau_1,
            'tau_2': self.tau_2,
            'zeta': self.zeta,
            'zeta_BB_max': self.zeta_BB_max,
            'zeta_BB_min': self.zeta_BB_min,
        }

    # def get_init_cfg(self) -> dict[str, Any]:
    #     """Get RADA-RGD Initialization settings."""
    #     return {

    #     }
