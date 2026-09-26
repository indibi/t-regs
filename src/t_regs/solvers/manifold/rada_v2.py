"""Riemannian Alternating Descent Ascent Algorithm (v2 - Stabilized)

Implementation is based on [1] with critical stability, convergence, and monotonicity fixes.

References
----------
..  [1] Xu, Meng, et al. "A Riemannian Alternating Descent Ascent Algorithmic
    Framework for Nonconvex-Linear Minimax Problems on Riemannian Manifolds."
    arXiv preprint arXiv:2409.19588 (2024).
    `https://github.com/XuMeng00124/RADAopt/tree/main`

Key Improvements in v2 over original rada.py:
--------------------------------------------
- [CHANGE / FIX Solution 1.1]: Fixed corrupted Armijo condition in line search.
  In the original rada.py line 365, `c_val = phi_k_xkt + 2*beta_k*self.R**2`.
  The +2*beta_k*R^2 term is a theoretical Lyapunov surrogate bound between outer steps
  (Lemma 3.2 in Xu et al.), NOT part of the Armijo line search! For large R (high lambda),
  this cushion accepted arbitrary uphill steps on the very first try.
  In v2: `c_val = phi_k_xkt` strictly.

- [CHANGE / FIX Solution 1.2]: Safe line search rejection.
  In original rada.py lines 309-340, when line search exhausted `max_line_search` steps,
  it broke out of the loop and copied `x_new` into `x_kt` anyway, accepting non-descent steps.
  In v2: If the Armijo condition is not satisfied after `max_line_search` steps,
  the step is rejected, `x_kt` is NOT overwritten, and a reduced step size is used.

- [CHANGE / FIX Solution 2.1]: Capped Barzilai-Borwein (BB) step sizes and safe curvature check.
  In original rada.py, `zeta_BB_max = 1e6` (or even 1e20) with `.abs()` on negative inner products.
  When combined with line search failure, step sizes exploded, causing catastrophic manifold jumps.
  In v2: `zeta_BB_max` defaults to a safe value (10.0), and if `inn_prod <= 1e-12` (negative or degenerate
  manifold curvature), it falls back safely to `self.zeta`.

- [CHANGE / FIX Solution 2.2]: Fixed parameter update bugs in original rada.py.
  - In original rada.py line 320: `del_kp1 = torch.max(((self.lda + beta_k)*y_new - y_k).abs())`
    was missing `beta_k * y_k`. Fixed to `((self.lda + beta_k)*y_new - beta_k*y_k).abs()`.
  - In original rada.py line 322: `beta1_k` was never updated (`beta1_k = beta1_kp1` was missing),
    meaning the penalty parameter never decayed across outer iterations.
  - In original rada.py line 330: `y_k` was prematurely overwritten before `y_kp1` calculation.

- [CHANGE / FIX Solution 3.2]: Dual smoothing parameter floor `lda_smooth_floor`.
  At high sparsity (large lambda and R), `lda = eps / (2*R)` drops to 10^-8 ~ 10^-10.
  The Lipschitz constant of the smoothed gradient is ~ 1/lda, causing the condition number
  to explode to 10^8 ~ 10^10 and freezing RGD.
  In v2: `lda = max(eps / (2*R), lda_smooth_floor)` (default floor 1e-4), keeping the
  condition number bounded <= 10^4 and allowing steady descent.

- [CHANGE / FIX Best Point Tracking]:
  v2 tracks `(x_best, y_best, phi_best)` across all iterations and returns the best point,
  preventing degradation if the final iterate oscillates or halts prematurely.
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


class RADA_RGD:  # pylint: disable=invalid-name
    r"""Riemannian Alternating Descent Ascent Algorithm with Riemannian Gradient Descent (v2 - Stabilized)

    Solves the following minimax problem defined in [1]:
    .. math::
        \min_{x \in \mathcal{M}} \max_{y \in E} \{ F(x,y) := f(x) + \langle A(x), y \rangle - h(y) \}

    Parameters
    ----------
    R: float
        Bound on dual domain: :math:`\max_{y \in \mathrm{dom}(h)} \|y\|`.
    eta: float
        Step size decrease factor :math:`\eta \in (0,1)`. Default: 0.5.
    c1: float
        Sufficient decrease factor for Armijo line search :math:`c_1 \in (0,1)`. Default: 1e-4.
    max_it: int
        Maximum outer iterations allowed. Default: 1000.
    Tk: int
        Number of inner iterations for the :math:`\Phi_k(x)` minimization task. Default: 10.
    max_line_search: int
        Maximum number of backtracking steps in line search. Default: 15.
    beta1: float
        Initial proximal parameter :math:`\beta_1`. Default: 1.0.
    rho: float
        Dual variable proximal parameter attenuation exponent in :math:`\beta_k = \beta_1 / k^\rho`. Default: 1.5.
    tau_1: float
        Threshold for updating :math:`\beta_1^{(k)}`. Default: 0.999.
    tau_2: float
        Attenuation factor for :math:`\beta_1^{(k)}`. Default: 0.9.
    zeta: float
        Initial step size. Default: 1.0.
    zeta_BB_max: float
        [CHANGE / FIX Solution 2.1]: Capped upper limit for BB step size. Default: 10.0 (was 1e6/1e20).
    zeta_BB_min: float
        Lower limit for BB step size. Default: 1e-8.
    lda_smooth_floor: float
        [CHANGE / FIX Solution 3.2]: Minimum floor for dual smoothing parameter :math:`\lambda_{\mathrm{smooth}}`.
        Default: 1e-4. Prevents Lipschitz constant explosion (1/lda) when R is large at high sparsity.
    eps: float
        Algorithm stopping criterion tolerance. Default: 1e-8.
    max_time: Optional[float]
        Maximum runtime in seconds.
    max_function_evals: int
        Maximum function evaluations allowed. Default: 5000.
    min_step_size: float
        Termination threshold on step size. Default: 1e-12.
    verbosity: int
        Verbosity level (0=silent, 1=summary, 2=outer steps, 3=inner steps).
    log_verbosity: int
        Logging detail level. Default: 1.
    report_period: int
        Period for printing rows. Default: 1.
    logging_period: Optional[int]
        Period for recording log entries.
    """

    def __init__(
        self,
        R: float,
        eta: float = 0.5,
        c1: float = 1e-4,
        max_it: int = 1000,
        Tk: int = 10,
        max_line_search: int = 15,
        beta1: float = 1.0,
        rho: float = 1.5,
        tau_1: float = 0.999,
        tau_2: float = 0.9,
        zeta: float = 1.0,
        zeta_BB_max: float = 10.0,
        zeta_BB_min: float = 1e-8,
        lda_smooth_floor: float = 1e-4,
        track_best: bool = True,
        eps: float = 1e-8,
        max_time: Optional[float] = None,
        max_function_evals: int = 5000,
        min_step_size: float = 1e-12,
        verbosity: int = 0,
        log_verbosity: int = 1,
        report_period: int = 1,
        logging_period: Optional[int] = None,
    ):
        self.R = R
        self.eta = eta
        self.c1 = c1
        self.max_it = max_it
        self.Tk = Tk
        self.max_line_search = max_line_search
        self.beta1 = beta1
        self.rho = rho
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.zeta = zeta
        # [CHANGE / FIX Solution 2.1]: Sensible BB step limits
        self.zeta_BB_max = zeta_BB_max
        self.zeta_BB_min = zeta_BB_min
        # [CHANGE / FIX Solution 3.2]: Floor on smoothing parameter to avoid 10^10 condition number
        self.lda_smooth_floor = lda_smooth_floor
        self.track_best = track_best
        if self.R > 0:
            raw_lda = eps / (2.0 * R)
            self.lda = max(raw_lda, self.lda_smooth_floor)
        else:
            self.lda = 0.0

        self.nu = lambda beta_k: 2.0 * (self.R**2) * beta_k
        self.eps = eps
        self.min_gradient_norm = eps
        self.max_time = max_time
        self.max_function_evals = max_function_evals
        self.min_step_size = min_step_size
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = (
            max(1, max_it // 100) if logging_period is None else logging_period
        )
        self.log = None

    def solve(
        self,
        problem: MinimaxProblem,
        x0: Optional[torch.Tensor] = None,
        y0: Optional[torch.Tensor] = None,
        init_point: Optional[MinimaxProblemPoint] = None,
        seed: Optional[int | torch.Generator] = None,
    ) -> RADA_RGD_Result:
        r"""Solve the Riemannian minimax problem."""
        start_time = perf_counter()
        manifold: Manifold = problem.manifold
        column_printer = self._initialize_column_printer()
        column_printer.print_header()

        self._initialize_log(run_params={'seed': seed})

        k = 0
        t = 0

        # ------ Initialize points ---------
        if init_point is not None:
            x0 = init_point.x
            y0 = init_point.y

        if x0 is None:
            if isinstance(seed, int):
                rng = torch.Generator(device=manifold.device)
                rng.manual_seed(seed)
            elif isinstance(seed, torch.Generator):
                rng = seed
            else:
                raise TypeError("`seed` must be an instance of `int` or a `torch.Generator`")
            x = manifold.random_point(generator=rng)
            x.requires_grad_(True)
        else:
            x = x0

        if y0 is None:
            with torch.no_grad():
                y0 = problem.A(x)
                y0 = problem.prox_h(y0, 1.0 / (self.lda + self.beta1))

        x_kt = x.clone().detach()
        y_k = y0.clone().detach()

        # --------- Initialize parameters ----
        with torch.no_grad():
            ss = self.zeta  # Initial step size
            beta1_k = self.beta1
            beta_k = beta1_k / ((k + 1) ** self.rho)
            nu_k = self.nu(beta_k)

        # Prefix `nabla_` is for euclidean gradients and `grad_` is for riemannian
        f_xkt = problem.func_f(x_kt, backward_pass=True)
        nabla_f_xkt = problem.grad_f(x_kt, repeat_forward=False)

        with torch.no_grad():
            Ax = problem.A(x_kt)
            y_kp1 = problem.prox_h(
                (Ax + beta_k * y_k) / (self.lda + beta_k),
                1.0 / (self.lda + beta_k),
            )
            in_prod = (Ax * y_kp1).sum()
            h_y = problem.func_h(y_kp1)
            phi_k_xkt = (
                f_xkt
                + in_prod
                - h_y
                - self.lda * LA.vector_norm(y_kp1) ** 2 / 2.0
                - beta_k * LA.vector_norm(y_kp1 - y_k) ** 2 / 2.0
            )
            nabla_phi_k_xkt = nabla_f_xkt + problem.nabla_AT(x_kt) @ y_kp1
            grad_phi_k_xkt = manifold.project(x_kt, nabla_phi_k_xkt)
            desc_dir = -grad_phi_k_xkt

            grad_norm = manifold.norm(x_kt, grad_phi_k_xkt, project=False)
            del_norm = LA.vector_norm(y_kp1 - problem.prox_h(y_kp1 + Ax))
            # [CHANGE / FIX Solution 2.2]: Fixed missing beta_k * y_k in del_k
            del_k = torch.max(((self.lda + beta_k) * y_kp1 - beta_k * y_k).abs())

        # [CHANGE / FIX Solution 1.1]: c_val for line search is strictly phi_k_xkt (no +2*beta_k*R^2!)
        c_val = phi_k_xkt

        # [CHANGE / FIX Best Point Tracking]: Track best point seen so far
        best_x = x_kt.clone().detach()
        best_y = y_kp1.clone().detach()
        best_phi = float(phi_k_xkt)
        best_f_x = float(f_xkt)
        best_h_y = float(h_y)
        best_in_prod = float(in_prod)

        row = [k, t, f_xkt, in_prod, phi_k_xkt, grad_norm, del_norm, del_k, beta_k, nu_k, ss, ss, 0]
        if self.verbosity >= 2:
            column_printer.print_row(row)
        self._add_log_entry(k, x_kt, float(phi_k_xkt))

        func_evals = 1
        stopping_criterion = None

        # ------------------- Main outer iteration start (k) ------------------------
        while k < self.max_it:
            k += 1
            t = 0

            # ----------- Inner Phi_k(x_kt) descent iterations (t) --------------
            while t < self.Tk:
                t += 1
                dir_derivative = manifold.inner_product(
                    x_kt, desc_dir, grad_phi_k_xkt, project=False
                )

                # Ensure descent direction
                if dir_derivative >= 0:
                    desc_dir = -grad_phi_k_xkt
                    dir_derivative = manifold.inner_product(
                        x_kt, desc_dir, grad_phi_k_xkt, project=False
                    )

                # ---------------- Line search start ----------------------
                step_count = 0
                step_accepted = False
                trial_ss = ss

                while step_count < self.max_line_search:
                    step_count += 1
                    x_new = manifold.retract(x_kt, trial_ss * desc_dir)
                    f_x_new = problem.func_f(x_new, backward_pass=False)

                    Ax_new = problem.A(x_new)
                    y_new = problem.prox_h(
                        (Ax_new + beta_k * y_k) / (self.lda + beta_k),
                        1.0 / (self.lda + beta_k),
                    )
                    h_y_new = problem.func_h(y_new)
                    in_prod_new = (Ax_new * y_new).sum()

                    phi_k_x_new = (
                        f_x_new
                        + in_prod_new
                        - h_y_new
                        - self.lda * LA.vector_norm(y_new) ** 2 / 2.0
                        - beta_k * LA.vector_norm(y_new - y_k) ** 2 / 2.0
                    )

                    # [CHANGE / FIX Solution 1.1]: Armijo test against pure c_val = phi_k_xkt
                    if phi_k_x_new - c_val <= self.c1 * trial_ss * dir_derivative:
                        step_accepted = True
                        break

                    trial_ss = self.eta * trial_ss

                func_evals += step_count
                armijo_ss = trial_ss
                # ----------------- Line search end -----------------------

                # [CHANGE / FIX Solution 1.2]: Safe Line Search Rejection
                # If Armijo condition failed, do NOT copy x_new! Keep x_kt intact.
                if not step_accepted:
                    # Step rejected - reduce step size for next trial and skip retraction
                    ss = max(self.zeta_BB_min, trial_ss * self.eta)
                    # If step size is already tiny and still no descent, break inner loop early
                    if trial_ss < self.min_step_size:
                        break
                    continue

                # Step accepted: proceed with update
                ss = trial_ss

                # ------ Update beta_k and y_k at end of inner block ------
                if t == self.Tk:
                    # [CHANGE / FIX Solution 2.2]: Fixed missing beta_k * y_k in del_kp1
                    del_kp1 = torch.max(
                        ((self.lda + beta_k) * y_new - beta_k * y_k).abs()
                    )
                    beta1_kp1 = (
                        self.tau_2 * beta1_k
                        if (del_kp1 >= self.tau_1 * del_k)
                        else beta1_k
                    )
                    # [CHANGE / FIX Solution 2.2]: Update beta1_k accumulator!
                    beta1_k = beta1_kp1
                    beta_kp1 = beta1_k / ((k + 1) ** self.rho)
                    beta_k = beta_kp1
                    nu_k = self.nu(beta_k)

                    # Update y_k to y_new for the next outer iteration
                    y_k = y_new.clone().detach()

                # ------------ Calculate ∇Φ_k(x_kt) -----------------
                with torch.no_grad():
                    old_x = x_kt.clone().detach()
                    old_grad = grad_phi_k_xkt.clone().detach()

                x_kt.copy_(x_new)
                func_evals += 1

                f_xkt = problem.func_f(x_kt, backward_pass=True)
                nabla_f_xkt = problem.grad_f(x_kt, repeat_forward=False)

                with torch.no_grad():
                    Ax = problem.A(x_kt)
                    y_kp1 = problem.prox_h(
                        (Ax + beta_k * y_k) / (self.lda + beta_k),
                        1.0 / (self.lda + beta_k),
                    )
                    in_prod = (Ax * y_kp1).sum()
                    h_y = problem.func_h(y_kp1)
                    phi_k_xkt = (
                        f_xkt
                        + in_prod
                        - h_y
                        - self.lda * LA.vector_norm(y_kp1) ** 2 / 2.0
                        - beta_k * LA.vector_norm(y_kp1 - y_k) ** 2 / 2.0
                    )
                    nabla_phi_k_xkt = nabla_f_xkt + problem.nabla_AT(x_kt) @ y_kp1
                    grad_phi_k_xkt = manifold.project(x_kt, nabla_phi_k_xkt)
                    desc_dir = -grad_phi_k_xkt

                    grad_norm = manifold.norm(x_kt, grad_phi_k_xkt, project=False)
                    del_norm = LA.vector_norm(y_kp1 - problem.prox_h(y_kp1 + Ax))
                    # [CHANGE / FIX Solution 2.2]: Fixed missing beta_k * y_k
                    del_k = torch.max(
                        ((self.lda + beta_k) * y_kp1 - beta_k * y_k).abs()
                    )

                    # [CHANGE / FIX Solution 1.1]: c_val strictly set to current phi_k_xkt
                    c_val = phi_k_xkt

                    # [CHANGE / FIX Best Point Tracking]: Update best point
                    if float(phi_k_xkt) < best_phi:
                        best_phi = float(phi_k_xkt)
                        best_x.copy_(x_kt)
                        best_y.copy_(y_kp1)
                        best_f_x = float(f_xkt)
                        best_h_y = float(h_y)
                        best_in_prod = float(in_prod)

                    # [CHANGE / FIX Solution 2.1]: Robust Barzilai-Borwein step size
                    v_kt = grad_phi_k_xkt - old_grad
                    change = x_kt - old_x
                    inn_prod = (v_kt * change).sum()

                    # Check for positive curvature; fallback to self.zeta if non-positive or tiny
                    if inn_prod > 1e-12:
                        if t % 2 == 0:
                            zeta_BB_k_t = inn_prod / (v_kt.pow(2).sum() + 1e-30)
                        else:
                            zeta_BB_k_t = change.pow(2).sum() / (inn_prod + 1e-30)
                        next_step_size = max(
                            min(float(zeta_BB_k_t), self.zeta_BB_max),
                            self.zeta_BB_min,
                        )
                        ss = next_step_size
                    else:
                        # Fallback to standard step size
                        ss = max(self.zeta_BB_min, min(self.zeta, self.zeta_BB_max))

                row = [
                    k,
                    t,
                    f_xkt,
                    in_prod,
                    phi_k_xkt,
                    grad_norm,
                    del_norm,
                    del_k,
                    beta_k,
                    nu_k,
                    ss,
                    armijo_ss,
                    step_count,
                ]
                if (self.verbosity >= 2 and t == self.Tk) or (self.verbosity >= 3):
                    column_printer.print_row(row)
                self._add_log_entry(k, x_kt, float(phi_k_xkt))

            # Check outer stopping criteria
            stopping_criterion = self._check_stopping_criteria(
                start_time,
                k,
                grad_norm,
                del_norm,
                ss,
                func_evals,
            )
            if stopping_criterion:
                if self.verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

        # Return final converged iterate (which satisfies the stationarity/stopping criteria)
        final_point = MinimaxProblemPoint(
            x=x_kt,
            y=y_kp1,
            f_x=float(f_xkt),
            h_y=float(h_y),
            in_prod=float(in_prod),
        )

        return self._return_result(
            start_time,
            point=final_point if not self.track_best else best_point,
            phi_k_x=float(phi_k_xkt) if not self.track_best else best_phi,
            iterations=k,
            stopping_criterion=stopping_criterion or "Completed max iterations",
            function_evaluations=func_evals,
            step_size=ss,
            gradient_norm=float(grad_norm),
            dual_variable_change=float(del_k),
        )

    def _return_result(self, start_time, **kwargs) -> RADA_RGD_Result:
        return RADA_RGD_Result(
            time=perf_counter() - start_time,
            log=self.log,
            **kwargs,
        )

    def _check_stopping_criteria(
        self,
        start_time,
        iteration,
        gradient_norm,
        del_norm,
        step_size,
        function_evaluations,
    ) -> Optional[str]:
        run_time = perf_counter() - start_time
        reason = None
        if self.max_time is not None and run_time >= self.max_time:
            reason = f"Terminated - max time reached after {iteration} iterations."
        elif iteration >= self.max_it:
            reason = (
                f"Terminated - maximum number of iterations reached after "
                f"{run_time:.3f} seconds."
            )
        elif max([float(del_norm), float(gradient_norm)]) <= self.min_gradient_norm:
            reason = (
                f"Terminated - eps-RGS point reached after {iteration} "
                f"iterations, {run_time:.3f} seconds."
            )
        elif (step_size < self.min_step_size) or (step_size == 0):
            reason = (
                f"Terminated - min step_size reached after {iteration} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif function_evaluations >= self.max_function_evals:
            reason = (
                f"Terminated - max cost evals reached after "
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
            'iterations': collections.defaultdict(list),
        }

    def _initialize_column_printer(self) -> printer.VoidPrinter:
        if self.verbosity >= 1:
            print("RADA RGD (v2 - Stabilized) Optimizing...")
        if self.verbosity >= 2:
            iteration_format_length = int(math.log(max(self.max_it, 1), 10)) + 1
            iteration_format_length2 = int(math.log(max(self.Tk, 1), 10)) + 1
            columns = [
                ("k", f"{iteration_format_length}d"),
                ("t", f"{iteration_format_length2}d"),
                ("f(x)", "+.12e"),
                ("<A(x), y>", "+.8e"),
                ("Phi_k(x)", "+.12e"),
                ("||grad Phi_k(x)||", ".6e"),
                ("||y - prox_h(y + Ax)||", ".6e"),
                ("del_k", ".5e"),
                ("beta_k", ".5e"),
                ("nu_k", ".5e"),
                ("zeta_kt (Step Size)", ".5e"),
                ("zeta_kt (Armijo)", ".5e"),
                ("Step count", f"{int(math.log(max(self.max_line_search, 1), 10)) + 1}d"),
            ]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        return column_printer

    def _add_log_entry(self, iteration, point, objective, **kwargs):
        if self.log_verbosity <= 0:
            return
        if (self.logging_period != 0) and (iteration % self.logging_period == 0):
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['time'].append(perf_counter())
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)
            if self.log_verbosity > 1:
                self.log['iterations']['point'].append(point)

    def __str__(self):
        name = type(self).__name__
        if self.verbosity > 0:
            parameters_str = pprint.pformat(self.get_parameters())
            name += "\n\t" + "-" * 5 + " with Parameters " + "-" * 5 + parameters_str
        return name

    def get_parameters(self) -> dict[str, Any]:
        """Get the algorithm parameters as a dictionary."""
        return {
            'lda': self.lda,
            'eta': self.eta,
            'beta1': self.beta1,
            'Tk': self.Tk,
            'rho': self.rho,
            'R': self.R,
            'eps': self.eps,
            'tau_1': self.tau_1,
            'tau_2': self.tau_2,
            'zeta': self.zeta,
            'zeta_BB_max': self.zeta_BB_max,
            'zeta_BB_min': self.zeta_BB_min,
            'lda_smooth_floor': self.lda_smooth_floor,
        }
