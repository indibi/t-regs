"""Re-implementation of Tensor Robust PCA Algorithm [1] with PyTorch

References
----------
..  [1] Lu, C., Feng, J., Chen, Y., Liu, W., Lin, Z., & Yan, S. (2019).
    Tensor robust principal component analysis with a new tensor nuclear norm.
    IEEE transactions on pattern analysis and machine intelligence, 42(4),
    925-938.

Author: Mert Indibi 2/23/2026
"""
import math
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.functional.F.softshrink as softshrink  # pylint: disable=import-error
import torch.linalg as LA

from ...proximal_ops.prox_tnn import prox_tnn
from ...solvers import admm
from ...utils import printer

@dataclass
class TRPCAResult:
    """TRPCA Decomposition Algorithm Result
    
    Attributes
    ----------
        L: torch.Tensor
            Low t-rank part
        E: torch.Tensor
            Sparse part
        Y: torch.Tensor
            Dual variable
        objective: float
        stopping_criterion: str
            Algorithm termination reason.
        time: float
            Time taken by the algorithm in seconds.
        lagrangian: float
        primal_residual: float
        dual_residual: float
        penalty_param: float
        iterations: int
        log: Optional[Dict]
    """
    L: torch.Tensor
    E: torch.Tensor
    Y: torch.Tensor
    objective: float
    stopping_criterion: str
    time: float
    lagrangian: float
    primal_residual: float
    dual_residual: float
    penalty_param: float
    iterations: int
    log: Optional[Dict] = None


class TRPCA(admm.ADMMBaseClass):
    r"""Tensor Robust PCA with t-SVD and Tensor Nuclear Norm
    """
    def run(
        self,
        X: torch.Tensor,
        lda: float,
        tnn_mode: int = 1,
        rho: float = 1.1,
        mu_0: float = 1e-3,
        mu_max: float = 1e10,
        eps: float = 1e-8,
        E_init: torch.Tensor = None,
        L_init: torch.Tensor = None,
        Y_init: torch.Tensor = None,
        device:str|torch.device= 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype:torch.dtype = torch.double,
        ) -> TRPCAResult:
        """Optimize for TRPCA problem
        
        Parameters
        ----------
            X: torch.Tensor
                Tensor to be decomposed.
            lda: float | None
                Sparsity regularization parameter
            tnn_mode: int
                The mode to slice the tensor.
            mu_0: float
                Initial step size for ADMM
            mu_max: float
                Maximum step size for ADMM
            rho: float
                ADMM step size growth factor
            eps: float
                Convergence criterion
            E_init: admm.ADMMPoint | None
                Initial point for ADMM to start.
            device: torch.device | str
                Device to run the algorithm on.
            dtype: torch.dtype
        """
        X = X.to(device, dtype)
        L = (torch.zeros_like(X) if L_init is None
             else L_init.to(device=device, dtype=dtype))
        E = (torch.zeros_like(X) if L_init is None
             else E_init.to(device=device, dtype=dtype))
        Y = (torch.zeros_like(X) if L_init is None
             else Y_init.to(device=device, dtype=dtype))
        
        
        col_printer = self._initialize_printer()
        start_time = perf_counter()
        self._initialize_log(run_config={
            'lda':lda,
            'mu_0': mu_0,
            'mu_max': mu_max,
            'rho': rho,
            'eps': eps,
        })
        it = 0
        mu_k = mu_0
        while True:
            it +=1

            L_new, tnn, t_ranks = prox_tnn(X-E-Y/mu_k, tau=1/mu_k, mode=tnn_mode)

            E_new = softshrink(X - L_new - Y/mu_k, lda/mu_k)

            r = (L_new + E_new - X)
            s = (E_new - E)
            Y_new = Y + mu_k*r

            # Calculate obj etc.
            del_E_max = float(torch.max(s.abs()))
            del_L_max = float(torch.max((L_new - L).abs()))
            pri_res_max = float(torch.max(r.abs()))
            obj = float(tnn + lda*LA.vector_norm(E_new, ord=1))    # pylint: disable=not-callable
            lagr = float(obj + (r*Y_new).sum())
            pri_res = float(LA.vector_norm(r, ord=2))  # pylint: disable=not-callable
            dual_res = float(LA.vector_norm(s, ord=2)) # pylint: disable=not-callable
            step_size = float(mu_k)
            aug_lagr = float(lagr + pri_res*mu_k)
            # Update variables
            E, E_new = E_new, E
            L, L_new = L_new, L
            Y, Y_new = Y_new, Y
            # Report
            col_printer.print_row(
                [it, obj, lagr, pri_res, dual_res, step_size, aug_lagr,
                del_L_max, del_E_max, pri_res_max]
            )
            self._add_log_entry(it, obj, lagr, pri_res, dual_res,
                                step_size=step_size,
                                del_E_max=del_E_max,
                                del_L_max=del_L_max,
                                pri_res_max=pri_res_max,
                                t_ranks=t_ranks,
                                )

            stop_criterion = self._check_stopping_criteria(
                start_time, it, eps, del_E_max, del_L_max, pri_res_max
                )
            if stop_criterion:
                if self._verbosity >=1:
                    print(stop_criterion)
                    print("")
                break
            else:
                mu_k = min([mu_k*rho, mu_max])

        return TRPCAResult(
            L = L,
            E = E,
            Y = Y,
            objective = obj,
            stopping_criterion = stop_criterion,
            time = perf_counter() - start_time,
            lagrangian = lagr,
            primal_residual = pri_res,
            dual_residual = dual_res,
            penalty_param = step_size,
            iterations = it,
            log = self._log,
        )


    def _initialize_printer(self):
        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(math.log(self._max_it, 10)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Objective", "+.6e"),
                    ("Lagrangian", ".6e"),
                    ("Primal Res.", ".6e"),
                    ("Dual Res.", "+.6e"),
                    ("Step size", ".6e"),
                    ("Aug Lagr.", ".6e"),
                    ("||L_{k+1}-L_k||_inf",".6e"),
                    ("||E_{k+1}-E_k||_inf",".6e"),
                    ("||L+E-X||_inf", ".6e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()
        column_printer.print_header()
        return column_printer


    def _check_stopping_criteria(self,
                             start_time,
                             iteration,
                             eps,
                             del_E_max,
                             del_L_max,
                             pri_res_max,
                             ):
        run_time = perf_counter() - start_time
        reason = None
        if run_time >= self._max_time:
            reason = f"Terminated - max time reached after {iteration} iterations."
        elif iteration>= self._max_it:
            reason = ("Terminated - maximum number of iterations reached after "
                      f"{run_time:.3f} seconds.")
        elif max([del_E_max, del_L_max, pri_res_max]) <= eps:
            reason = (
                "Terminated - convergence criteria is reached after "
                f"{iteration} iterations and {run_time:.3f} seconds."
            )
        return reason