"""Proximal operator solver for Latent Overlapping Grouped Norm.

Author: Mert Indibi
Date: 3/4/2026
"""
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Optional
import math

import torch
import torch.linalg as LA

from .prox_lp_lq import prox_grouped_l21
from ..utils.variable_grouping import LatentGrouping
from ..utils import printer
# from ..solvers import admm

@dataclass
class ProxGTV_ADMMResult:
    r"""Approximate solution of prox_LOGN obtained by the ADMM algorithm.

    Attributes
    ----------
    x: torch.Tensor
        Approximate solution of the proximal operator of GTV
    v: torch.Tensor
        Second primal variable corresponding to the differences on edges
    y: torch.Tensor
        ADMM dual variable
    objective:
        Objective value, i.e :math:`0.5\|x - y\|_2^2 + \lambda \mathrm{GTV}(x)`
    lagrangian:
        Lagrangian of the ADMM formulation
    dual_residual:
        :math:`\rho \|v^{k} - v^{k-1}\|_2`
    primal_residual:
        :math:`\|B^T x - v\|_2`
    penalty_param:
        ADMM augmented lagrangian penalty parameter :math:`\rho`
    """
    x: torch.Tensor
    v: torch.Tensor
    y: torch.Tensor
    objective: float
    lagrangian: float
    dual_residual: float
    primal_residual: float
    penalty_param: float
    iterations: int
    stopping_criterion: str
    time: float
    log: Optional[Dict] = None

def prox_gtv_admm(
    u: torch.Tensor,
    lda: float,
    Bt: torch.Tensor,    # pylint: disable=invalid-name
    grouping: torch.Tensor,
    group_weights: torch.Tensor,
    BBt: torch.Tensor,  # pylint: disable=invalid-name
    max_it: int = 200,
    eps: float = 1e-6,
    verbose: int = 0,
    max_time: float=100,
    rho: float=1e-2,
    ) -> ProxGTV_ADMMResult:
    r"""Approximate the result of the proximal operator of GTV regularizer.

    Approximately solves the following optimization problem with ADMM algorithm,
    .. math::
        \mathrm{prox}_{\lambda \cdot \mathrm{GTV}(\cdot)} =
            \mathrm{argmin}_{u \in mathbb{R}^p} 0.5\|u - x\|_2^2 + \lambda 
            \mathrm{GTV}(x)
    
    where :math:`GTV: \mathbb{R}^{p} \to \mathbb{R}_+` is Graph Total Variation
    semi-norm. Grouping and the group weights are specified
    via `grouping` that represents the variable groupings.

    Parameters
    ----------
    u: torch.Tensor
        Tensor of shape :math:`p \times B` to to apply the proximal operator
        where :math:`B` is the batch dimensions.
    lda: float
        Regularization parameter
    Bt: sparse csr matrix
        Oriented incidence matrix of size |E| x |V|
    grouping: torch.Tensor
        Grouping indicator matrix of size |V| x |E|
    group_weights: torch.Tensor
        weights of each group corresponding to the edges originating from
        each vertex.
    BBt: torch.Tensor
        Laplacian matrix
    Returns
    -------
    ProxGTV_ADMMResult:
        Approximate result of the proximal operator.
    """
    device = u.device
    dtype = u.dtype
    if u.ndim == 1:
        u = u.reshape((-1,1))
    elif u.ndim > 2:
        raise ValueError("Tensor must be 2 dimensional.")
    p = Bt.shape[1]
    e = Bt.shape[0]
    b = u.shape[1]
    # Initialize Printer
    if verbose >=1:
        print("Optimizing for prox GTV...")
        iteration_format_length = int(math.log(max_it, 10)) + 1
        column_printer = printer.ColumnPrinter(
            columns=[
                ("Iteration", f"{iteration_format_length}d"),
                ("Objective", "+.6e"),
                ("Lagrangian", ".6e"),
                ("Primal Res.", ".6e"),
                ("Dual Res.", "+.6e"),
                ("Aug Lagr.", ".6e"),
            ]
        )
    else:
        column_printer = printer.VoidPrinter()
    column_printer.print_header()

    # Initialize Variables
    x = torch.zeros_like(u)    # First primal variable
    v = torch.zeros((e, b), device=device, dtype=dtype)   # Second primal variable
    y = torch.zeros_like(v)    # dual variable

    start_time = perf_counter()
    it = 0

    I_pBBT_inv = rho*(BBt).to_dense() + torch.eye(p,
                                                       dtype=dtype,
                                                       device=device)
    I_pBBT_inv = LA.inv(I_pBBT_inv)  # pylint: disable=not-callable
    # TODO: Should use a sparse solver. Good for now.
    while True:
        # Update first admm block
        x = I_pBBT_inv@(u + rho*((v.T - y.T/rho)@Bt).T)
        # Update second admm block
        v_kp1, group_norms = prox_grouped_l21(Bt@x + y/rho,
                                                lda/rho,
                                                groups=grouping,
                                                weights=group_weights,
                                                return_group_norms=True
                                                )

        dual_res = float(rho*LA.vector_norm(v_kp1 - v))   # pylint: disable=not-callable
        v = v_kp1
        res = Bt@x - v
        pri_res = float(LA.vector_norm(res))    # pylint: disable=not-callable
        obj = float(0.5*LA.vector_norm(u - x) + lda*group_norms.sum())  # pylint: disable=not-callable
        
        # Update dual variables
        y = y + rho*res
        lagr = float((res*y).sum()) + obj
        aug_lagr = float(lagr + 0.5*rho*pri_res)

        column_printer.print_row(
            [it, obj, lagr, pri_res, dual_res, aug_lagr]
        )

        # Check Stopping criterion
        run_time = perf_counter() - start_time
        reason = None
        if run_time >= max_time:
            reason = f"Terminated - max time reached after {it} iterations."
        elif it>= max_it:
            reason = ("Terminated - maximum number of iterations reached after "
                      f"{run_time:.3f} seconds.")
        elif max([pri_res, dual_res]) <= eps:
            reason = (
                "Terminated - convergence criteria is reached after "
                f"{it} iterations and {run_time:.3f} seconds."
            )
        if reason:
            if verbose:
                print(reason)
                print("")
            break
        it +=1

    return ProxGTV_ADMMResult(
        x = x,
        v = v,
        y = y,
        objective = obj,
        lagrangian = lagr,
        dual_residual = dual_res,
        primal_residual = pri_res,
        penalty_param = rho,
        iterations = it,
        stopping_criterion = reason,
        time = perf_counter() - start_time
        )


# class Prox_LOGN_ADMM(admm.ADMMBaseClass):
    
#     def run(self,
#             x: torch.Tensor,
#             lda: float,
#             grouping: LatentGrouping,
#             v1_init: torch.Tensor | None = None,
#             v2_init: torch.Tensor | None = None,
#             y_init:  torch.Tensor | None = None,
#             ):
#         pass

#     def _check_stopping_criteria

