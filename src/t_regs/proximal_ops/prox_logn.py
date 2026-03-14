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
class ProxLOGN_ADMMResult:
    r"""Approximate solution of prox_LOGN obtained by the ADMM algorithm.

    Attributes
    ----------
    x: torch.Tensor
        Approximate output of the proximal operator of LOGN
    v1: torch.Tensor
        First copy of the latent variables, first admm primal variable
    v2: torch.Tensor
        Second copy of the latent variables, second admm primal variable
    y: torch.Tensor
        ADMM dual variable
    objective:
        Objective value, i.e :math:`0.5\|x - y\|_2^2 + \lambda \mathrm{LOGN}(x)`
    lagrangian:
        Lagrangian of the ADMM formulation
    dual_residual:
        :math:`\rho \|v_2^{k} - v_2^{k-1}\|_2`
    primal_residual:
        :math:`\|v_1 - v_2\|_2`
    penalty_param:
        ADMM augmented lagrangian penalty parameter :math:`\rho`
    """
    x: torch.Tensor
    v1: torch.Tensor
    v2: torch.Tensor
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

def prox_logn_admm(
    x: torch.Tensor,
    lda: float,
    grouping: LatentGrouping,
    max_it: int = 200,
    eps: float = 1e-6,
    verbose: int = 0,
    max_time: float=100,
    rho: float=1e-2,
    ) -> ProxLOGN_ADMMResult:
    r"""Approximate the result of the proximal operator of LOGN regularizer.

    Approximately solves the following optimization problem with ADMM algorithm,
    .. math::
        \mathrm{prox}_{\lambda \cdot \mathrm{LOGN}(\cdot)} =
            \mathrm{argmin}_{u \in mathbb{R}^p} 0.5\|u - x\|_2^2 + \lambda 
            \mathrm{LOGN}(x)
    
    where :math:`LOGN: \mathbb{R}^{p} \to \mathbb{R}_+` is the Latent 
    overlapping grouped norm. Grouping and the group weights are specified
    via `grouping` that represents the variable groupings.

    Parameters
    ----------
    x: torch.Tensor
        Tensor of shape :math:`p \times B` to to apply the proximal operator
        where :math:`B` is the batch dimensions.
    lda: float
    grouping: LatentGrouping
        Grouping object specifying the variable groupings and weights.
    
    Returns
    -------
    ProxLOGN_ADMMResult:
        Approximate result of the proximal operator.
    """
    device = x.device
    dtype = x.dtype
    ogdim = x.shape
    if x.ndim == 1:
        x = x.reshape((-1,1))
    elif x.ndim > 2:
        raise ValueError("Tensor must be 2 dimensional.")
    # p = x.shape[0]
    b = x.shape[1]
    nolv = grouping.nolv
    
    # Initialize Printer
    if verbose >=1:
        print("Optimizing for prox LOGN...")
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
    v1 = torch.zeros((nolv, b), device=device, dtype=dtype) # First dual variable
    v2 = torch.zeros_like(v1)   # Second primal variable
    y = torch.zeros_like(v1)    # dual variable

    start_time = perf_counter()
    it = 0
    HT = grouping.H_l
    H = HT.transpose(0,1)
    HHT_pI_inv=(H@HT).to_dense() + rho*torch.eye(nolv,dtype=dtype,device=device)
    HHT_pI_inv = LA.inv(HHT_pI_inv).to_sparse_csr() # pylint: disable=not-callable
    weights = grouping.weights.reshape((-1,1))
    # HHT + rho * I is a very sparse matrix.
    # TODO: Should use a sparse solver. Good for now.
    while True:
        # Update first admm block
        v1 = HHT_pI_inv@(H@x + rho*v2 - y)
        # Update second admm block
        v2_kp1, group_norms = prox_grouped_l21(v1 + y/rho,
                                                lda/rho,
                                                groups=grouping.grouping_l,
                                                weights=weights,
                                                return_group_norms=True
                                                )

        dual_res = float(rho*LA.vector_norm(v2_kp1 - v2))   # pylint: disable=not-callable
        v2 = v2_kp1
        res = v1 - v2
        pri_res = float(LA.vector_norm(res))    # pylint: disable=not-callable
        obj = float(0.5*LA.vector_norm(x - HT@v1) + lda*group_norms.sum())  # pylint: disable=not-callable
        
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

    return ProxLOGN_ADMMResult(
        x = (HT@v2).reshape(ogdim),
        v1 = v1,
        v2 = v2,
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

