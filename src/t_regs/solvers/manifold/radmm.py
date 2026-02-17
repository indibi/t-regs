r"""Riemannian ADMM solver for Smooth + Convex Non-smooth Problems on a Manifold

Solves the problems of form
.. math::
    \begin{*aligned}
    \min_{x, y} & f(x) + g(y) \\
    \mathrm{s.t.} &\quad Ax=y \\
    & x\in \mathcal{M}
    \end{*aligned}

where :math:`\mathcal{M}` is a riemmannian manifold.

References
----------
..  [1] Li, Jiaxiang, Shiqian Ma, and Tejes Srivastava. "A Riemannian 
    alternating direction method of multipliers." Mathematics of Operations
    Research 50.4 (2025): 3222-3242.
"""

import collections
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

import torch
import numpy as np

from ...manifolds import Manifold
from .line_searcher import LineSearcher
from ...utils import printer

@dataclass
class RADMMResult:
    x: Any
    y: Any
    dual_var: Any        # lambda
    gradient_norm: float # ||G^{k}||
    residual_norm: float # ||Ax - y||
    fx: float            # f(x) + g(Ax)
    gy: float            # g(y)
    objective: float     # f(x) + g(Ax)
    iterations: int
    stopping_criterion: str
    time: float
    step_size: Optional[float] = None
    log: Optional[Dict] = None

class RADMM:
    r"""Riemannian ADMM solver for Smooth + Convex Problems on a Manifold

    Solves the problems of form
    .. math::
        \begin{*aligned}
        \min_{x, y} & f(x) + g(y) \\
        \mathrm{s.t.} &\quad Ax=y \\
        & x\in \mathcal{M}
        \end{*aligned}

    where :math:`\mathcal{M}` is a riemmannian manifold.

    Parameters
    ----------
        line_searcher: LineSearcher | None = None
            Line search algorithm to use if a step size is not provided.
        step_size: float | None = None
            Step size for the retraction step. 
        max_it : int = 1000
            Maximum number of iterations the algorithm is allowed to run for
        eps : float = 1e-8
            Termination threshold based on the subgradient of the augmented
            lagrangian, i.e. :math:`\epsilon` stationarity condition.
        verbosity : int = 0
            Level of verbosity of the algorithm
        log_verbosity : int = 1
            Level of verbosity for algorithm logging.
                0: No logging performed
                1: Everything but the point is logged
                2: Algorithm iterations are also saved.
        report_period : int = 1
            Controls how often the results are reported as algorithm iterates.
        logging_period : int = 1
            Controls how often the results are logged.
    """
    def __init__(self,
                line_searcher: LineSearcher | None = None,
                step_size: float | None = None,
                max_it: int = 1000,
                eps: float = 1e-8,
                verbosity: int = 0,
                log_verbosity: int = 1,
                report_period: int = 1,
                logging_period: int = 1,
                ):
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

        self.max_it = max_it
        self.eps_stationarity = eps
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = logging_period

        self.log = None
    
    def solve(self, ):
        r"""

        Parameters
        ----------
        manifold: Manifold
            Manifold to optimize function :math:`f(x)+g(Ax)`over.
        
        gamma: float
            Moreau envelope smoothing parameter for function :math:`g(y)`
        rho: float
            Augmented Lagrangian penalty parameter
        """
        pass