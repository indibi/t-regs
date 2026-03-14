r"""Abstract Base Classes for ADMM algorithms"""

import collections
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Dict, Optional
import abc

from .penalty_strategy import ResidualBalancingPP, ADMMPenaltyParameterStrategy

@dataclass
class ADMMPoint:
    """ADMM Algorithm Primal and Dual Variables
    
    Attributes
    ----------
        first_block: Any
            First block of variables.
        second_block: Any
            Second Block of variables.
        dual: Any
            Dual variables.
    """
    first_block: Any
    second_block: Any
    dual: Any


@dataclass
class ADMMResult:
    r"""ADMM Algorithm Result

    Attributes
    ----------
        point: ADMMPoint
        objective: float
            :math:`f(x) + g(z)`.
        lagrangian: float
            :math:`f(x) + g(z) + y^T(Ax+Bz-c)`.
        primal_residual: float
            :math:`\|r\|_2 = \|Ax+Bz-c\|_2`.
        dual_residual: float
            :math:`\|s^k\|_2 = \|\rho A^T B (z^{k}-z^{k-1}\|_2`.
        penalty_param: float
            Augmented lagrangian penalty parameter.
        iterations: int
            Number of iterations that the algorithm ran for.
        stopping_criterion: str
            Termination reason of the algorithm.
        time: float
            Algorithm run time in seconds.
        log: Dict | None
            Log of the algorithm run.
    """
    point: ADMMPoint
    objective: float
    lagrangian: float
    primal_residual: float
    dual_residual: float
    penalty_param: float
    iterations: int
    stopping_criterion: str
    time: float
    log: Optional[Dict] = None


class ADMMBaseClass(abc.ABC):
    r"""Abstract Base Class for ADMM Algorithms

    Solves the following optimization problem:
    .. math::
        \min_{\beta} f(x) + g(z)
        \text{subject to} Ax + Bz = c

    with variables :math:`x \in \mathbb{R}^n` and :math:`z \in \mathbb{R}^m`
    where :math:`A \in \mathbb{R}^{p\times n}, B \in \mathbb{R}^{p\times m},
    and :math:`c \in \mathbb{R}^p.
    
    References
    ----------
    ..  [1] S. Boyd and N. Parikh and E. Chu and B. Peleato and J.
        Eckstein (2010), “Distributed optimization and statistical
        learning via the alternating direction method of multipliers”
    """
    def __init__(
        self,
        *vargs,
        max_it: int = 1000,
        eps_abs: float = 1e-6,
        eps_rel: float = 0,
        max_time: float | None = None,
        verbosity: int = 0,
        log_verbosity: int = 1,
        report_period: int = 1,
        logging_period: int = 1,
        **kwargs
        ):
        """Initialize the ADMM Solver
        
        Parameters
        -----------
        max_it: int
            Maximum number of ADMM iterations allowed
        eps_abs: float
            Absolute termination criterion
        eps_rel: float
            Relative termination criterion
        max_time: float | None = None
            Upper bound on run time of the solver in seconds.
        verbosity: int = 0
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
        self._max_it = max_it
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel
        self._max_time = max_time
        self._verbosity = verbosity
        self._log_verbosity = log_verbosity
        self._report_period = report_period
        self._logging_period = logging_period
        self._log = None

    def __str__(self):
        return type(self).__name__
    
    @abc.abstractmethod
    def run(
        self,
        *vargs,
        initial_point: ADMMPoint | None = None,
        **kwargs) -> ADMMResult:
        """Run ADMM optimizer
        
        Parameters
        ----------
            *vargs:
                Problem-specific arguments
            initial_point: Any = None
                Starting point of the algorithm.
            **kwargs:
                Problem-specific keyword arguments.
        Returns
        -------
            ADMMResult
        """

    def _return_result(self, *, start_time, **kwargs) -> ADMMResult:
        return ADMMResult(
            time=perf_counter()-start_time,
            log=self._log,
            **kwargs,
        )

    def _initialize_log(self, *, run_config=None):
        self._log = {
            'optimizer': str(self),
            'stopping_criteria': {
                'max_time': self._max_time,
                'max_it': self._max_it,
                'eps_abs': self._eps_abs,
                'eps_rel': self._eps_rel,
                },
            'run_config': run_config,
            'iterations': collections.defaultdict(list)
            if self._log_verbosity >= 1
            else None,
        }

    def _add_log_entry(
        self,
        iteration,
        objective,
        lagrangian,
        primal_residual,
        dual_residual,
        **kwargs
        ):
        if self._log_verbosity <=0:
            return
        if (self._logging_period !=0) and (iteration % self._logging_period ==0):
            self._log['iterations']['iteration'].append(iteration)
            self._log['iterations']['time'].append(perf_counter())
            self._log['iterations']['objective'].append(objective)
            self._log['iterations']['lagrangian'].append(lagrangian)
            self._log['iterations']['primal_residual'].append(primal_residual)
            self._log['iterations']['dual_residual'].append(dual_residual)
            for key, value in kwargs.items():
                self._log['iterations'][key].append(value)


    def _check_stopping_criteria(self,
                             start_time,
                             iteration,
                             pri_res,
                             dual_res,
                             eps_pri,
                             eps_dual
                             ):
        run_time = perf_counter() - start_time
        reason = None
        if run_time >= self._max_time:
            reason = f"Terminated - max time reached after {iteration} iterations."
        elif iteration>= self._max_it:
            reason = ("Terminated - maximum number of iterations reached after "
                      f"{run_time:.3f} seconds.")
        elif ((pri_res < eps_pri) and (dual_res < eps_dual)):
                reason = (
                    "Terminated - convergence criteria is reached after "
                    f"{iteration} iterations and {run_time:.3f} seconds."
                )
        return reason
