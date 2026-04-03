"""Module for Riemannian Gradient Descent."""

import collections
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import numpy as np

from ...manifolds import Manifold
from .problem import Problem
from .line_searcher import LineSearcher, ArmijoPointSearch
from ...utils import printer

@dataclass
class GradientDescentResult: # pylint: disable=missing-class-docstring
    point: Any
    objective: float
    iterations: int
    stopping_criterion: str
    time: float
    function_evaluations: Optional[int] = None
    step_size: Optional[float] = None
    gradient_norm: Optional[float] = None
    log: Optional[Dict] = None



class RiemmannianGradientDescent:
    r"""Riemmannian Gradient Descent Solver

    Solves optimization problems of the form
    .. math::
        \min_{x \in \mathcal{M}} f(x)
    
    where :math:`f: \mathcal{M} \to \mathbb{R}` is a differentiable function
    and :math:`\mathcal{M}` is a (smooth) Riemmannian Manifold.

    Parameters
    ----------
        line_searcher: LineSearcher | None
            Line searching algorithm used for the descent step size. Defaults to
            the default parameters of backtracking line search.
        step_size: float | None = None
            Fixed step size for the retraction step.
        max_it : int = 1000
            Maximum number of iterations the algorithm is allowed to run for
        min_gradient_norm : float = 1e-8
            Termination threshold based on the norm of the riemmannian gradient
        max_time : float | None = None
            Upper bound on run time of the solver in seconds.
        max_function_evals : int = 5000
            Maximum number of function evaluations allowed for the algorithm.
        min_step_size : float = 0
            Termination threshold used with line search algorithm.
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
                 min_gradient_norm : float = 1e-8,
                 max_time: float | None = None,
                 max_function_evals: int = 5000,
                 min_step_size:float = 0,
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
            elif isinstance(line_searcher, (LineSearcher, ArmijoPointSearch)):
                self.line_searcher = line_searcher
            else:
                raise TypeError(("No step size is provided and the line "
                                 "search algorithm is not of type LineSearcher "
                                 "or ArmijoPointSearch")
                )
        else:
            if step_size <= 0:
                raise ValueError(("Provided step size must be positive"))

        self.max_it = max_it
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
              problem: Problem,
              x0: torch.Tensor | None = None,
              generator: torch.Generator = None
              ) -> GradientDescentResult:
        r"""Solve the optimization problem
        
        Parameters
        ----------
        problem:
            Problem object representing the manifold constrained optimization.
        x0:
            Initial point to start the optimization
        generator:
            Torch pseudo random generator used to initialize `x0` when it is not
            provided.
        
        Returns
        -------
            result: GradientDescentResult
        """
        manifold: Manifold = problem.manifold
        if x0 is None:
            x = manifold.random_point(generator=generator)
        else:
            x = x0

        if self.verbosity >= 1:
            print("Optimizing...")
        if self.verbosity >= 2:
            iteration_format_length = int(np.log10(self.max_it)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.5e"),
                    ("Gradient norm", ".5e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

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

        start_time = perf_counter()
        func_evals = 1
        it = 0
        f_x = problem.objective(x, backward_pass=True)
        nabla_x = problem.grad(x, repeat_forward=False)
        descend_dir = -manifold.project(x, nabla_x) # Descent direction
        # `descend_dir` already on the tangent space, no need to project.
        grad_norm = manifold.norm(x, descend_dir, project=False)
        log = {}
        column_printer.print_row([it, f_x, grad_norm])
        self._add_log_entry(start_time, it, x, f_x)
        while True:
            it += 1
            if self.step_size is None:
                if isinstance(self.line_searcher, ArmijoPointSearch):
                    search_result = self.line_searcher.search(
                                problem, x, descend_dir, f_x, descend_dir,
                                )
                    step_size = search_result.step_size
                    x = search_result.x_new
                    step_count = search_result.step_count
                    descend_dir = -search_result.new_grad_f_x
                    log = {'step_size': step_size, 'step_count': step_count}
                else:
                    step_size, x, step_count = self.line_searcher.search(
                    problem, x, descend_dir, f_x, -(grad_norm**2)
                    )
                    f_x = problem.objective(x, backward_pass=True)
                    nabla_x = problem.grad(x, repeat_forward=False)
                    func_evals += 1
                    descend_dir = -manifold.project(x, nabla_x)
                    grad_norm = manifold.norm(x, descend_dir, project=False)
                func_evals += step_count
            else:
                step_size = self.step_size
                x = manifold.retract(x, step_size*descend_dir)

                f_x = problem.objective(x, backward_pass=True)
                nabla_x = problem.grad(x, repeat_forward=False)
                func_evals += 1
                descend_dir = -manifold.project(x, nabla_x)
                grad_norm = manifold.norm(x, descend_dir, project=False)

            column_printer.print_row([it, f_x, grad_norm])
            self._add_log_entry(start_time, it, x, f_x, **log)

            stopping_criterion = self._check_stopping_criteria(start_time,
                                                               it,
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
            objective = f_x,
            iterations = it,
            stopping_criterion = stopping_criterion,
            function_evaluations = func_evals,
            step_size = step_size,
            gradient_norm = grad_norm,
        )


    def _return_result(self, start_time, **kwargs) -> GradientDescentResult:
        return GradientDescentResult(
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
        elif (step_size < self.min_step_size):#  or (step_size ==0):
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

    def _add_log_entry(self, start_time, iteration, point, objective, **kwargs):
        if self.log_verbosity <=0:
            return
        if (self.logging_period !=0) and (iteration % self.logging_period ==0):
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['time'].append(perf_counter()-start_time)
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)

            if self.log_verbosity >1:
                self.log['iterations']['point'].append(point)


    def __str__(self):
        if self.step_size is not None:
            name = type(self).__name__ + " with fixed step size"
        else:
            name = type(self).__name__ + (
                f" with {self.line_searcher.step_size_strategy} step size")
        return name

    def get_parameters(self) -> dict: # pylint: disable=missing-function-docstring
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
