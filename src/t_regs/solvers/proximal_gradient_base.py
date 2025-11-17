from abc import ABC, abstractmethod
from collections import defaultdict
from warnings import warn
from time import perf_counter

import torch
import numpy as np
import matplotlib.pyplot as plt

from .metric_tracker import MetricTracker

class ProximalGradientBase(ABC):
    r"""Base class for Proximal Gradient optimization algorithms.
    
    Solves optimization problems of the form:
    ..:math
        \min_x f(x) + g(x)
    
    where :math:`f` is a differentiable function and :math:`g` is a
    possibly non-differentiable function with a known proximal operator.

    Children classes must implement the following methods:
        - func_f(self, x: torch.Tensor | np.ndarray) -> float:
        - func_g(self, x: torch.Tensor | np.ndarray) -> float:
        - prox_step(self, x, step_size) -> torch.Tensor:
        - grad_f(self, x): -> torch.Tensor:
    Please see the method docstrings for more information.

    Methods
    -------
        __call__(self, x0: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor:
            Iterate over the algorithm until convergence or reaching maximum iterations.
        

    Attributes
    ----------
        x : torch.Tensor | np.ndarray
            Current estimate of the solution.
        device : str | torch.device
            Device to use for computations ('cpu' or 'cuda:{i}').
        dtype : torch.dtype
            Data type for computations.
        it : int
            Current iteration number.
        objective : list[float]
            History of objective function values.
        delta_objective : list[float]
            History of changes in objective function values.
        delta_x : list[float]
            History of changes in solution estimates.
        converged : bool
            Whether the algorithm has converged.
        times : dict[str, list[float]]
            Time taken for different function calls in the algorithm.
        n_calls_to_functions : dict[str, list[int]]
            Number of calls to different functions in the algorithm.
        report_period : int
            Frequency of reporting the status.
        verbose : int
            Verbosity level.
        metric_tracker : MetricTracker | None
            Helper object to track metrics during optimization. Please see the
            MetricTracker class for more information.
        lipschitz_constant : float | None
            Lipschitz constant of the gradient of :math:`f`.
        step_size_strategy : str {'fixed', 'backtracking'}
            Strategy for selecting the step size. Options are:
            'fixed', 'backtracking'.
        step_sizes : list[float] | None
            Step size for the gradient step for each iteration.
    """
    @torch.no_grad()
    def __init__(self, step_size: float | None = None,
                        lipschitz_constant: float | None = None,
                        step_size_strategy: str = 'fixed',
                        device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
                        dtype: torch.dtype = torch.float64,
                        metric_tracker : MetricTracker | None = None,
                        verbose: int = 0,
                        report_period: int = 1,
                        max_iter: int = 100,
                        err_tol: float = 1e-6,
                        **kwargs):
        """Initialize the Proximal Gradient Base class.

        Parameters
        ----------
            step_size : float | None = None
                Initial step size for the gradient step. If not provided,
                it will be set to 1 / lipschitz_constant if available for
                'fixed' step size strategy. For 'backtracking' strategy, it will
                be initialized to 1.0.
            lipschitz_constant : float | None = None
                Lipschitz constant of the gradient of `f`. Used for fixed step
                size strategy.
            step_size_strategy : str = 'fixed' {'fixed', 'backtracking'}
                Strategy for selecting the step size. Options are:
                'fixed', 'backtracking'.
            device : torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu'
                Device to use for computations ('cpu' or 'cuda:{i}').
            dtype : torch.dtype = torch.float64
                Data type for computations.
            metric_tracker : MetricTracker | None = None
                Helper object to track metrics during optimization. Please see the
                MetricTracker class for more information.
            verbose : int = 0
                Verbosity level. 0: silent, 1: basic reporting, 2: detailed timing.
            report_period : int = 1
                Frequency of reporting the status.
            max_iter : int = 100
                Maximum number of iterations for the algorithm.
            err_tol : float = 1e-6
                Error tolerance for convergence.
        """
        self.lipschitz_constant = lipschitz_constant
        self.step_size_strategy = step_size_strategy

        self.device = device
        self.dtype = dtype

        self.it = 0
        self.objective = []
        self.delta_objective = []
        self.delta_x = []
        self.converged = False
        self.times = defaultdict(list)
        self.n_calls_to_functions = defaultdict(lambda : 0)
        self.step_sizes = []
        self.max_iter = max_iter
        self.err_tol = err_tol

        self.report_period = report_period
        self.verbose = verbose
        self.metric_tracker = metric_tracker

        if step_size is None and step_size_strategy == 'fixed':
            if self.lipschitz_constant is None:
                self.step_size_strategy = 'backtracking'
                warn_msg = ("Lipschitz constant not provided. "
                            "Switching to backtracking line search for step size selection.")
                warn(warn_msg, UserWarning)
            else:
                self.step_sizes.append( 1.0 / self.lipschitz_constant)
        
        if step_size_strategy == 'backtracking':
            if self.lipschitz_constant is not None:
                self.step_sizes.append( 1.0 / self.lipschitz_constant)
            else:
                self.step_sizes.append(1.0)
            self.shrinkage_factor = kwargs.get('shrinkage_factor', 0.85)

        self.x = None

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor | np.ndarray,
                     **kwargs):
        """Iterate over the algorithm until convergence or reaching maximum iterations."""
        self.converged = False
        self.err_tol = kwargs.get('err_tol', self.err_tol)
        self.max_iter += kwargs.get('max_iter', 0)

        if isinstance(x0, np.ndarray):
            self.x = torch.tensor(x0, device=self.device, dtype=self.dtype)
        elif isinstance(x0, torch.Tensor):
            if x0.is_cuda:
                self.device = f'cuda:{x0.get_device()}'
            self.x = x0.to(device=self.device, dtype=self.dtype)


        while not self.converged and self.it < self.max_iter:
            self._update()
            self._report_status()
            self._check_convergence()
            self._call_tracker()
            if self.converged:
                break
            self.it += 1
        return self.x


    def update(self):
        """Perform a single iteration update."""
        grad = self._grad_f(self.x)

        if self.step_size_strategy == 'backtracking':
            x_temp = self._backtracking_line_search(grad)
        elif self.step_size_strategy == 'fixed':

            step_size = self.step_sizes[-1]
            x_temp = self.x - step_size * grad
            x_temp = self._prox_step(x_temp, step_size)

            obj_f = self._func_f(x_temp)
            obj_g = self._func_g(x_temp)
            self.objective.append( (obj_f + obj_g).item() )

        delta_x = torch.norm((x_temp - self.x).ravel()).item()
        self.delta_x.append(delta_x)
        self.x = x_temp

    def _backtracking_line_search(self, grad):
        ss = self.step_sizes[0]
        sf = self.shrinkage_factor

        obj_current_f = self._func_f(self.x)
        while True:
            x_trial = self._prox_step(self.x - ss * grad, ss)
            generalized_grad = (self.x - x_trial) / ss

            obj_trial_f = self._func_f(x_trial)
            rhs = (obj_current_f 
                   - ss * torch.dot(grad.ravel(), generalized_grad.ravel())
                   + (ss / 2) * torch.norm(generalized_grad.ravel())**2
                   )
            if obj_trial_f <= rhs:
                break
            ss *= sf
        self.step_sizes.append(ss)
        obj_g = self._func_g(x_trial)
        self.objective.append( (obj_trial_f + obj_g).item() )
        return x_trial


    def _update(self):
        if self.verbose > 1:
            tic = perf_counter()
            self.update()
            toc = perf_counter()
            self.times['iter'].append(toc - tic)
        else:
            self.update()

    def _func_f(self, x):
        if self.verbose > 1:
            tic = perf_counter()
            obj_f = self.func_f(x)
            toc = perf_counter()
            self.times['func_f'].append(toc - tic)
            self.n_calls_to_functions['func_f'] += 1
        else:
            obj_f = self.func_f(x)
        return obj_f


    def _func_g(self, x):
        if self.verbose > 1:
            tic = perf_counter()
            obj_g = self.func_g(x)
            toc = perf_counter()
            self.times['func_g'].append(toc - tic)
            self.n_calls_to_functions['func_g'] += 1
        else:
            obj_g = self.func_g(x)
        return obj_g

    def _grad_f(self, x):
        if self.verbose > 1:
            tic = perf_counter()
            grad = self.grad_f(x)
            toc = perf_counter()
            self.times['grad_f'].append(toc - tic)
            self.n_calls_to_functions['grad_f'] += 1
        else:
            grad = self.grad_f(x)
        return grad


    def _prox_step(self, x, step_size):
        if self.verbose > 1:
            tic = perf_counter()
            x_prox = self.prox_step(x, step_size)
            toc = perf_counter()
            self.times['prox_step'].append(toc - tic)
            self.n_calls_to_functions['prox_step'] += 1
        else:
            x_prox = self.prox_step(x, step_size)
        return x_prox

    def _report_status(self):
        """Report the current status of the optimization."""
        if self.verbose > 0 and self.it % self.report_period == 0:
            msg = f"It-{self.it:05}: Objective = {self.objective[-1]:.3e}"
            if self.it > 1:
                msg += f", Delta Objective = {self.delta_objective[-1]:.3e}"
                msg += f" Delta x = {self.delta_x[-1]:.3e}"
            print(msg)

    def _check_convergence(self):
        """Check convergence based on the change in objective value."""
        if self.it == 0:
            return
        delta_obj = abs(self.objective[-1] - self.objective[-2])
        self.delta_objective.append(delta_obj)
        if delta_obj < self.err_tol and self.delta_x[-1] < self.err_tol:
            self.converged = True
            if self.verbose > 0:
                print((f"Converged at iteration {self.it} with delta "
                        f"objective {delta_obj:.3e}."))

    def _call_tracker(self):
        """Track metrics of the state variables with MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)

    @abstractmethod
    def func_f(self, x):
        """Evaluate the differentiable part of the objective function `f`."""
        pass

    @abstractmethod
    def func_g(self, x):
        """Evaluate the function `g`."""
        pass

    @abstractmethod
    def prox_step(self, x, step_size):
        """Evaluate the proximal step operation for function `g` with given step size."""
        pass

    @abstractmethod
    def grad_f(self, x):
        """Calculate the gradient of the differentiable part f."""
        pass

    def plot_alg_run(self):
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].semilogy(self.objective)
        axs[0].set_title('Objective function')
        axs[1].semilogy(self.delta_objective)
        axs[1].set_title('Delta Objective')
        axs[2].semilogy(self.delta_x)
        axs[2].set_title('Delta x')
        for ax in axs:
            ax.grid()
        return fig, axs
