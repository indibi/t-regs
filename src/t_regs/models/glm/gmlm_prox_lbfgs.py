# pylint: disable=not-callable, invalid-name
"""LBFGS wrapper for solving the GMLM Proximal step."""

from typing import Optional, Any, Callable, overload
from collections import defaultdict
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from torch.optim import LBFGS

from gmlm import GeneralizedMultiLinearModel as GMLM

class GMLMProxLBFGS:
    r"""LBFGS Wrapper solving optimization problem defined by GMLMProximal step

    Solves the optimization problem defined with,
    .. math::
        \min_{B} F(B) + \tau/2 \|B\|_F^2 + \rho/2 \|B - prox_point\|_F^2

    Parameters
    ----------
    max_steps: Optional[int] = 100
        Maximal number of steps taken by calling `torch.optim.LBFGS` optimizer.
    lr: Optional[float] = 1
        The learning rate.
    max_iter: int = 20
        maximal number of iterations per optimization step.
    max_eval: int = max_iter * 1.25
        maximal number of function evaluations per optimization step.
    tolerance_grad: float = 1e-7
        termination tolerance on first order optimality (default: 1e-7).
    tolerance_change: float = 1e-9
        termination tolerance on function value/parameter changes (default: 1e-9).
    history_size: int = 100
        update history size.
    line_search_fn: Optional[str] = 'strong_wolfe'
        Either 'strong_wolfe' or None.
    max_time:
        Maximum allowed time for the algorithm to run in seconds.
    verbosity:
        Verbosity level of the algorithm.
    log_verbosity:
        Verbosity level for logging.
    report_period:
        Period for reporting progress.
    logging_period:
        Period for logging details.
    """
    def __init__(
        self,
        max_steps: int = 100,
        lr: float = 1,
        max_iter: int = 20,
        max_eval: int = 5000,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-7,
        history_size: int = 100,
        line_search_fn: Optional[str] = 'strong_wolfe',
        max_time: float | None = None,
        verbosity: int = 0,
        log_verbosity: int = 1,
        report_period: int = 1,
        logging_period: int = 0,
        ):
        self.max_steps = max_steps
        self.lr = lr
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.max_time = max_time
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = logging_period
        self.log = None

    @overload
    def solve(self,
            B: GMLM,
            X: torch.Tensor,
            y: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            tau: float = 0,
            rho: float = 0,
            prox_point: Optional[torch.Tensor] = None,
            X_val: Optional[torch.Tensor] = None,
            y_val: Optional[torch.Tensor] = None,
            ) -> dict[str, Any]:
        pass

    @overload
    def solve(self,
            B: GMLM,
            dataloader: DataLoader,
            tau: float = 0,
            rho: float = 0,
            prox_point: Optional[torch.Tensor] = None,
            val_dataloader: DataLoader = None,
            distributed_admm: bool = False,
            ) -> dict[str, Any]:
        pass


    def solve(
        self,
        B: GMLM,
        *vargs, # X, y or Dataset/dataloader
        tau: float = 0,
        rho: float = 0,
        prox_point: Optional[torch.Tensor] = None,
        **kwargs,
        ) -> dict[str, Any]:
        start_time = perf_counter()
        self._initialize_log(tau, rho, prox_point)

        optimizer = LBFGS(B.parameters(), **self.lbfgs_cfg)
        closure = self._init_objective(
            B, optimizer, tau, rho, prox_point, vargs, kwargs
            )
        # Define closure function
        step_n = 0
        while True:
            _ = optimizer.step(closure)
            flat_grad = optimizer._gather_flat_grad() # pylint: disable=protected-access
            grad_max = flat_grad.abs().max()
            metrics = self._evaluate(B, tau, rho, prox_point, vargs, kwargs)

            objective = metrics.pop('objective')
            lbfgs_state = optimizer.state_dict()['state'][0]
            metrics['func_evals'] = lbfgs_state.get('func_evals', 0)
            metrics['grad_max'] = grad_max.item()
            self._add_log_entry(    # pylint: disable=redundant-keyword-arg
                start_time,
                step_n,
                lbfgs_state.get('n_iter'),
                objective,
                **metrics
                )
            reason = self._check_stopping_criteria(
                start_time, step_n, objective, grad_max, lbfgs_state
            )

            step_n +=1
            if reason is not None:
                if self.verbosity >=1:
                    print(reason)
                    print("")
                break
        return {
            'time': perf_counter() - start_time,
            'log': self.log,
            # 'B': B,
            'stopping_criterion': reason,
            'lbfgs_state': lbfgs_state,
            'objective': objective,
            'metrics':metrics,
            # **metrics
        }

    @torch.no_grad()
    def _evaluate(self, B: GMLM, tau, rho, prox_point, vargs, kwargs):
        results = {
            'objective': 0.0,
            'prox_sse_penalty': 0.0,
            'ridge_penalty': 0.0,
            'train_loss': 0.0,
            'val_loss': None,
            'train_score': 0.0,
            'val_score': None
        }
        if len(vargs) == 1:
            dataloader = vargs[0]
            if not isinstance(dataloader, DataLoader):
                raise TypeError(
                    "`data` must be an instance of `torch.nn.DataLoader`."
                    f" But type(dataloader)={type(dataloader)} is provided."
                    )
            loss = 0.0
            score = 0.0
            total_samples = 0
            for bdata in dataloader:
                Xb, yb, wb = bdata if len(bdata)==3 else (*bdata, None)
                total_samples += yb.shape[0]
                etab = B(Xb)
                loss += B.loss_fn(etab, yb, weights=wb)
                score += B.score(etab, yb)*yb.shape[0]
            loss /= total_samples
            score /= total_samples
            results['train_loss'] = loss.item()
            results['train_score'] = score
            objective = loss
            if (prox_point is not None) and (rho !=0):
                results['prox_sse_penalty'] = (
                     0.5*rho*torch.linalg.vector_norm(B.B-prox_point)**2
                     ).item()
                objective += results['prox_sse_penalty']
            if tau>0:
                results['ridge_penalty'] = (
                    0.5*tau*torch.linalg.vector_norm(B.B)**2
                    ).item()
                objective += results['ridge_penalty']
        elif len(vargs) == 2:
            X, y = vargs
            weights = kwargs.get('weights', None)
            total_samples = y.shape[0]
            if not (isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)):
                raise TypeError(
                    "`X` and `y` must be an instance of `torch.Tensor` but "
                    f"`(type(X),type(y)`=({type(X),type(y)} is provided."
                    )
            eta = B(X)
            loss = B.loss_fn(eta, y, weights=weights) / total_samples
            results['train_score'] = B.score(eta, y)
            results['train_loss'] = loss.item()
            objective = loss

            if (prox_point is not None) and (rho !=0):
                results['prox_sse_penalty'] = (
                    0.5*rho*torch.linalg.vector_norm(B.B-prox_point)**2
                    ).item()
                objective += results['prox_sse_penalty']
            if tau>0:
                results['ridge_penalty'] = (
                    0.5*tau*torch.linalg.vector_norm(B.B)**2
                    ).item()
                objective += results['ridge_penalty']
        else:
            raise ValueError(f"Too many arguments provided: vargs={len(vargs)}")
        results['objective'] = objective.item()

        val_dataloader = kwargs.get('val_dataloader', None)
        if val_dataloader is not None:
            loss = 0.0
            score = 0.0
            total_samples = 0
            for bdata in val_dataloader:
                Xb, yb, wb = bdata if len(bdata)==3 else (*bdata, None)
                total_samples += yb.shape[0]
                etab = B(Xb)
                loss += B.loss_fn(etab, yb, weights=wb)
                score += B.score(etab, yb)*yb.shape[0]
            loss /= total_samples
            score /= total_samples
            results['val_loss'] = loss.item()
            results['val_score'] = score
        else:
            X_val, y_val = kwargs.get('X_val', None), kwargs.get('y_val', None)
            if (X_val is not None) and (y_val is not None):
                total_samples = y_val.shape[0]
                eta = B(X_val)
                loss = B.loss_fn(eta, y_val, weights=weights) / total_samples
                results['val_score'] = B.score(eta, y_val)
                results['val_loss'] = loss.item()
        return results


    def _init_objective(self, B:GMLM, optimizer, tau, rho, prox_point, vargs, kwargs) -> Callable:
        if len(vargs) == 1:
            dataloader = vargs[0]
            if not isinstance(dataloader, DataLoader):
                raise TypeError(
                    "`data` must be an instance of `torch.nn.DataLoader`."
                    f" But type(dataloader)={type(dataloader)} is provided."
                    )
            def closure():
                optimizer.zero_grad()
                loss = 0.0
                total_samples = 0
                for bdata in dataloader:
                    Xb, yb, wb = bdata if len(bdata)==3 else (*bdata, None)
                    total_samples += yb.shape[0]
                    etab = B(Xb)
                    loss += B.loss_fn(etab, yb, weights=wb)
                loss /= total_samples
                if (prox_point is not None) and (rho !=0):
                    loss += 0.5*rho*torch.linalg.vector_norm(B.B-prox_point)**2
                if tau>0:
                    loss += 0.5*tau*torch.linalg.vector_norm(B.B)**2
                loss.backward()
                return loss
        elif len(vargs) == 2:
            X, y = vargs
            weights = kwargs.get('weights', None)
            total_samples = y.shape[0]
            if not (isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)):
                raise TypeError(
                    "`X` and `y` must be an instance of `torch.Tensor` but "
                    f"`(type(X),type(y)`=({type(X),type(y)} is provided."
                    )
            def closure():
                optimizer.zero_grad()
                eta = B(X)
                loss = B.loss_fn(eta, y, weights=weights) / total_samples
                if (prox_point is not None) and (rho !=0):
                    loss += 0.5*rho*torch.linalg.vector_norm(B.B-prox_point)**2 # pylint: disable=not-callable
                if tau>0:
                    loss += 0.5*tau*torch.linalg.vector_norm(B.B)**2 # pylint: disable=not-callable
                loss.backward()
                return loss
            return closure
        else:
            raise ValueError(f"Too many arguments provided: vargs={len(vargs)}")
        return closure

    def _initialize_log(self, tau, rho, prox_point):
        self.log = {
            'solver': str(self),
            'lbfgs_config': self.lbfgs_cfg,
            'run_params': {
                'tau': tau,
                'rho': rho,
                'prox_point': prox_point
            },
            'iterations': defaultdict(list)
            }


    def _add_log_entry(self, start_time, step, iteration, objective, **kwargs):
        if self.log_verbosity <=0:
            return
        if (self.logging_period !=0) and (step % self.logging_period ==0):
            self.log['iterations']['time'].append(perf_counter()-start_time)
            self.log['iterations']['step'].append(step)
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)

    def _check_stopping_criteria(
            self, start_time, n_steps, objective, grad_max, lbfgs_state
            ) -> str:
        run_time = perf_counter() - start_time
        reason = None
        prev_obj = lbfgs_state.get('prev_loss', None)
        change = (
            abs(objective - prev_obj) if prev_obj is not None
                                    else self.tolerance_change*2
            )
        func_evals = lbfgs_state.get('func_evals', 0) + n_steps
        if grad_max <= self.tolerance_grad:
            reason = (f"Success - tolerance_grad met: grad_max={grad_max:.2e}"
                      f"<= tolerance_grad={self.tolerance_grad:.2e}")
        elif change <= self.tolerance_change:
            reason = (f"Success - tolerance_change met: change={change:.2e}"
                      f"<= tolerance_change={self.tolerance_change:.2e}")
        elif n_steps >= self.max_steps:
            reason = (f"Stopping - max_steps reached: n_steps={n_steps}"
                      f">= max_steps={self.max_steps}")
        elif (self.max_time is not None) and (run_time >= self.max_time):
            reason = (f"Stopping - max_time reached: run_time={run_time:.2f}s"
                      f">= max_time={self.max_time:.2f}s")
        elif func_evals >= self.max_eval:
            reason = (f"Stopping - max_eval reached: func_evals={func_evals}"
                      f">= max_eval={self.max_eval}")
        return reason

    @property
    def lbfgs_cfg(self):
        """Configuration of the L-BFGS solver."""
        return {
            'lr': self.lr,
            'max_iter': self.max_iter,
            'max_eval': self.max_eval,
            'tolerance_grad': self.tolerance_grad,
            'tolerance_change': self.tolerance_change,
            'history_size': self.history_size,
            'line_search_fn': self.line_search_fn,
        }

    def __str__(self):
        name = type(self).__name__
        return name

    # def _return_result(self, start_time, B, lbfgs_state, **kwargs) -> GMLMProxLBFGSResult:
    #     log = self.log if hasattr(self, 'log') else None
    #     self.log = None
    #     return GMLMProxLBFGSResult(
    #         time=perf_counter() - start_time,
    #         log=log,
    #         B=B,
    #         lbfgs_state=lbfgs_state,
    #         **kwargs
    #         )
