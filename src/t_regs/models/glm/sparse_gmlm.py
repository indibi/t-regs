# pylint: disable=not-callable, invalid-name
from typing import Optional, Sequence, overload, Any
from collections import defaultdict
from time import perf_counter
from pprint import pprint
import math

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softshrink

from gmlm import GeneralizedMultiLinearModel as GMLM
from gmlm_prox_lbfgs import GMLMProxLBFGS
from t_regs.utils import printer

# TODO: Add a descriptive docstring.
class SparseGMLM(GMLM):
    r"""Sparse Generalized Multi-linear Model"""
    def __init__(self,
        regression_type: str,
        feature_dims: Sequence[int],
        lda: float,
        tau: float,
        rho: float,
        ws: Optional[torch.Tensor] = None,  # Sparsity weights/mask
        task_dims: Optional[Sequence[int]] = None,
        coeff: Optional[torch.Tensor] = None,
        max_it: Optional[int] = 250,
        max_time: Optional[float] = 60,
        eps_abs: Optional[float] = 1e-8,
        eps_rel: Optional[float] = 1e-4,
        verbosity: int = 0,
        log_verbosity: int = 1,
        report_period: int = 1,
        logging_period: int = 1,
        proxsolver_cfg: Optional[dict[str, Any]] = None
        ):
        super().__init__(
            regression_type,
            feature_dims,
            task_dims=task_dims,
            coeff=coeff
            )
        self.lda = lda
        self.tau = tau
        self.ws = ws    # Sparsity Weights
        self.rho = rho
        self.max_it = max_it
        self.max_time = max_time
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.logging_period = logging_period
        self.report_period = report_period
        # Initialize proxsolver
        self._proxsolver = GMLMProxLBFGS(**proxsolver_cfg)
        self._proxsolver_result = None
        self.max_eval = self._proxsolver.max_eval
        self.log = None

        self.B_s = GMLM(
            regression_type,
            feature_dims,
            task_dims=task_dims,
            coeff=torch.zeros_like(self.B.data),
            )
        self.B_s.requires_grad_(False)
        self._gamma = torch.nn.Parameter(
            data=torch.zeros_like(self.B.data), requires_grad=False
            )

    @property
    def hyperparameters(self):
        return {
            'regression_type': self.regression_type,
            'feature_dims': self.feature_dims,
            'task_dims': self.task_dims,
            'dims': self.dims,
            'lda': self.lda,
            'tau': self.tau,
            'rho': self.rho,
            'ws': self.ws,
        }

    def _fit(self,
        use_dloader:bool,
        dataloader: DataLoader = None,
        val_dataloader: Optional[DataLoader] = None,
        X:torch.Tensor=None,
        y:torch.Tensor=None,
        weights: torch.Tensor=None,
        X_val:Optional[torch.Tensor]=None,
        y_val:Optional[torch.Tensor]=None,
        ):
        st = perf_counter()
        self._initialize_log()
        # Initialize ADMM parameters
        B_old = self.B.data.clone().detach()

        column_printer = self._init_printer()
        column_printer.print_header()

        it = 0
        func_evals = 0
        while True:
            # Update B_s
            with torch.no_grad():
                if self.ws is None:
                    self.B_s.B.copy_(
                        softshrink(
                            (self.B_s.B.data - self._gamma/self.rho),
                            self.lda/self.rho
                            ) # / (1+self.tau/self.rho)
                        )
                else:
                    raise NotImplementedError("Weighted l1 norm is not implemented yet.")

            # Update B_old
            B_old.copy_(self.B.data)
            with torch.no_grad():
                prox_point = self.B_s.B.data + self._gamma/self.rho
            if use_dloader:
                result = self._proxsolver.solve(
                    self, dataloader, tau=self.tau, rho=self.rho,
                    prox_point=prox_point, val_dataloader=val_dataloader
                )
            else:
                result = self._proxsolver.solve(
                    self, X, y, weights=weights,
                    rho=self.rho, tau=self.tau, prox_point=prox_point,
                    X_val=X_val, y_val=y_val
                )
            self._proxsolver_result = result
            # ----------------- Update dual variable ----------------------
            with torch.no_grad():
                r_kp1 = self.B_s.B.data - self.B.data
                self._gamma += self.rho*r_kp1
                s_kp1 = B_old - self.B.data
                xnorm = float(torch.linalg.vector_norm(self.B_s.B.data))
                znorm = float(torch.linalg.vector_norm(self.B.data))
                rel_eps_dual = float(torch.linalg.vector_norm(self._gamma))
                rel_eps_pri = max(xnorm, znorm)

            # ----------- Check convergence and log metrics ---------------
            metrics = result['metrics']
            pri_res = torch.linalg.vector_norm(r_kp1)
            dual_res = torch.linalg.vector_norm(s_kp1)

            fid_grad_max = metrics['grad_max']
            func_evals += metrics['func_evals'] #result['log']['iterations']['func_evals'][-1]

            cost = metrics['train_loss']
            cost += metrics['ridge_penalty']
            if self.ws is None:
                cost += self.lda*(self.B_s.B.data.abs().sum())
            else:
                raise NotImplementedError(
                    "Weighted l1 norm is not implemented yet."
                    )

            row = [it, metrics['train_score'], metrics['val_score'],
                   cost]

            if self.verbosity >=3:
                row.append(None) # ("Train score (B_s)", ".4f"),
                row.append(None) # ("Val score (B_s)", ".4f"),
            row += [pri_res, dual_res, fid_grad_max]
            for i,r in enumerate(row):
                if r is None:
                    row[i] = 0

            column_printer.print_row(row)
            reason = self._check_stopping_criteria(
                st, it, func_evals, pri_res, dual_res, fid_grad_max,
                rel_eps_pri, rel_eps_dual
                )

            it +=1

            self._add_log_entry(st, it, cost,
                pri_res=pri_res, dual_res=dual_res,
                **metrics,
                )
            if reason is not None:
                if self.verbosity >=1:
                    print(reason)
                    print("")
                break


    @overload
    def fit(self,
        X:torch.Tensor,
        y:torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        X_val:Optional[torch.Tensor]=None,
        y_val:Optional[torch.Tensor]=None,
        ):
        pass

    @overload
    def fit(self,
        dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        ):
        pass

    def fit(self, *vargs, **kwargs):
        if len(vargs) == 1 and isinstance(vargs[0], DataLoader):
            use_dloader = True
            dataloader = vargs[0]
            val_dataloader = kwargs.get('val_dataloader', None)
            X, y = None, None
            X_val, y_val = None, None
            weights = None
        elif (
            (len(vargs) == 2)
            and isinstance(vargs[0], torch.Tensor)
            and isinstance(vargs[1], torch.Tensor)
            ):
            use_dloader = False
            X, y = vargs
            weights = kwargs.get('weights', None)
            X_val = kwargs.get('X_val', None)
            y_val = kwargs.get('y_val', None)
            dataloader, val_dataloader = None, None
        else:
            raise ValueError("") # TODO: add descriptive error message
        self._fit(
            use_dloader=use_dloader,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            weights=weights,
            X=X, y=y,
            X_val=X_val, y_val=y_val,
            )


    def _add_log_entry(self, st, it, obj, **kwargs):
        if self.log_verbosity <=0:
            return
        if (self.logging_period !=0) and (it % self.logging_period ==0):
            self.log['iterations']['iteration'].append(it)
            self.log['iterations']['time'].append(perf_counter() - st)
            self.log['iterations']['objective'].append(obj)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)


    def _check_stopping_criteria(
        self,
        st,
        it,
        func_evals,
        pri_res,
        dual_res,
        fid_grad_max,   # TODO: Add check for fidelity terms gradient.
        rel_eps_pri,
        rel_eps_dual
        ) -> str | None:
        sqrt_p = self.B.data.numel()**0.5
        rt = perf_counter()-st
        reason=None

        eps_pri = (sqrt_p*self.eps_abs + rel_eps_pri*self.eps_rel)
        eps_dual = (sqrt_p*self.eps_abs + rel_eps_dual*self.eps_rel)
        if (pri_res <= eps_pri) and (dual_res <= eps_dual):
            reason = ("Success - min primal and dual residual magnitude reached"
                f" with pri_res = {eps_pri:.3e} <= {eps_pri:.3e}"
                f" and dual_res = {dual_res:.3e} <= {eps_dual:.3e}")
        elif (self.max_time is not None) and (rt >= self.max_time):
            reason = f"Terminated - max time reached after {it} iterations."
        elif it>= self.max_it:
            reason = ("Terminated - maximum number of iterations reached after "
                      f"{rt:.3f} seconds.")
        elif func_evals >= self.max_eval:
            reason = ("Terminated - maximum number of func evals reached after "
                      f"{rt:.3f} seconds.")
        return reason

    def _init_printer(self):
        if self.verbosity >=1:
            print("Starting Sparse GMLM Solver")
        if self.verbosity >=2:
            print("Hyper-parameters:")
            pprint(self.hyperparameters)
            iteration_format_length = int(math.log(self.max_it, 10)) + 1
            columns = [
                ("Iteration", f"{iteration_format_length}d"),
                ("Train score", ".4f"),
                ("Val score", '.4f'),
                ('Cost', ".10e"),
                ]
            if self.verbosity >=3:
                columns += [
                    ("Train score (B_s)", ".4f"),
                    ("Val score (B_s)", ".4f"),
                ]
            columns += [
                ("||r||", ".4e"),
                ("||s||", ".4e"),
                ("||grad F(B)||", ".4e"),
            ]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        return column_printer

    def _initialize_log(self):
        self.log = {
            'hyper_parameters': self.hyperparameters,
            'algorithm': 'ADMM-LBFGS',
            'stopping_criteria': {
                'max_time': None,
                'max_it': None,
                'eps_rel': self.eps_rel,
                'eps_abs': self.eps_abs,
                'max_eval': self.max_eval,
            },
            'prox_lbfgs_params': self._proxsolver.lbfgs_cfg,
            'iterations': defaultdict(list)
        }
