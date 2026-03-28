from warnings import warn
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Any, Optional, Sequence
from time import perf_counter
from pprint import pprint

from ...models.regression.regression_base import RegressionBaseClass
from ...multilinear_ops.tensor_products import multi_mode_product as mmp
from ...multilinear_ops.matricization import matricize
from ...manifolds import Steifel, Euclidean, GeneralizedSteifel
from ...solvers.manifold import RiemmannianGradientDescent as RGD
from ...solvers.manifold.line_searcher import LineSearcher
from ...solvers.manifold.problem import Problem
from ...utils import printer


class GenTuckerBCD(RegressionBaseClass):
    r"""Sparse Tucker Regression with Riemmannian Block Coordinate Descent
    
    Parameters
    ----------
    regression_type: str
        The type of the regression model. Currently the available options are 
        `'linear'` and `'logistic'` regression.
    feature_dims:
        Dimensions of the feature tensors.
    ranks:
        Multi-linear rank of the regression coefficients, :math:`(r_1,\cdots,
        r_M)`
    tau:
        Core tensor frobenius norm regularization parameter.
    ldas:
        Sparsity regularization parameter for factor matrices :math:`U_m \in 
        \mathbb{R}^{I_m \times r_m}`
    Ds:
        Analysis or Generalized Lasso Penalty Matrix for the sparsity 
        regularization on :math:`U_m`. Defaults to identity matrix when provided
        None
    thetas:
        Smoothness regularization parameters for factor matrices :math:`U_m`
    Ls:
        Laplacian matrices to promote smoothness of the factor matrices 
        :math:`U_m`
    """
    regression_types = ['linear', 'logistic']
    algorithm_options = ['BCD_RGD', 'BCD_RADMM', 'BCD_RADA_RGD', 'BCD_RADA_PGD']
    def __init__(self,
                regression_type: str,
                feature_dims: Sequence[int],
                task_dims: Sequence[int],
                ranks: Sequence[int],
                tau: float,
                ldas: Sequence[float],
                thetas: Sequence[float],
                Ds: Sequence[torch.Tensor | None] | None = None,
                Ls: Sequence[torch.Tensor | None] | None = None,
                main_algorithm: str = 'BCD_RGD',
                max_it: int = 5000,
                min_gradient_norm: float = 1e-8,
                max_time: float | None = None,
                verbosity: int = 0,
                log_verbosity: int = 1,
                report_period: int = 1,
                logging_period: int = 1,
                subproblem_solvers: dict[str, Any] = None,
                generalized_steifel: bool = False,
                **kwargs,
                ):
        super().__init__(**kwargs)
        self.regression_type = regression_type
        self.feature_dims = feature_dims
        if len(task_dims) != 0:
            raise NotImplementedError("Multi-task regression is not implemented yet.")
        self.task_dims = task_dims
        self.coefficient_dims = tuple(list(task_dims) +
                                      list(feature_dims))
        self.ranks = ranks
        self.tau = tau
        self.ldas = ldas
        self.thetas = thetas
        self.M = len(feature_dims)
        
        self._X_modes  = [i+1 for i in range(1, self.M+1)]

        if Ds is None:
            self.Ds = [torch.eye(self.feature_dims[i],
                                        device=self.device, dtype=self.dtype)
                            for i in range(self.M)]
        
        if Ls is None:
            self.Ls = [torch.eye(self.feature_dims[i],
                                        device=self.device, dtype=self.dtype)
                            for i in range(self.M)]
        else:
            self.Ls = Ls
        
        for i in range(self.M):
            if self.Ls[i] is None:
                self.Ls[i] = torch.eye(self.feature_dims[i],
                                        device=self.device, dtype=self.dtype)

            if self.Ds[i] is None:
                self.Ds[i] = torch.eye(self.feature_dims[i],
                                        device=self.device, dtype=self.dtype)

        self.solver = None
        self.log = None

        if generalized_steifel:
            self.manifolds = [GeneralizedSteifel(self.feature_dims[i],
                                                self.ranks[i],
                                                self.Ls[i],
                                                device=self.device,
                                                dtype=self.dtype)
                           for i in range(self.M)]
        else:
            self.manifolds = [Steifel(n, p, device=self.device, dtype=self.dtype)
                           for n,p in zip(feature_dims, ranks)]
        self.manifold_C = Euclidean(self.ranks)
        self.Us = None #[manifold.random_point() for manifold in self.manifolds]
        self.C = None # self.manifold_C.random_point()

        self.main_algorithm = main_algorithm
        self.max_it = max_it
        self.min_gradient_norm = min_gradient_norm
        self.max_time = max_time
        self.verbosity = verbosity
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = logging_period
        self.subproblem_solvers = subproblem_solvers

        self._initialize_solvers()

    def _predict(self, X, y=None, return_prob=False, **kwargs):
        X_prime = mmp(X,
                    self.Us,
                    modes = self._X_modes,
                    skip_modes = [],
                    transpose = True)
        C_dot_modes = [i for i in range(self.M)]
        X_dot_modes = X_dot_modes = [i for i in range(1, self.M+1)]
        etas = torch.tensordot(X_prime, self.C, dims=[X_dot_modes, C_dot_modes])

        if self.regression_type == 'logistic':
            probs = torch.sigmoid(etas)
            if return_prob:
                return probs
            else:
                return (probs>0.5).float()

        elif self.regression_type == 'linear':
            return etas

    def _score(self, pred, Y):
        if self.regression_type == 'linear':
            ss_total = torch.sum((Y - torch.mean(Y, dim=0))**2)
            ss_residual = torch.sum((Y - pred)**2)
            r2_score = 1.0 - (ss_residual / ss_total)
            return r2_score.item()
        elif self.regression_type == 'logistic':
            accuracy = torch.sum(Y == pred).item()
            accuracy = accuracy/ Y.numel()
            return accuracy


    def _fit(self, X, y, X_val=None, y_val=None,
             Us_0: Optional[Sequence[torch.Tensor]] = None,
             C_0: Optional[torch.Tensor] = None,
             seed: Optional[int] = None):

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        self._initialize_log(seed=seed)

        if self.Us is None:
            if Us_0 is None:
                # self.Us = [self.manifolds[mode].random_point(generator=generator)
                            # for mode in range(self.M)]
                self.Us = self._initialize_U_with_hosvd_of_grad(X, y)
            else:
                self.Us = Us_0
        if self.C is None:
            if C_0 is None:
                self.C = self.manifold_C.random_point(generator=generator)
            else:
                self.C = C_0

        if self.main_algorithm == 'BCD_RGD':
            return self._BCD_RGD_fit(X, y, X_val, y_val)
        else:
            raise NotImplementedError(f"Main algorithm {self.main_algorithm} "
                                      "is not implemented yet.")

    def _initialize_U_with_hosvd_of_grad(self, X, y) -> Sequence[torch.Tensor]:
        etas = torch.zeros_like(y)
        if self.regression_type == 'logistic':
            probs = F.sigmoid(etas)
            residuals = y - probs
        elif self.regression_type == 'linear':
            residuals = y - etas
        else:
            raise NotImplementedError(("Only linear and logistic regression"
                                      " is implemented currently"))
        # grad = - residuals.sum(dim=0)/residuals.shape[0]
        y_shp = tuple([1]*len(X.shape))
        grad = -torch.sum((residuals.reshape((-1, *y_shp[1:])) * X), dim=0
                    )/residuals.shape[0]
        Us = []
        for mode in range(1, self.M+1):
            grad_m = matricize(grad, [mode])
            U, _, _ = torch.linalg.svd(grad_m, full_matrices=False)
            Us.append(U[:, :self.ranks[mode-1]])
        return Us


    def _BCD_RGD_fit(self, X, y, X_val, y_val):
        if self.verbosity >=1:
            print("Starting BCD_RGD Solver for Tucker Regression")
        if self.verbosity >=2:
            print("Hyper-parameters:")
            pprint(self.hyper_parameters)
            iteration_format_length = int(np.log10(self.max_it)) + 1
            columns = [("Iteration", f"{iteration_format_length}d"),
                       ("Train score", ".4f")]
            if X_val is not None and y_val is not None:
                columns += [("Val score", ".4f")]
            columns += [("Cost", ".5e"),
                        ("Gradient norm", ".5e"),
                        ] + [
                    (f"U_{mode} grad norm", ".5e") for mode in range(1, self.M+1)
                ] + [("C grad norm", ".5e")]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()
        start_time = perf_counter()


        modes = list(range(1, self.M+1))
        it = 0
        while True:
            it += 1
            solver = self.subproblem_solvers['C']
            problem = self._initialize_C_subproblem_for_RGD(X, y,
                                                        return_lipschitz=True)
            if problem.lipschitz_constant is not None:
                solver.step_size = 1.0 / problem.lipschitz_constant

            rgd_result = solver.solve(problem, x0=self.C)
            self.solver_results['C'] = rgd_result
            self.C = rgd_result.point

            for _ in range(1):
                for mode in modes:
                    U = self.Us[mode-1]
                    solver = self.subproblem_solvers[f'U_{mode}']
                    problem = self._initialize_U_subproblem_for_RGD(X, y, mode,
                                                        return_lipschitz=True)
                    if problem.lipschitz_constant is not None:
                        solver.step_size = 1.0 / problem.lipschitz_constant

                    rgd_result = solver.solve(problem, x0=U)
                    self.solver_results[f'U_{mode}'] = rgd_result
                    self.Us[mode-1] = rgd_result.point

                    # func_f, grad_f, L = self._initialize_C_subproblem_for_RGD(
                    #     X,
                    #     y,
                    #     return_lipschitz=True
                    #     )
                    # solver = self.subproblem_solvers['C']
                    # solver.step_size = 1.0 / L
                    # rgd_result = solver.solve(func_f,
                    #                     grad_f,
                    #                     manifold=self.manifold_C,
                    #                     x0=self.C)
                    # self.solver_results['C'] = rgd_result
                    # self.C = rgd_result.point

            # Calculate values for logging
            with torch.no_grad():
                train_pred = self.predict(X)
                scores = {"train_score": float(self.score(train_pred, y))}
                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_score = float(self.score(val_pred, y_val))
                    scores["val_score"] = val_score

            # Check convergence
            problem = self._initialize_C_subproblem_for_RGD(X, y,
                                                        return_lipschitz=False)
            obj_val = problem.objective(self.C, backward_pass=True)
            grad_C = problem.grad(self.C, repeat_forward=False)
            grad_C_norm = problem.manifold.norm(self.C, grad_C)
            grad_norms = {"grad_C": float(grad_C_norm)}
            for mode in modes:
                U = self.Us[mode-1]
                problem = self._initialize_U_subproblem_for_RGD(X, y, mode)
                grad_U = problem.grad(U, repeat_forward=True)
                grad_U_norm = self.manifolds[mode-1].norm(U, grad_U)
                grad_norms[f'grad_U_{mode}'] = float(grad_U_norm)
            max_grad_norm = max(grad_norms.values())

            row = [it, scores['train_score']]
            if X_val is not None and y_val is not None:
                row += [scores['val_score']]
            row += [obj_val, max_grad_norm]
            row += [grad_norms[f'grad_U_{mode}'] for mode in range(1, self.M+1)]
            row += [grad_norms['grad_C']]
            column_printer.print_row(row)
            self._add_log_entry(start_time,
                                it,
                                obj_val,
                                gradient_norm=max_grad_norm,
                                **scores,
                                **grad_norms,
                                )

            reason = self._check_stopping_criteria(start_time,
                                                  it,
                                                  max_grad_norm)
            if reason is not None:
                if self.verbosity >=1:
                    print(reason)
                break


    def _initialize_solvers(self):
        # ['BCD_RGD', 'BCD_RADMM', 'BCD_RADA_RGD', 'BCD_RADA_PGD']
        ma = self.main_algorithm
        if ma not in self.algorithm_options:
            raise ValueError((f"Unknown main_algorithm {ma}. "
                              f"Available options are {self.algorithm_options}."
                              ))
        if ma == 'BCD_RGD':
            if (self.ldas is not None) and any(lda for lda in self.ldas):
                warn ("Sparsity regularization is ignored for BCD_RGD solver")
            if self.subproblem_solvers is None:
                self.subproblem_solvers = {}
                for mode in range(1, self.M+1):
                    self.subproblem_solvers[f'U_{mode}'] = RGD(
                                        line_searcher = LineSearcher(
                                            tau=0.7793,
                                            c_1=0.95,
                                        ),
                                        min_gradient_norm = 1e-8,
                                        max_time = self.max_time/((self.M+1)*10),
                                        # verbosity = self.verbosity-1,
                                        log_verbosity = max(self.log_verbosity,1),
                                        )
                self.subproblem_solvers['C'] = RGD(
                                        line_searcher = LineSearcher(
                                            tau=0.7793,
                                            c_1=0.95,
                                        ),
                                        min_gradient_norm = 1e-8,
                                        max_time = self.max_time/((self.M+1)*10),
                                        # verbosity = self.verbosity-1,
                                        log_verbosity = max(self.log_verbosity,1),
                                        )
        elif ma == 'BCD_RADMM':
            raise NotImplementedError("BCD_RADMM is not implemented yet.")
        elif ma == 'BCD_RADA_RGD':
            raise NotImplementedError("BCD_RADA_RGD is not implemented yet.")
        elif ma == 'BCD_RADA_PGD':
            raise NotImplementedError("BCD_RADA_PGD is not implemented yet.")


    def _initialize_U_subproblem_for_RGD(self, X, y, mode, return_lipschitz=False):
        """Initializes U_{mode} subproblem for Riemannian Gradient Solver
        
        Returns
        -------
        func_f:
        grad_f:
        """
        # TODO: Deal with multi-task regression
        regression_type = self.regression_type
        theta = self.thetas[mode-1]
        L = self.Ls[mode-1]
        lipschitz_const = None
        with torch.no_grad():
            X_prime = mmp(X,
                        self.Us,
                        modes = self._X_modes,
                        skip_modes = [mode+1],
                        transpose = True)
            C_dot_modes = [i for i in range(self.M) if i != mode-1]
            X_dot_modes = [i for i in range(1, self.M+1) if i != mode]
            CX = torch.tensordot(X_prime,
                                 self.C,
                                 dims=[X_dot_modes, C_dot_modes])
        def objective(U):
            etas = torch.tensordot(CX, U, dims=[[1,2], [0,1]])
            if regression_type == 'logistic':
                loss = F.binary_cross_entropy_with_logits(etas,
                                                            y,
                                                            reduction='mean')
            elif regression_type == 'linear':
                residuals = y - etas
                loss = (residuals**2).sum()/residuals.numel()
            else:
                raise NotImplementedError((
                    "Currently only `logistic` and `linear` regression is"
                    " implemented."
                ))

            # tr(U^T L U) = <U, L U>
            if ((theta is not None ) or (theta != 0)):
                drichlet_energy = 0.5*theta*( U*(L@U) ).sum()
                loss += drichlet_energy
            return loss

        with torch.no_grad():
            if return_lipschitz:
                N = y.shape[0]
                x = matricize(CX, [1])
                batch_sum_out_prod = torch.einsum('bi, bj->ij', x, x)/N
                lipschitz_const = torch.linalg.norm(batch_sum_out_prod, 2)
                if regression_type == 'linear':
                    lipschitz_const += theta*float((L**2).sum())
                elif regression_type == 'logistic':
                    lipschitz_const = (0.25*lipschitz_const
                                       + theta*float((L**2).sum())
                                       )
        problem = Problem(self.manifolds[mode-1],
                          objective,
                          lipschitz_const=lipschitz_const
                          )
        return problem

    def _initialize_C_subproblem_for_RGD(self, X, y, return_lipschitz=False):
        lipschitz_const = None
        with torch.no_grad():
            X_prime = mmp(X,
                        self.Us,
                        modes = self._X_modes,
                        skip_modes = [],
                        transpose = True)
        C_dot_modes = [i for i in range(self.M)]
        X_dot_modes = [i for i in range(1, self.M+1)]
        tau = self.tau
        regression_type = self.regression_type
        def objective(C):
            etas = torch.tensordot(X_prime, C, dims=[X_dot_modes, C_dot_modes])
            if regression_type == 'logistic':
                loss = F.binary_cross_entropy_with_logits(etas,
                                                            y,
                                                            reduction='mean')
            elif regression_type == 'linear':
                residuals = y - etas
                loss = (residuals**2).sum()/residuals.numel()
            if (tau !=0):
                loss += tau*0.5*(C**2).sum()
            return loss

        if return_lipschitz:
            N = y.shape[0]
            x = matricize(X_prime, [1])
            batch_sum_out_prod = torch.einsum('bi, bj->ij', x, x)/N
            lipschitz_const = torch.linalg.norm(batch_sum_out_prod, 2)
            if regression_type == 'linear':
                lipschitz_const += tau
            elif regression_type == 'logistic':
                lipschitz_const = 0.25* lipschitz_const + tau
        problem = Problem(self.manifold_C,
                          objective,
                          lipschitz_const=lipschitz_const)
        return problem

    def _initialize_log(self, seed=None):
        self.hyper_parameters = {
                'regression_type': self.regression_type,
                'feature_dims': self.feature_dims,
                'task_dims': self.task_dims,
                'coefficient_dims': self.coefficient_dims,
                'ranks': self.ranks,
                'tau': self.tau,
                'ldas': self.ldas,
                'thetas': self.thetas,
            }
        self.log = {
            'hyper_parameters': self.hyper_parameters,
            'algorithm': self.main_algorithm,
            'stopping_criteria': {
                'max_time': self.max_time,
                'max_it': self.max_it,
                'min_gradient_norm': self.min_gradient_norm,
            },
            'subproblem_solver_params': {key: solver.get_parameters()
                                        for key, solver in
                                        self.subproblem_solvers.items()},
            'seed': seed,
            'iterations': defaultdict(list)
        }

    def _add_log_entry(self, start_time, iteration, objective, **kwargs):
        if self.log_verbosity <=0:
            return
        if (self.logging_period !=0) and (iteration % self.logging_period ==0):
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['time'].append(perf_counter() - start_time)
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)


    def _check_stopping_criteria(self,
                             start_time,
                             iteration,
                             gradient_norm
                             ):
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
        return reason
