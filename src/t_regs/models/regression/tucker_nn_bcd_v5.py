"""Regularized Riemannian Block Coordinate Descent optimizer module for TuckerRegressor"""

from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional, Sequence, Tuple, overload
from time import perf_counter
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

from .tucker_nn_regressor import TuckerRegressor, TuckerCovariateTransform
from ...models.regression.regression_base import RegressionBaseClass
from ...multilinear_ops.matricization import matricize
from ...multilinear_ops.tensor_products import multi_mode_product as mmp
from ...solvers.manifold.rada_v3 import RADA_RGD
from ...solvers.manifold.minimax_problem import MinimaxProblem
from ...solvers.manifold.problem import Problem
from ...utils import printer



class TuckerRegressorBCD(RegressionBaseClass):
    r"""Sparse, Generalized Tucker Regression with Block Coordinate Descent

    Parameters
    ----------
    tucker_regressor: TuckerRegressor
        :class:`TuckerRegressor` object with the regression coefficients.
    tau:
        Ridge regularization parameter, i.e. core tensor frobenius norm
        regularization.
    ldas:
        Sparsity regularization parameter for feature direction matrices
        :math:`U_m,\;\mathrm{for}\;m=1,...,N`.
    thetas:
        Smoothness regularization parameters for factor matrices :math:`U_m`
    Ds:
        Analysis or Generalized Lasso Penalty Matrix for the sparsity 
        regularization on :math:`U_m`. Defaults to identity matrix when provided
        None
    Ls:
        Laplacian matrices to promote smoothness of the factor matrices 
        :math:`U_m`
    min_gradient_norm: float = 1e-8,
       Terminate when the smallest of the norms of riemannian gradients of the
       blocks of variables become smaller than `min_gradient_norm`.
    max_it: int = 100,
    max_time: float | None = None,
        Terminate when algorithm has been going for `max_time` longer than.
    subproblem_solvers: dict[str, Any] = None
    lbfgs_config:

    verbosity: int = 0,
    log_verbosity: int = 1,
    report_period: int = 1,
    logging_period: int = 1,
    dataloader_cfg: Optional[Dict[str, Any]]
        configuration dictionary that will be passed down to the :class:`Data
        Loader`.
    init_with_hosvd_of_grad: bool = True
        Initialize the feature directions, i.e. the factor matrices with the
        Higher-order SVD of the gradient of the full coefficients.
    transform_covariates: bool = True
        Transform covariate samples for subproblems.
    """
    def __init__(
        self,
        tucker_regressor: TuckerRegressor,
        tau: float,
        ldas: Sequence[float],
        thetas: Sequence[float],
        Ds: Sequence[torch.Tensor | None] | None = None,
        Ls: Sequence[torch.Tensor | None] | None = None,
        min_gradient_norm: float = 1e-5,
        max_it: int = 100,
        max_time: float | None = None,
        subproblem_solvers: dict[str, Any] = None,
        lbfgs_config: dict[str,Any] = None,
        verbosity: int = 0,
        log_verbosity: int = 1,
        report_period: int = 1,
        logging_period: int = 1,
        dataloader_cfg: dict[str, Any] = None,
        init_with_hosvd_of_grad: bool = False,
        init_with_lbfgs_full_glm: bool = True,
        transform_covariates: bool = True,
        fit_intercept: bool = False,
        wandb_kwargs: Optional[dict] = None,
        wandb_kwarg: Optional[dict] = None,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.wandb_kwargs = wandb_kwargs if wandb_kwargs is not None else wandb_kwarg
        # --------- Hyper-parameters -----------
        self.tucker_regressor = tucker_regressor
        self.tau = tau
        self.ldas = ldas
        self.thetas = thetas
        self.N = self.tucker_regressor.covariant_degree # pylint: disable=invalid-name
        self.M = self.tucker_regressor.contravariant_degree # pylint: disable=invalid-name
        if Ds is None:
            self.Ds = [torch.eye(   # pylint: disable=invalid-name
                self.tucker_regressor.feature_dims[i],
                device=self.device,
                dtype=self.dtype
                ) for i in range(self.N)
            ]
        else:
            self.Ds = Ds    # pylint: disable=invalid-name
        self.Ls = Ls    # pylint: disable=invalid-name
        # >>>>>>>>>> Convergence settings <<<<<<<<<<
        self.max_it = max_it
        self.min_gradient_norm = min_gradient_norm
        self.max_time = max_time
        self.verbosity = verbosity
        self.subproblem_solvers = subproblem_solvers
        self.lbfgs_config = {} if lbfgs_config is None else lbfgs_config
        # >>>>>>>>>> Logging <<<<<<<<<<
        self.log = None
        self.hyper_parameters = None
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = logging_period
        # -- Fitting scheme configuration --
        self.dataloader_cfg = {} if dataloader_cfg is None else dataloader_cfg
        self.init_with_hosvd_of_grad = init_with_hosvd_of_grad
        self.init_with_lbfgs_full_glm = init_with_lbfgs_full_glm
        self.transform_covariates = transform_covariates
        self.fit_intercept = fit_intercept
        self._initialize_solvers()
        self._t_loss = None
        self._v_loss = None
        self._t_score = None
        self._v_score = None
        self._wandb_run = None




    def _BCD_fit(   # pylint: disable=invalid-name
        self,
        use_dloader:bool,
        train_dataset: Dataset = None,
        val_dataset: Optional[Dataset] = None,
        X:torch.Tensor=None,    # pylint: disable=invalid-name
        y:torch.Tensor=None,    # pylint: disable=invalid-name
        X_val:Optional[torch.Tensor]=None,  # pylint: disable=invalid-name
        y_val:Optional[torch.Tensor]=None,  # pylint: disable=invalid-name
        seed: int = None,
        wandb_kwargs: dict = None,
        ):
        if wandb_kwargs is None:
            wandb_kwargs = self.wandb_kwargs
        self._initialize_log(
            seed=seed,
            wandb_kwargs = wandb_kwargs,
            )
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        column_printer = self._init_printer(
            with_val= X_val is not None or val_dataset is not None
            )
        column_printer.print_header()
        start_time = perf_counter()
        # modes = list(range(1, self.N+1))
        modes = [n-self.tucker_regressor.contravariant_degree
                 for n in self.tucker_regressor.lr_feature_modes]
        it = 0
        while True:
            it += 1
            # >>>>>>>>>> Solve Factor Subproblems <<<<<<<<<<
            for i, mode in enumerate(modes):
                U = self.tucker_regressor.Us[i] # pylint: disable=invalid-name
                solver = self.subproblem_solvers[f'U_{mode}']
                mnmx_problem, _ = self._initialize_U_subproblem(
                    use_dloader, mode, dataset=train_dataset, X=X, y=y
                    )
                if self.solver_results[f'U_{mode}'] is not None:
                    y0 = self.solver_results[f'U_{mode}'].point.y
                else:
                    y0 = None
                rada_result = solver.solve(mnmx_problem,
                    x0=self.tucker_regressor.Us[i],
                    y0= y0,
                    seed=seed
                    )
                self.solver_results[f'U_{mode}'] = rada_result
                self.tucker_regressor.Us[i].requires_grad = False
                self.tucker_regressor.Us[i].copy_(
                    rada_result.point.x.data
                    )
             # >>>>>>>>>> Solve Core Tensor Subproblem <<<<<<<<<<
            mnmx_problem, _ = self._initialize_C_subproblem(
                use_dloader,
                dataset=train_dataset,
                X=X,
                y=y,
                )
            opt_state = self._solve_core_lbfgs(mnmx_problem)
            self.solver_results['C'] = {'state': opt_state}
            # >>>>>>>>>> Evaluate objectives <<<<<<<<<<
            objectives = self._evaluate_objectives(
                use_dloader,
                dataset = train_dataset,
                X = X,
                y = y,
                X_val = X_val,
                y_val = y_val,
                val_dataset = val_dataset,
            )
            # >>>>>>> Check Gradients for Convergence <<<<<<<<
            problem, _ = self._initialize_C_subproblem(
                use_dloader,
                dataset=train_dataset,
                X=X,
                y=y
                )
            grad_C = problem.grad_f(    # pylint: disable=invalid-name
                self.tucker_regressor.core,
                repeat_forward=True
                )
            grad_C_norm = torch.linalg.vector_norm(grad_C) # pylint: disable=not-callable, invalid-name
            grad_norms = {"grad_C": float(grad_C_norm)}

            for i, mode in enumerate(modes):
                U = self.tucker_regressor.Us[i]
                problem, _ = self._initialize_U_subproblem(
                    use_dloader,
                    mode,
                    dataset=train_dataset,
                    X=X,
                    y=y,
                    )
                grad_U = problem.grad_f(U, repeat_forward=True)
                with torch.no_grad():
                    Uy = problem.prox_h(problem.A(U))
                    nabla_AT = problem.nabla_AT(U)
                    grad_U = grad_U + nabla_AT@Uy
                    grad_U_norm = problem.manifold.norm(U, grad_U)
                grad_norms[f'grad_U_{mode}'] = float(grad_U_norm)
            max_grad_norm = max(grad_norms.values())
            # >>>>>> Report ------------>
            row = [it, objectives['train_score']]
            if ((X_val is not None and y_val is not None)
                or val_dataset is not None):
                row += [objectives['val_score']]
                row += [objectives['val_loss']]
            row += [objectives['objective'], max_grad_norm]
            row += [grad_norms[f'grad_U_{mode}'] for mode in modes]
            row += [grad_norms['grad_C']]
            column_printer.print_row(row)
            self._add_log_entry(start_time, it,
                gradient_norm=max_grad_norm,
                **objectives,
                **grad_norms,
                )
            reason = self._check_stopping_criteria(start_time, it, grad_norms)
            if reason is not None:
                if self.verbosity >=1:
                    print(reason)
                break
        if self._wandb_run is not None:
            self._wandb_run.finish()

    @overload
    def fit(self,
        X:torch.Tensor,
        y:torch.Tensor,
        X_val:Optional[torch.Tensor]=None,
        y_val:Optional[torch.Tensor]=None
        ):
        pass

    @overload
    def fit(self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        ):
        pass

    def fit(self, *vargs, **kwargs):
        if len(vargs) == 1 and isinstance(vargs[0], Dataset):
            use_dloader = True
            train_dataset = vargs[0]
            val_dataset = kwargs.get('val_dataset', None)
            X, y = None, None
            X_val, y_val = None, None
        elif (
            (len(vargs) == 2)
            and isinstance(vargs[0], torch.Tensor)
            and isinstance(vargs[1], torch.Tensor)
            ):
            use_dloader = False
            X, y = vargs
            X_val = kwargs.get('X_val', None)
            y_val = kwargs.get('y_val', None)
            train_dataset, val_dataset = None, None
        else:
            raise ValueError("") # TODO: add descriptive error message
        if self.init_with_lbfgs_full_glm:
            self._init_with_lbfgs_full_glm(
                use_dloader=use_dloader,
                train_dataset=train_dataset,
                X=X, y=y,
                )
        elif self.init_with_hosvd_of_grad:
            self._initialize_Us_with_hosvd_of_grad(
                use_dloader=use_dloader,
                train_dataset=train_dataset,
                X=X, y=y,
                )
        seed = kwargs.get('seed', 0)
        wandb_kwargs = kwargs.get('wandb_kwargs', self.wandb_kwargs)
        self._BCD_fit(
            use_dloader=use_dloader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            X=X, y=y,
            X_val=X_val, y_val=y_val,
            seed=seed,
            wandb_kwargs=wandb_kwargs,
            )



    @torch.no_grad()
    def _predict(self, X:torch.Tensor, y=None, fw_mode='full', mode_n=None
            ) -> torch.Tensor:
        return self.tucker_regressor.predict(X, y, fw_mode, mode_n)

    @torch.no_grad()
    def _score(self, pred:torch.Tensor, Y:torch.Tensor) -> float:
        return self.tucker_regressor.score(pred, Y.reshape(-1,1))


    def _init_with_lbfgs_full_glm(
        self,
        use_dloader: bool,
        train_dataset: Optional[Dataset] = None,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        ):
        """Initializes the core and the factor matrices with Truncated HoSVD"""
        if self.verbosity > 0:
            print("Initializing with HoSVD Truncation of GLM LBFGS solution")
        B = torch.zeros(
            self.tucker_regressor.dims, # pylint: disable=invalid-name
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
            )
        params = [B]
        if self.fit_intercept:
            intercept = torch.zeros(
                self.tucker_regressor.task_dims,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True
                )
            params.append(intercept)
        optimizer = torch.optim.LBFGS(
            params,
            **self.lbfgs_config
            )
        tau = self.tau
        def closure():
            optimizer.zero_grad()
            loss = 0
            ridge_penalty=0
            if tau > 0:
                ridge_penalty = (B**2).sum()*tau/2
                ridge_penalty.backward()
            if use_dloader:
                dloader = DataLoader(train_dataset, **self.dataloader_cfg)
                nsamples = 0
                grad = 0
                for _, batch_samples in tqdm(enumerate(dloader),
                                            desc="Batches: ",
                                            total=len(dloader),
                                            ):
                    Xb = batch_samples['X'].to(self.device, non_blocking=True)
                    eta = torch.tensordot(Xb,B,
                            dims=(
                                [n+1 for n in range(self.N)],
                                [n+self.M for n in range(self.N)],
                            ),
                        )
                    if self.fit_intercept:
                        eta += intercept
                    loss_b = self.tucker_regressor.loss_fn(
                        eta,
                        batch_samples['Y'].to(
                            self.device, non_blocking=True
                            ),
                        # weights=weights
                        )
                    loss_b.backward()
                    loss += loss_b.item()
                    bnsamples = Xb.shape[0]
                    nsamples += bnsamples
                loss = loss / nsamples
            else:
                eta = torch.tensordot(X,B,
                        dims=(
                            [n+1 for n in range(self.N)],
                            [n+self.M for n in range(self.N)],
                        ),
                    )
                if self.fit_intercept:
                    eta += intercept
                loss = self.tucker_regressor.loss_fn(eta, y) / X.shape[0]
                loss.backward()
            cost = loss + ridge_penalty
            return cost
        optimizer.step(closure)
        if self.fit_intercept:
            self.tucker_regressor.intercept.requires_grad = False
            self.tucker_regressor.intercept.copy_(intercept.detach())
        # v_modes = [n+1 for n in range(self.M)]
        # u_modes = [n+self.M+1 for n in range(self.N)]
        u_modes = self.tucker_regressor.lr_feature_modes
        Us = []
        for i, m in enumerate(u_modes):
            Bm = matricize(B, [m])
            # U, _, _ = torch.linalg.svd(Bm, full_matrices=True)
            Q, R = torch.linalg.qr(Bm, mode='reduced')  # pylint: disable=not-callable
            if Q.shape[-1] < self.tucker_regressor.ranks[m-1]:
                Q,R = torch.linalg.qr(Bm, mode='complete')  # pylint: disable=not-callable
            rdiag = torch.diag(R)
            if len(rdiag) < self.tucker_regressor.ranks[m-1]:
                rdiag = torch.concat(
                    [rdiag,
                     torch.zeros(self.tucker_regressor.ranks[m-1]-len(rdiag),
                        device=Bm.device, dtype=Bm.dtype)
                     ]
                    )
            rsign = torch.sign(rdiag)
            rsign[rsign==0] = 1
            order_idx = torch.argsort(rdiag.abs(), descending=True)
            top_r_idx = order_idx[:self.tucker_regressor.ranks[m-1]]
            U = Q[:,top_r_idx]*(rsign[top_r_idx].reshape(1,-1))
            Us.append(U.detach())
            Us[-1].requires_grad = False
            self.tucker_regressor.Us[i].requires_grad = False
            self.tucker_regressor.Us[i].copy_(Us[-1])

        Vs = []
        v_modes = self.tucker_regressor.lr_task_modes
        for i,m in enumerate(v_modes):
            Bm = matricize(B, [m])
            V, _, _ = torch.linalg.svd(Bm, full_matrices=True)  # pylint: disable=not-callable
            # Vs.append(
            #     V[:, :self.tucker_regressor.task_ranks[i]].detach()
            #     )
            Vs.append(
                V[:, :self.tucker_regressor.ranks[m-1]].detach()
                )
            Vs[-1].requires_grad = False
            self.tucker_regressor.Vs[i].requires_grad = False
            self.tucker_regressor.Vs[i].copy_(Vs[-1])
        # C = mmp(B, Us, modes=u_modes, transpose=True).detach()
        C = mmp(B, Vs+Us, modes=u_modes+v_modes, transpose=True).detach()
        self.tucker_regressor.core.requires_grad = False
        self.tucker_regressor.core.copy_(C)


    def _initialize_Us_with_hosvd_of_grad(self,
            use_dloader: bool,
            train_dataset: Optional[Dataset] = None,
            X: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            ):
        """Initialize factors :math:`U_n` with Higher-order SVD of gradient at B=0"""
        B = torch.zeros(self.tucker_regressor.dims, # pylint: disable=invalid-name
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=True)

        if self.verbosity >0:
            print("Initializing with the HoSVD of the gradient evaluated at 0")
            if use_dloader:
                print((f"Using training dataloader with {len(train_dataset)} "
                       "samples"))
        if use_dloader:
            train_dataset.reset_transform()
            dloader = DataLoader(train_dataset, **self.dataloader_cfg)
            loss = 0
            nsamples = 0
            grad = 0
            for _, batch_samples in tqdm(enumerate(dloader),
                                         desc="Batches: ",
                                         total=len(dloader),
                                         ):
                Xb = batch_samples['X'].to(self.device, non_blocking=True)
                eta = torch.tensordot(Xb,B,
                    dims=(
                        [n+1 for n in range(self.N)],
                        [n+self.M for n in range(self.N)],
                    ),
                )
                loss_b = self.tucker_regressor.loss_fn(
                    eta,
                    batch_samples['Y'].to(
                        self.device, non_blocking=True
                        )
                    )
                loss_b.backward()
                loss += loss_b.item()
                bnsamples = Xb.shape[0]
                nsamples += bnsamples
            loss = loss / nsamples
            grad = -B.grad / nsamples
        else:
            eta = torch.tensordot(X,B,
                dims=(
                    [n+1 for n in range(self.N)],
                    [n+self.M for n in range(self.N)],
                ),
            )
            loss = self.tucker_regressor.loss_fn(eta, y) / X.shape[0]
            loss.backward()
            grad = -B.grad
        for n in range(self.N):
            grad_m = matricize(grad, [n+self.M+1])
            U, _, _ = torch.linalg.svd(grad_m, full_matrices=False) # pylint: disable=not-callable
            self.tucker_regressor.Us[n].requires_grad = False
            self.tucker_regressor.Us[n].copy_(
                U[:, :self.tucker_regressor.feature_ranks[n]]
                )
            self.tucker_regressor.Us[n].requires_grad = True


    @torch.no_grad()
    def _evaluate_objectives(
        self,
        use_dloader: bool,
        dataset: Dataset = None,
        X: torch.Tensor = None,
        y: torch.Tensor = None,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        val_dataset: Dataset = None,
    ) -> dict[str, float]:
        X_transform = self.tucker_regressor.set_forward_mode('core')
        X_val_prime = None
        if use_dloader:
            if hasattr(dataset, 'update_transform'):
                dataset.update_transform(X_transform)
            cfg = self.dataloader_cfg.copy()
            dataloader = DataLoader(dataset, **cfg)
            numbatches = len(dataloader)
            X_primes = [None]*numbatches
            y_s = [None]*numbatches
            for i_batch, sample_batched in enumerate(dataloader):
                X_primes[i_batch] = sample_batched['X'].to(
                    self.device, non_blocking=True)
                y_s[i_batch] = sample_batched['Y'].to(
                    self.device, non_blocking=True)
            X_prime = torch.cat(X_primes)
            y = torch.cat(y_s)
            if val_dataset is not None:
                val_dataset.update_transform(X_transform)
                val_dataloader = DataLoader(val_dataset, **cfg)
                val_numbatches = len(val_dataloader)
                val_X_primes = [None] * val_numbatches
                val_y_s = [None] * val_numbatches
                for i_batch, sample_batched in enumerate(val_dataloader):
                    val_X_primes[i_batch] = sample_batched['X'].to(
                        self.device, non_blocking=True
                        )
                    val_y_s[i_batch] = sample_batched['Y'].to(
                        self.device, non_blocking=True
                        )
                X_val_prime = torch.cat(val_X_primes)
                y_val = torch.cat(val_y_s)
        else:
            X_prime = X_transform(X)
            if (X_val is not None) and (y_val is not None):
                X_val_prime = X_transform(X_val)

        objectives = {}
        # -------- Calculate Training and Validation Loss
        eta = self.tucker_regressor.functional_core(
            core=self.tucker_regressor.core,
            x=X_prime
            )
        objectives['train_loss'] = float(
            self.tucker_regressor.loss_fn(eta, y) / y.shape[0]
        )
        pred = self.tucker_regressor.inverse_link(eta)
        objectives['train_score'] = float(self.tucker_regressor.score(pred, y))
        self._t_loss = objectives['train_loss']
        self._t_score = objectives['train_score']
        if X_val_prime is not None:
            eta_val = self.tucker_regressor.functional_core(
                core=self.tucker_regressor.core,
                x=X_val_prime
                )
            objectives['val_loss'] = float(
                self.tucker_regressor.loss_fn(eta_val, y_val) / y_val.shape[0]
            )
            pred_val =  self.tucker_regressor.inverse_link(eta_val)
            objectives['val_score'] = float(
                self.tucker_regressor.score(pred_val, y_val)
            )
            self._v_loss = objectives['val_loss']
            self._v_score = objectives['val_score']
        obj_val = objectives['train_loss']
        modes = [n-self.tucker_regressor._M 
                    for n in self.tucker_regressor.lr_feature_modes]
        for i, mode in enumerate(modes):
            U = getattr(self.tucker_regressor, f'U_{mode}')
            L = self.Ls[mode-1]
            theta = self.thetas[i]
            drichlet_energy = 0
            if ((theta is not None) and (theta > 0) and (L is not None)):
                drichlet_energy = 0.5*theta*( U*(L@U) ).sum()
            objectives[f'U_{mode}_de'] = float(drichlet_energy)
            lda = self.ldas[mode-1]
            D = self.Ds[mode-1]
            l1_penalty = 0
            if ((lda is not None) and (lda > 0) and (D is not None)):
                l1_penalty = lda*((D@U).abs().sum())
            objectives[f'U_{mode}_l1'] = float(l1_penalty)
            obj_val += drichlet_energy + l1_penalty
            objectives[f'U_{mode}_nnz_ratio'] = float(
                (U.abs() >= 1e-6).sum()/U.numel()
            )

        C = self.tucker_regressor.core
        ridge_penalty = self.tau*C.pow(2).sum()/2 if self.tau>0 else 0
        objectives['ridge_penalty'] = float(ridge_penalty)
        obj_val += ridge_penalty
        objectives['objective'] = float(obj_val)
        return objectives


    def _solve_core_lbfgs(self, mnmx_problem: MinimaxProblem):
        C_param = self.tucker_regressor.core.data.clone().requires_grad_(True)
        params = [C_param]
        if self.fit_intercept:
            self.tucker_regressor.intercept.requires_grad = True
            params.append(self.tucker_regressor.intercept)
        optimizer = torch.optim.LBFGS(params, **self.lbfgs_config)
        def closure():
            optimizer.zero_grad()
            loss_t = mnmx_problem._func_f(C_param)
            loss_t.backward()
            return loss_t
        optimizer.step(closure)
        new_C = C_param.detach()
        self.tucker_regressor.core.requires_grad = False
        self.tucker_regressor.core.copy_(new_C)
        if self.fit_intercept:
            self.tucker_regressor.intercept.requires_grad = False
        return optimizer.state_dict()


    def _initialize_U_subproblem(
        self,
        use_dloader: bool,
        mode: int,
        lda: float = None,
        theta: float = None,
        dataset: Dataset = None,
        X: torch.Tensor = None,
        y: torch.Tensor = None,
    ) -> Tuple[Problem, TuckerCovariateTransform]:
        fwtype = 'feature_dir_n_full' if self.transform_covariates else 'full'
        X_transform = self.tucker_regressor.set_forward_mode(fwtype, mode)
        if use_dloader:
            if hasattr(dataset, 'update_transform'):
                dataset.update_transform(X_transform)
            cfg = self.dataloader_cfg.copy()
            dataloader = DataLoader(dataset, **cfg)
            numbatches = len(dataloader)
            X_primes = [None] * numbatches
            y_s = [None] * numbatches
            for i_batch, sample_batched in enumerate(dataloader):
                X_primes[i_batch] = sample_batched['X'].to(self.device, non_blocking=True)
                y_s[i_batch] = sample_batched['Y'].to(self.device, non_blocking=True)
            X_prime = torch.cat(X_primes)
            y = torch.cat(y_s)
        else:
            X_prime = X_transform(X)

        theta = self.thetas[mode-1] if theta is None else theta
        L = self.Ls[mode-1] if self.Ls is not None else None

        def objective(U):
            eta = self.tucker_regressor.functional_feature_dir_n_full(
                U=U, x=X_prime
                )
            loss = self.tucker_regressor.loss_fn(eta, y) / y.shape[0]
            if (theta is not None) and (theta > 0) and (L is not None):
                drichlet_energy = 0.5 * theta * (U * (L @ U)).sum()
                loss = loss + drichlet_energy
            return loss

        mu = self.ldas[mode - 1] if lda is None else lda

        def func_h(Y, *vargs, **kwargs):
            linf_y = Y.abs().max()
            return torch.inf if linf_y > mu else 0.0

        def prox_h(Y, *vargs, **kwargs):
            return torch.clamp(Y, -mu, +mu)

        mapping_A = self.Ds[mode-1]
        nabla_AT = mapping_A.T

        def func_g(U):
            return (mapping_A @ U).abs().sum()*mu

        mnmx_problem = MinimaxProblem(
            getattr(self.tucker_regressor, f'U_{mode}').manifold, # <<<<<<<<<<<<<<<<<<
            objective,
            func_h,
            prox_h,
            mapping_A,
            nabla_AT=nabla_AT,
            func_g=func_g,
        )
        return mnmx_problem, X_transform



    def _initialize_C_subproblem(
        self,
        use_dloader: bool,
        dataset: Dataset = None,
        X: torch.Tensor = None,
        y: torch.Tensor = None,
        ) -> Tuple[Problem, TuckerCovariateTransform]:
        X_transform = self.tucker_regressor.set_forward_mode('core')
        if use_dloader:
            if hasattr(dataset, 'update_transform'):
                dataset.update_transform(X_transform)
            cfg = self.dataloader_cfg.copy()
            dataloader = DataLoader(dataset, **cfg)
            numbatches = len(dataloader)
            X_primes = [None] * numbatches
            y_s = [None] * numbatches
            for i_batch, sample_batched in enumerate(dataloader):
                X_primes[i_batch] = sample_batched['X'].to(
                    self.device, non_blocking=True
                    )
                y_s[i_batch] = sample_batched['Y'].to(
                    self.device, non_blocking=True
                    )
            X_prime = torch.cat(X_primes)
            y = torch.cat(y_s)
        else:
            X_prime = X_transform(X)
        tau = self.tau

        def objective(C):
            eta = self.tucker_regressor.functional_core(core=C, x=X_prime)
            loss = self.tucker_regressor.loss_fn(eta, y) / y.shape[0]
            if tau != 0:
                loss = loss + tau * C.pow(2).sum()/2
            return loss

        mu = 0.0

        def func_h(Y, *vargs, **kwargs):
            linf_y = Y.abs().max()
            return torch.inf if linf_y > mu else 0.0

        def prox_h(Y, *vargs, **kwargs):    # pylint: disable=invalid-name
            return torch.clamp(Y, -mu, +mu)
        # This is just a placeholder. Will need to be fixed.
        mapping_A = lambda x: x # pylint: disable=invalid-name, unnecessary-lambda-assignment
        nabla_AT = torch.eye(   # pylint: disable=invalid-name
            self.tucker_regressor.core.shape[-2],
            device=self.device,
            dtype=self.dtype,
        )
        mnmx_problem = MinimaxProblem(
            self.tucker_regressor.core.manifold,
            objective,
            func_h,
            prox_h,
            mapping_A,
            nabla_AT=nabla_AT,
        )
        return mnmx_problem, X_transform

    def _initialize_solvers(self):
        if self.subproblem_solvers is None:
            self.subproblem_solvers = {}
            # for mode in range(1, self.N+1):
            for _, fmode in enumerate(self.tucker_regressor.lr_feature_modes):
                mode = fmode-self.tucker_regressor._M
                lda = self.ldas[mode-1]
                fd = self.tucker_regressor.feature_dims[mode-1]
                fr = self.tucker_regressor.feature_ranks[mode-1]
                self.subproblem_solvers[f'U_{mode}'] = RADA_RGD(
                    R=lda*(fd*fr)**0.5 if lda!=0 else 0,
                    c1=1e-1,
                    max_it=1000,
                    Tk=10,
                    max_line_search=25,
                    beta1=0.1*fd*fr**0.5,
                    rho=1.5,
                    tau_1=0.999,
                    tau_2=0.9,
                    zeta=1.0,
                    zeta_BB_min=1e-12,
                    zeta_BB_max=1e12,
                    eps=1e-6,
                    max_time= None,
                    max_function_evals=10000*10,
                    verbosity=0,
                    eps_floor=1e-10,
                    eps_anneal_factor = 0.1,
                    max_it_ceil = 10000,
                    max_it_growth = 3.2,
                )
                self.subproblem_solvers[f'U_{mode}'].min_gradient_norm = 1e-4


    def _initialize_log(self, seed=None, wandb_kwargs=None):
        self.hyper_parameters = self.hyperparameters
        self.log = {
            'hyper_parameters': self.hyperparameters,
            'algorithm': 'TuckerBCD-RADA-LBFGS',
            'stopping_criteria': {
                'max_time': self.max_time,
                'max_it': self.max_it,
                'min_gradient_norm': self.min_gradient_norm,
                },
            'subproblem_solver_params': {
                key: solver.get_parameters()
                for key, solver in self.subproblem_solvers.items()
                },
            'init_with_hosvd_of_grad': self.init_with_hosvd_of_grad,
            'init_with_lbfgs_full_glm': self.init_with_lbfgs_full_glm,
            'dataloader_cfg': self.dataloader_cfg,
            'seed': seed,
            'iterations': defaultdict(list)
            }
        self.log['subproblem_solver_params']['C'] = self.lbfgs_config
        if wandb_kwargs is not None:
            wandb_kwargs = deepcopy(wandb_kwargs)
            run_cfg = {
                **self.hyperparameters,
                'seed': seed,
                'init_with_hosvd_of_grad': self.init_with_hosvd_of_grad,
                'init_with_lbfgs_full_glm': self.init_with_lbfgs_full_glm,
                'max_time': self.max_time,
                'max_it': self.max_it,
                'min_gradient_norm': self.min_gradient_norm,
            }
            wandb_kwargs['config'] = {
                **wandb_kwargs.get('config', {}),**run_cfg
                }
            self._wandb_run = wandb.init(
                **wandb_kwargs
                )

    def _add_log_entry(self, start_time, iteration, objective, **kwargs):
        if self.log_verbosity <=0:
            return
        if (self.logging_period !=0) and (iteration % self.logging_period ==0):
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['time'].append(perf_counter() - start_time)
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)
            if self._wandb_run is not None:
                self._wandb_run: wandb.Run
                metrics = {key: self.log['iterations'][key][-1]
                           for key in self.log['iterations'].keys()}
                self._wandb_run.log(metrics, step=iteration)


    def _check_stopping_criteria(self, start_time, iteration, grad_norms):
        max_grad_norm = max(grad_norms.values())
        run_time = perf_counter() - start_time
        reason = None
        if (self.max_time is not None) and (run_time >= self.max_time):
            reason = f"Terminated - max time reached after {iteration} iterations."
        elif iteration>= self.max_it:
            reason = ("Terminated - maximum number of iterations reached after "
                      f"{run_time:.3f} seconds.")
        elif max_grad_norm <= self.min_gradient_norm:
            reason = (
                f"Terminated - min grad norm reached after {iteration} "
                f"iterations, {run_time:.3f} seconds."
            )
        else:
            stalled = []
            for key in self.solver_results.keys():
                if key != 'C':
                    reas = self.solver_results[key].stopping_criterion
                    if reas is not None:
                        stalled.append(reas.startswith('Stalled'))
            if all(stalled) and grad_norms['grad_C'] <= self.min_gradient_norm:
                reason = (
                    "Stalled - The subproblem solvers have stalled and are "
                    "unable to improve the objective value within resources."
                )
        return reason


    def _init_printer(self, with_val=False):
        if self.verbosity >=1:
            print("Starting BCD_RGD Solver for Tucker Regression")
        if self.verbosity >=2:
            print("Hyper-parameters:")
            pprint(self.hyperparameters)
            M = self.tucker_regressor.contravariant_degree  # pylint: disable=invalid-name
            iteration_format_length = int(np.log10(self.max_it)) + 1
            columns = [
                ("Iteration", f"{iteration_format_length}d"),
                ("Train score", ".4f")
                ]
            if with_val:
                columns += [("Val score", ".4f")]
                columns += [("Val loss", ".4e")]
            columns += [
                ("Cost", ".10e"),
                ("Gradient norm", ".4e"),
                ] + [
                (f"U_{mode-M} grad norm", ".4e")
                    for mode in self.tucker_regressor.lr_feature_modes
                ] + [
                ("C grad norm", ".4e")
                ]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        return column_printer


    @property
    def hyperparameters(self):
        """Hyper-parameters defining the optimization problem of the model."""
        return {
            'regression_type': self.tucker_regressor.regression_type,
            'feature_dims': self.tucker_regressor.feature_dims,
            'task_dims': self.tucker_regressor.task_dims,
            'feature_ranks': self.tucker_regressor.feature_ranks,
            'task_ranks': self.tucker_regressor.task_ranks,
            'coefficient_dims': (self.tucker_regressor.task_dims
                                 + self.tucker_regressor.feature_dims),
            'ranks': self.tucker_regressor.ranks,
            'tau': self.tau,
            'ldas': self.ldas,
            'thetas': self.thetas,
            }

    def _fit(self):
        pass
