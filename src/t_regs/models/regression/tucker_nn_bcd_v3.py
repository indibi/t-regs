"""Regularized Riemannian Block Coordinate Descent optimizer module for TuckerRegressor"""
# pylint: disable=invalid-name

import math
# from warnings import warn
from collections import defaultdict
from typing import Any, Optional, Sequence, Tuple, overload
from time import perf_counter
from pprint import pprint

import torch
# import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# TODO: Look into if generalized_svd is needed for initialization.
# from ..matrix_decomp.generalized_svd import generalized_svd
from .tucker_nn_regressor import TuckerRegressor, TuckerCovariateTransform
from ...models.regression.regression_base import RegressionBaseClass
from ...multilinear_ops.matricization import matricize
from ...solvers.manifold.rada import RADA_RGD
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
    Ds:
        Analysis or Generalized Lasso Penalty Matrix for the sparsity 
        regularization on :math:`U_m`. Defaults to identity matrix when provided
        None
    thetas:
        Smoothness regularization parameters for factor matrices :math:`U_m`
    Ls:
        Laplacian matrices to promote smoothness of the factor matrices 
        :math:`U_m`
    main_algorithm: str = 'BCD_RGD',
    max_it: int = 5000,
    min_gradient_norm: float = 1e-8,
       Terminate when the smallest of the norms of riemannian gradients of the
       blocks of variables become smaller than `min_gradient_norm`.
    max_time: float | None = None,
        Terminate when algorithm has been going for `max_time` longer than.
    verbosity: int = 0,
    log_verbosity: int = 1,
    report_period: int = 1,
    logging_period: int = 1,
    subproblem_solvers: dict[str, Any] = None
    dataloader_cfg: Optional[Dict[str, Any]]
        configuration dictionary that will be passed down to the :class:`Data
        Loader`.
    init_with_hosvd_of_grad: bool = True
        Initialize the feature directions, i.e. the factor matrices with the
        Higher-order SVD of the gradient of the full coefficients.
    transform_covariates: bool = True
        Transform covariate samples for subproblems.
    """
    # TODO: Add reference to HoSVD.
    algorithm_options = ['BCD_RGD', 'BCD_RADA_RGD']
    def __init__(self,
        tucker_regressor: TuckerRegressor,
        tau: float,
        ldas: Sequence[float],
        lda_core: float,
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
        dataloader_cfg: dict[str, Any] = None,
        initialization_cfg: dict[str, Any] = None,
        init_with_hosvd_of_grad: bool =True,
        transform_covariates: bool = True,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.tucker_regressor = tucker_regressor
        self.tau = tau
        self.ldas = ldas
        self.lda_core = lda_core
        self.thetas = thetas
        self.main_algorithm = main_algorithm
        self.max_it = max_it
        self.min_gradient_norm = min_gradient_norm
        self.max_time = max_time
        self.verbosity = verbosity
        self.log = None
        self.hyper_parameters = None
        self.log_verbosity = log_verbosity
        self.report_period = report_period
        self.logging_period = logging_period
        self.subproblem_solvers = subproblem_solvers
        self.dataloader_cfg = {} if dataloader_cfg is None else dataloader_cfg
        self.init_with_hosvd_of_grad = init_with_hosvd_of_grad
        self.transform_covariates = transform_covariates
        self.N = self.tucker_regressor.covariant_degree
        self.M = self.tucker_regressor.contravariant_degree
        if Ds is None:
            self.Ds = [torch.eye(
                self.tucker_regressor.feature_dims[i],
                device=self.device,
                dtype=self.dtype
                ) for i in range(self.N)
            ]
        else:
            self.Ds = Ds
        self.Ls = Ls
        self._initialize_solvers()
        self._t_loss = None
        self._v_loss = None
        self._t_score = None
        self._v_score = None


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
        if self.init_with_hosvd_of_grad:
            self._initialize_Us_with_hosvd_of_grad(
                use_dloader=use_dloader,
                train_dataset=train_dataset,
                X=X, y=y,
                )
        seed = kwargs.get('seed', 0)
        self._BCD_fit(
            use_dloader=use_dloader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            X=X, y=y,
            X_val=X_val, y_val=y_val,
            seed=seed
            )


    @torch.no_grad()
    def _predict(self, X:torch.Tensor, y=None, fw_mode='full', mode_n=None
            ) -> torch.Tensor:
        return self.tucker_regressor.predict(X, y, fw_mode, mode_n)

    @torch.no_grad()
    def _score(self, pred:torch.Tensor, Y:torch.Tensor) -> float:
        return self.tucker_regressor.score(pred, Y.reshape(-1,1))

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
                # grad += B.grad.detach()
                # B.grad = None
                bnsamples = Xb.shape[0]
                nsamples += bnsamples
            loss = loss / nsamples
            # loss.backward()
            grad = -B.grad / nsamples
            # grad = -grad/nsamples
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


    def _BCD_fit(self,
        use_dloader:bool,
        train_dataset: Dataset = None,
        val_dataset: Optional[Dataset] = None,
        X:torch.Tensor=None,
        y:torch.Tensor=None,
        X_val:Optional[torch.Tensor]=None,
        y_val:Optional[torch.Tensor]=None,
        seed: int = None,
        ):
        self._initialize_log(seed=seed)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        column_printer = self._init_printer(
            with_val= (
                ((X_val is not None) and (y_val is not None))
                or (val_dataset is not None)
                )
            )
        column_printer.print_header()
        start_time = perf_counter()

        modes = list(range(1, self.N+1))
        it = 0
        while True:
            it += 1

            # Solve core tensors subproblem
            solver = self.subproblem_solvers['C']
            mnmx_problem, _ = self._initialize_C_subproblem(
                use_dloader,
                dataset=train_dataset,
                X=X,
                y=y,
                )
            if self.solver_results['C'] is not None:
                y0 = self.solver_results['C'].point.y
            else:
                y0 = None
            rada_result = solver.solve(
                mnmx_problem,
                x0=self.tucker_regressor.core,
                y0=y0,
                seed=seed,
                )
            self.solver_results['C'] = rada_result
            self.tucker_regressor.core.requires_grad = False
            self.tucker_regressor.core.copy_(rada_result.point.x.data)

            for _, mode in enumerate(modes):
                # Solve `U_{mode}`s subproblem
                U = self.tucker_regressor.Us[mode-1]
                solver = self.subproblem_solvers[f'U_{mode}']
                mnmx_problem, X_transform = self._initialize_U_subproblem(
                    use_dloader,
                    mode,
                    dataset=train_dataset,
                    X=X,
                    y=y,
                    val_dataset=val_dataset if mode == self.N else None,
                    X_val=X_val if mode == self.N else None,
                    y_val=y_val if mode == self.N else None,
                    save_score_to_self= True if mode == self.N else False,
                    )

                if self.solver_results[f'U_{mode}'] is not None:
                    y0 = self.solver_results[f'U_{mode}'].point.y
                else:
                    y0 = None
                rada_result = solver.solve(mnmx_problem,
                                x0=self.tucker_regressor.Us[mode-1],
                                y0=y0,
                                seed=seed,
                                )
                self.solver_results[f'U_{mode}'] = rada_result
                self.tucker_regressor.Us[mode-1].requires_grad = False
                self.tucker_regressor.Us[mode-1].copy_(
                    rada_result.point.x.data
                    )

            # Calculate values for logging
            scores = {
                'train_score': self._t_score,
                'train_loss': self._t_loss,
                'val_score': self._v_score,
                'val_loss': self._v_loss,
                }
            in_prods = {}
            # Check convergence
            problem, X_transform = self._initialize_C_subproblem(
                use_dloader,
                dataset=train_dataset,
                X=X,
                y=y
                )

            grad_C = problem.grad_f(
                self.tucker_regressor.core,
                repeat_forward=True
                )
            with torch.no_grad():
                y_mmx = self.solver_results['C'].point.y
                grad_C = grad_C + y_mmx
                grad_C_norm = problem.manifold.norm(
                    self.tucker_regressor.core,
                    grad_C
                    )
                in_prods['core'] = problem.inner_product(
                    self.tucker_regressor.core,
                    y_mmx,
                )
            grad_norms = {"grad_C": float(grad_C_norm)}
            for mode in modes:
                U = self.tucker_regressor.Us[mode-1]
                problem, _ = self._initialize_U_subproblem(
                    use_dloader,
                    mode,
                    dataset=train_dataset,
                    X=X,
                    y=y,
                    )
                grad_U = problem.grad_f(U, repeat_forward=True)
                with torch.no_grad():
                    Uy = self.solver_results[f'U_{mode}'].point.y
                    nabla_AT = problem.nabla_AT(U)
                    grad_U = grad_U + nabla_AT@Uy
                    grad_U_norm = problem.manifold.norm(U, grad_U)
                    in_prod = problem.inner_product(
                        self.solver_results[f'U_{mode}'].point.x,
                        self.solver_results[f'U_{mode}'].point.y,
                    )
                    in_prods[f'U_{mode}'] = in_prod
                    # obj_val += self.ldas[mode-1]* ((nabla_AT@Uy)*U).sum()
                grad_norms[f'grad_U_{mode}'] = float(grad_U_norm)
            max_grad_norm = max(grad_norms.values())

            objectives = self._calculate_objectives(in_prods)
            row = [it, scores['train_score']]
            if ((X_val is not None and y_val is not None)
                or val_dataset is not None):
                row += [scores['val_score']]
            row += [objectives['objective'], max_grad_norm]
            row += [grad_norms[f'grad_U_{mode}'] for mode in range(1, self.N+1)]
            row += [grad_norms['grad_C']]
            # print(row)
            column_printer.print_row(row)
            obj_val = objectives.pop('objective')
            self._add_log_entry(
                start_time,
                it,
                obj_val,
                gradient_norm=max_grad_norm,
                **scores,
                **objectives,
                **grad_norms,
                )

            reason = self._check_stopping_criteria(
                start_time,
                it,
                max_grad_norm
                )
            if reason is not None:
                if self.verbosity >=1:
                    print(reason)
                break

    @torch.no_grad()
    def _calculate_objectives(
            self,
            in_prods:dict[str, float]
            ) -> dict[str, float]:
        obj_val = self._t_loss
        objectives = {}
        modes = list(range(1, self.N+1))
        for i, mode in enumerate(modes):
            U = self.tucker_regressor.Us[i]
            L = self.Ls[i]
            theta = self.thetas[i]
            drichlet_energy = 0
            if ((theta is not None) and (theta > 0) and (L is not None)):
                drichlet_energy = 0.5*theta*( U*(L@U) ).sum()
            objectives[f'U_{mode}_de'] = float(drichlet_energy)

            lda = self.ldas[i]
            D = self.Ds[i]
            l1_penalty = 0
            if ((lda is not None) and (lda > 0) and (D is not None)):
                l1_penalty = lda*((D@U).abs().sum())
            disparity =  l1_penalty - in_prods[f'U_{mode}']
            objectives[f'U_{mode}_l1'] = float(l1_penalty)
            objectives[f'U_{mode}_l1_disparity'] = float(disparity)
            obj_val += drichlet_energy + l1_penalty

            objectives[f'U_{mode}_nnz_ratio'] = float(
                (U.abs() >= 1e-8).sum()/U.numel()
            )

        C = self.tucker_regressor.core
        ridge_penalty = self.tau*C.pow(2).sum()
        objectives['ridge_penalty'] = float(ridge_penalty)

        core_l1_penalty = self.lda_core*C.abs().sum()
        objectives['core_l1_penalty'] = float(core_l1_penalty)
        disparity = core_l1_penalty - in_prods['core']
        objectives['core_l1_disparity'] = float(disparity)
        obj_val += core_l1_penalty + ridge_penalty

        objectives['core_nnz_ratio'] = float(
            (C.abs() >= 1e-8).sum()/C.numel()
        )
        objectives['objective'] = float(obj_val)
        return objectives


    def _initialize_solvers(self):
        # ['BCD_RGD', 'BCD_RADMM', 'BCD_RADA_RGD', 'BCD_RADA_PGD']
        ma = self.main_algorithm
        if self.subproblem_solvers is None:
            self.subproblem_solvers = {}
            for mode in range(1, self.N+1):
                lda = self.ldas[mode-1]
                fd = self.tucker_regressor.feature_dims[mode-1]
                fr = self.tucker_regressor.feature_ranks[mode-1]
                self.subproblem_solvers[f'U_{mode}'] = RADA_RGD(
                    R=lda*(fd*fr)**0.5 if lda!=0 else 1e-6*(fd*fr)**0.5,
                    c1=1e-1,
                    max_it=10000,
                    Tk=2,
                    max_line_search=10,
                    beta1=0.01*fd*fr**0.5,
                    rho=2.0,
                    tau_1=0.999,
                    tau_2=0.9,
                    zeta=1.0,
                    zeta_BB_min=1e-20,
                    zeta_BB_max=1e20,
                    eps=1e-8,
                    max_time= None,
                    max_function_evals=10*1000*5,
                    verbosity=0,
                )
            core_dim = math.prod(self.tucker_regressor.ranks)
            lda_core = self.lda_core
            self.subproblem_solvers['C'] = RADA_RGD(
                R=lda_core*(core_dim)**0.5 if lda_core!=0 else 1e-6*(core_dim)**0.5,
                c1=1e-1,
                max_it=1000,
                Tk=1,
                max_line_search=25,
                beta1=0.01*(core_dim)**0.5,
                rho=2.0,
                tau_1=0.999,
                tau_2=0.9,
                zeta=1.0,
                zeta_BB_min=1e-20,
                zeta_BB_max=1e20,
                eps=1e-8,
                max_time= None,
                max_function_evals=10*1000*5,
                verbosity=0,
            )



    def _initialize_C_subproblem(self,
        use_dloader:bool,
        dataset: Dataset = None,
        X:torch.Tensor=None,
        y:torch.Tensor=None,
        ) -> Tuple[Problem,TuckerCovariateTransform]:
        X_transform = self.tucker_regressor.set_forward_mode('core')
                            #  if self.transform_covariates else 'full')
        if use_dloader:
            if hasattr(dataset, 'update_transform'):
                dataset.update_transform(X_transform)
            cfg = self.dataloader_cfg.copy()
            # cfg['batch_size'] = len(dataset)

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
        else:
            X_prime = X_transform(X)
        tau = self.tau
        def objective(C):
            if not self.transform_covariates:
                loss = 0
                nsamples = 0
                for _, sample_batched in enumerate(dataloader):
                    eta = self.tucker_regressor.functional_core(
                        core=C,
                        x=sample_batched['X'].to(self.device, non_blocking=True),
                        )
                    loss += self.tucker_regressor.loss_fn(
                        eta,
                        sample_batched['Y'].to(self.device, non_blocking=True)
                        )
                    nsamples += sample_batched['X'].shape[0]
                loss = loss / nsamples
            else:
                eta = self.tucker_regressor.functional_core(
                        core=C,
                        x=X_prime,
                        )
                loss = self.tucker_regressor.loss_fn(eta, y)

            if tau != 0:
                loss += tau*C.pow(2).sum()
            return loss

        mu = self.lda_core
        def func_h(Y, *vargs, **kwargs): # pylint: disable=unused-argument
            linf_y = Y.abs().max()
            if linf_y > mu:
                return torch.inf
            else:
                return 0

        def prox_h(Y, *vargs, **kwargs): # pylint: disable=unused-argument
            return torch.clamp(Y, -mu, +mu)

        # Identity mapping
        mapping_A = lambda x:x # pylint: disable=unnecessary-lambda
        nabla_AT = lambda x:x# pylint: disable=unnecessary-lambda
        mnmx_problem = MinimaxProblem(
            self.tucker_regressor.core.manifold,
            objective,
            func_h,
            prox_h,
            mapping_A,
            nabla_AT
            )
        return mnmx_problem, X_transform

    def _initialize_U_subproblem(self,
        use_dloader:bool,
        mode:int,
        lda: float = None,
        theta: float = None,
        dataset: Dataset = None,
        X:torch.Tensor=None,
        y:torch.Tensor=None,
        val_dataset: Dataset=None,
        X_val:torch.Tensor=None,
        y_val:torch.Tensor=None,
        save_score_to_self: bool = False,
        ) -> Tuple[Problem,TuckerCovariateTransform]:
        fwtype = 'feature_dir_n_full' if self.transform_covariates else 'full'
        X_transform = self.tucker_regressor.set_forward_mode(fwtype, mode)
        X_val_prime = None
        if use_dloader:
            if hasattr(dataset, 'update_transform'):
                dataset.update_transform(X_transform)
            cfg = self.dataloader_cfg.copy()
            # cfg['batch_size'] = len(dataset)
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
            # samples = next(iter(dataloader))
            # X_prime = samples['X'].to(self.device, non_blocking=True)
            # y = samples['Y'].to(self.device, non_blocking=True)
            if val_dataset is not None:
                val_dataset.update_transform(X_transform)
                # val_dataloader = DataLoader(val_dataset, **cfg)
                # val_samples = next(iter(val_dataloader))
                # X_val_prime = val_samples['X'].to(self.device, non_blocking=True)
                # y_val = val_samples['Y'].to(self.device, non_blocking=True)
                val_dataloader = DataLoader(val_dataset, **cfg)
                val_numbatches = len(val_dataloader)
                val_X_primes = [None]*val_numbatches
                val_y_s = [None]*val_numbatches
                for i_batch, sample_batched in enumerate(val_dataloader):
                    val_X_primes[i_batch] = sample_batched['X'].to(
                        self.device, non_blocking=True)
                    val_y_s[i_batch] = sample_batched['Y'].to(
                        self.device, non_blocking=True)
                X_val_prime = torch.cat(val_X_primes)
                y_val = torch.cat(val_y_s)
            # use_dloader=False
        else:
            X_prime = X_transform(X)
            if (X_val is not None) and (y_val is not None):
                X_val_prime = X_transform(X_val)
        theta = self.thetas[mode-1] if theta is None else theta
        if self.Ls is not None:
            L = self.Ls[mode-1]
        else:
            L = None
        def objective(U):
            if not self.transform_covariates:
                loss = 0
                nsamples = 0
                t_score = 0
                for _, sample_batched in enumerate(dataloader):
                    # eta = self.tucker_regressor(sample_batched['X'])
                    eta = self.tucker_regressor.functional_feature_dir_n_full(
                        U=U,
                        x=sample_batched['X'],
                        )
                    loss += self.tucker_regressor.loss_fn(
                        eta,
                        sample_batched['Y']
                        )
                    b_samples = sample_batched['Y'].shape[0]
                    nsamples += b_samples
                    if save_score_to_self:
                        pred = self.tucker_regressor.inverse_link(eta)
                        # _score already has torch.no_grad
                        t_score += nsamples*self.tucker_regressor.score(
                            pred, sample_batched['Y'])
                if nsamples == 0:
                    raise ValueError("Dataloader empty?")
                loss = loss / nsamples
                t_score = t_score / nsamples

                # Calculate validation loss and scores.
                with torch.no_grad():
                    v_loss = 0
                    v_nsamples = 0
                    v_score = 0
                    if val_dataset is not None:
                        for _, sample_batched in enumerate(val_dataloader):
                            # eta = self.tucker_regressor(sample_batched['X'])
                            eta = self.tucker_regressor.functional_feature_dir_n_full(
                                U=U,
                                x=sample_batched['X'],
                                )
                            v_loss += self.tucker_regressor.loss_fn(
                                eta,
                                sample_batched['Y']
                                )
                            b_samples = sample_batched['Y'].shape[0]
                            v_nsamples += b_samples
                            if save_score_to_self:
                                pred = self.tucker_regressor.inverse_link(eta)
                                # _score already has torch.no_grad
                                v_score += b_samples*self._score(
                                    pred, sample_batched['Y'])
                        v_score = v_score / v_nsamples if v_nsamples !=0 else None
                        v_loss = v_loss / v_nsamples if v_nsamples !=0 else None

                if save_score_to_self:
                    self._t_loss = loss.item()
                    self._t_score = t_score.item()
                    self._v_loss = v_loss.item()
                    self._v_score = v_score.item()
            else:
                # eta = self.tucker_regressor(X_prime)
                eta = self.tucker_regressor.functional_feature_dir_n_full(
                    U=U,
                    x=X_prime,
                    )
                loss = self.tucker_regressor.loss_fn(eta, y) / y.shape[0]

                with torch.no_grad():
                    if save_score_to_self:
                        self._t_loss = loss.item()
                        pred = self.tucker_regressor.inverse_link(eta)
                        t_score = self._score(pred, y)
                        self._t_score = t_score
                        if X_val_prime is not None:
                            eta = self.tucker_regressor.functional_feature_dir_n_full(
                                U=U,
                                x=X_val_prime,
                                )
                            # eta = self.tucker_regressor(X_val_prime)
                            self._v_loss = self.tucker_regressor.loss_fn(
                                eta,
                                y_val
                                ) / y_val.shape[0]
                            pred = self.tucker_regressor.inverse_link(eta)
                            self._v_score = self._score(pred, y_val)
            # tr(U^T L U) = <U, L U>
            if ((theta is not None) and (theta > 0) and (L is not None)):
                drichlet_energy = 0.5*theta*( U*(L@U) ).sum()
                loss += drichlet_energy
            return loss

        mu = self.ldas[mode-1] if lda is None else lda

        def func_h(Y, *vargs, **kwargs): # pylint: disable=unused-argument
            linf_y = Y.abs().max()
            if linf_y > mu:
                return torch.inf
            else:
                return 0

        def prox_h(Y, *vargs, **kwargs): # pylint: disable=unused-argument
            return torch.clamp(Y, -mu, +mu)

        mapping_A = self.Ds[mode-1]
        nabla_AT = mapping_A.T
        mnmx_problem = MinimaxProblem(
            self.tucker_regressor.Us[mode-1].manifold,
            objective,
            func_h,
            prox_h,
            mapping_A,
            nabla_AT
            )
        return mnmx_problem, X_transform

    @property
    def hyperparameters(self):
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
            'lda_core': self.lda_core,
            'thetas': self.thetas,
            }

    def _initialize_log(self, seed=None):
        self.hyper_parameters = {
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
            'lda_core': self.lda_core,
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
            'subproblem_solver_params': {
                key: solver.get_parameters()
                for key, solver in self.subproblem_solvers.items()
                },
            'init_with_hosvd_of_grad': self.init_with_hosvd_of_grad,
            'dataloader_cfg': self.dataloader_cfg,
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
        if (self.max_time is not None) and (run_time >= self.max_time):
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

    def _init_printer(self, with_val=False):
        if self.verbosity >=1:
            print("Starting BCD_RGD Solver for Tucker Regression")
        if self.verbosity >=2:
            print("Hyper-parameters:")
            pprint(self.hyperparameters)
            iteration_format_length = int(np.log10(self.max_it)) + 1
            columns = [
                ("Iteration", f"{iteration_format_length}d"),
                ("Train score", ".4f")
                ]
            if with_val:
                columns += [("Val score", ".4f")]
            columns += [
                ("Cost", ".10e"),
                ("Gradient norm", ".4e"),
                ] + [
                (f"U_{mode} grad norm", ".4e") for mode in range(1, self.N+1)
                ] + [
                ("C grad norm", ".4e")
                ]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        return column_printer

    def _fit(self):
        pass
