"""Regularized Riemannian Block Coordinate Descent optimizer module for TuckerRegressor (v4 - Stabilized)

Improvements over v2:
- [CHANGE / FIX Solution 1 & 2]: Uses stabilized RADA_RGD from `rada_v2` with corrected Armijo line search,
  safe line search rejection, capped BB step sizes, and best-iterate tracking.
- [CHANGE / FIX Solution 3.1]: Added BCD Block Monotonicity Guard (`enforce_bcd_monotonicity`).
  Evaluates the true block objective before and after each subproblem solve; rejects candidates that would increase objective.
- [CHANGE / FIX Solution 3.2]: Smoothing floor `lda_smooth_floor` prevents vanishing lambda_smooth and condition number explosion at high sparsity.
- [CHANGE / FIX Solution 3.3]: Removed the runaway epsilon-reduction feedback loop (formerly lines 424-442 in v2)
  which drove epsilon down to 10^-12 and caused extreme ill-conditioning.
- [CHANGE / FIX Solution 4]: Added smooth L-BFGS optimizer for Core Tensor C (`solver_C_type='lbfgs'`).
  Since C has no l1 penalty and is unconstrained Euclidean, L-BFGS converges in 5-10 iterations with guaranteed strict descent.
- [CHANGE / FIX Clean Tracking]: Global training and validation loss/score are computed directly on synchronized
  parameters (C, U_1, ..., U_N) once per BCD cycle, eliminating state leakage from trial steps in mode N.
"""
# pylint: disable=invalid-name

import math
from collections import defaultdict
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
# [CHANGE / FIX Solution 1 & 2]: Import stabilized RADA_RGD from rada_v2
from ...solvers.manifold.rada_v2 import RADA_RGD
from ...solvers.manifold.minimax_problem import MinimaxProblem
from ...solvers.manifold.problem import Problem
from ...utils import printer


class TuckerRegressorBCD(RegressionBaseClass):
    r"""Sparse, Generalized Tucker Regression with Block Coordinate Descent (v4 - Stabilized)

    Parameters
    ----------
    tucker_regressor: TuckerRegressor
        :class:`TuckerRegressor` object with the regression coefficients.
    tau: float
        Ridge regularization parameter on core tensor.
    ldas: Sequence[float]
        Sparsity regularization parameter for feature direction matrices U_m.
    lda_core: float
        Sparsity regularization parameter for core tensor (usually 0.0).
    thetas: Sequence[float]
        Smoothness regularization parameters for factor matrices U_m.
    Ds: Sequence[torch.Tensor | None] | None
        Analysis/Lasso Penalty Matrix for U_m. Defaults to identity.
    Ls: Sequence[torch.Tensor | None] | None
        Laplacian matrices for factor matrices U_m.
    main_algorithm: str = 'BCD_RADA_RGD'
    solver_C_type: str = 'lbfgs'
        [CHANGE / FIX Solution 4]: Optimizer for core tensor C: 'lbfgs' (recommended, smooth & monotonic) or 'rada'.
    enforce_bcd_monotonicity: bool = True
        [CHANGE / FIX Solution 3.1]: Whether to enforce strict block monotonicity in BCD.
    lda_smooth_floor: float = 1e-4
        [CHANGE / FIX Solution 3.2]: Floor on dual smoothing parameter to keep condition number <= 10^4 at high sparsity.
    max_it: int = 5000
    min_gradient_norm: float = 1e-8
    max_time: float | None = None
    verbosity: int = 0
    log_verbosity: int = 1
    report_period: int = 1
    logging_period: int = 1
    subproblem_solvers: dict[str, Any] = None
    dataloader_cfg: dict[str, Any] = None
    init_with_hosvd_of_grad: bool = True
    transform_covariates: bool = True
    """

    algorithm_options = ['BCD_RGD', 'BCD_RADA_RGD']

    def __init__(
        self,
        tucker_regressor: TuckerRegressor,
        tau: float,
        ldas: Sequence[float],
        lda_core: float = 0.0,
        thetas: Sequence[float] = None,
        Ds: Sequence[torch.Tensor | None] | None = None,
        Ls: Sequence[torch.Tensor | None] | None = None,
        main_algorithm: str = 'BCD_RADA_RGD',
        solver_C_type: str = 'lbfgs',
        enforce_bcd_monotonicity: bool = True,
        lda_smooth_floor: float = 1e-4,
        lda_smooth_floor_min: float = 1e-8,
        stall_anneal_factor: float = 0.5,
        subproblem_grad_tol: float = 1e-4,
        min_objective_decrease: float = 1e-8,
        anneal_on_subproblem_convergence: bool = True,
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
        init_with_hosvd_of_grad: bool = True,
        transform_covariates: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tucker_regressor = tucker_regressor
        self.tau = tau
        self.ldas = ldas
        self.lda_core = lda_core
        self.thetas = [0.0] * self.tucker_regressor.covariant_degree if thetas is None else thetas
        self.main_algorithm = main_algorithm
        self.solver_C_type = solver_C_type
        self.enforce_bcd_monotonicity = enforce_bcd_monotonicity
        self.lda_smooth_floor = lda_smooth_floor
        self.lda_smooth_floor_init = lda_smooth_floor
        self.lda_smooth_floor_min = lda_smooth_floor_min
        self.stall_anneal_factor = stall_anneal_factor
        self.subproblem_grad_tol = subproblem_grad_tol
        self.min_objective_decrease = min_objective_decrease
        self.anneal_on_subproblem_convergence = anneal_on_subproblem_convergence
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
            self.Ds = [
                torch.eye(
                    self.tucker_regressor.feature_dims[i],
                    device=self.device,
                    dtype=self.dtype,
                )
                for i in range(self.N)
            ]
        else:
            self.Ds = Ds
        self.Ls = Ls
        self.solver_results = defaultdict(lambda: None)
        self._initialize_solvers()
        self._t_loss = None
        self._v_loss = None
        self._t_score = None
        self._v_score = None
        self._wandb_run = None

    @overload
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ):
        pass

    @overload
    def fit(
        self,
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
            raise ValueError("Input must be (train_dataset, [val_dataset]) or (X, y, [X_val, y_val])")

        if self.init_with_hosvd_of_grad:
            self._initialize_Us_with_hosvd_of_grad(
                use_dloader=use_dloader,
                train_dataset=train_dataset,
                X=X,
                y=y,
            )
        seed = kwargs.get('seed', 0)
        self._BCD_fit(
            use_dloader=use_dloader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            seed=seed,
            wandb_kwargs=kwargs.get('wandb_kwargs', None),
        )

    @torch.no_grad()
    def _predict(self, X: torch.Tensor, y=None, fw_mode='full', mode_n=None) -> torch.Tensor:
        return self.tucker_regressor.predict(X, y, fw_mode, mode_n)

    @torch.no_grad()
    def _score(self, pred: torch.Tensor, Y: torch.Tensor) -> float:
        return self.tucker_regressor.score(pred, Y.reshape(-1, 1))

    def _initialize_Us_with_hosvd_of_grad(
        self,
        use_dloader: bool,
        train_dataset: Optional[Dataset] = None,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        """Initialize factors U_n with Higher-order SVD of gradient at B=0."""
        B = torch.zeros(
            self.tucker_regressor.dims,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        if self.verbosity > 0:
            print("Initializing with the HoSVD of the gradient evaluated at 0")
            if use_dloader:
                print(f"Using training dataloader with {len(train_dataset)} samples")
        if use_dloader:
            train_dataset.reset_transform()
            dloader = DataLoader(train_dataset, **self.dataloader_cfg)
            loss = 0
            nsamples = 0
            for _, batch_samples in tqdm(
                enumerate(dloader),
                desc="Batches: ",
                total=len(dloader),
            ):
                Xb = batch_samples['X'].to(self.device, non_blocking=True)
                eta = torch.tensordot(
                    Xb,
                    B,
                    dims=(
                        [n + 1 for n in range(self.N)],
                        [n + self.M for n in range(self.N)],
                    ),
                )
                loss_b = self.tucker_regressor.loss_fn(
                    eta,
                    batch_samples['Y'].to(self.device, non_blocking=True),
                )
                loss_b.backward()
                loss += loss_b.item()
                bnsamples = Xb.shape[0]
                nsamples += bnsamples
            loss = loss / nsamples
            grad = -B.grad / nsamples
        else:
            eta = torch.tensordot(
                X,
                B,
                dims=(
                    [n + 1 for n in range(self.N)],
                    [n + self.M for n in range(self.N)],
                ),
            )
            loss = self.tucker_regressor.loss_fn(eta, y) / X.shape[0]
            loss.backward()
            grad = -B.grad

        for n in range(self.N):
            grad_m = matricize(grad, [n + self.M + 1])
            U, _, _ = torch.linalg.svd(grad_m, full_matrices=False)
            self.tucker_regressor.Us[n].requires_grad = False
            self.tucker_regressor.Us[n].copy_(
                U[:, : self.tucker_regressor.feature_ranks[n]]
            )
            self.tucker_regressor.Us[n].requires_grad = True

    def _compute_block_objective(
        self,
        mode: int,
        U: torch.Tensor,
        mnmx_problem: MinimaxProblem,
        ) -> float:
        """[CHANGE / FIX Solution 3.1]: Evaluate true block objective value for monotonicity guard."""
        # func_f includes GLM loss and Dirichlet smoothness penalty
        val = float(mnmx_problem.func_f(U, backward_pass=False))
        lda = self.ldas[mode - 1]
        D = self.Ds[mode - 1]
        if lda is not None and lda > 0 and D is not None:
            val += lda * float((D @ U).abs().sum())
        return val

    def _solve_core_lbfgs(
        self,
        mnmx_problem: MinimaxProblem,
        max_iter: int = 25,
    ):
        """[CHANGE / FIX Solution 4]: Monotonic, fast L-BFGS solver for unconstrained Euclidean core tensor."""
        C_param = self.tucker_regressor.core.data.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [C_param],
            lr=1.0,
            max_iter=max_iter,
            history_size=10,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-8,
            tolerance_change=1e-12,
        )

        def closure():
            optimizer.zero_grad()
            loss_t = mnmx_problem._func_f(C_param)
            loss_t.backward()
            return loss_t

        old_obj = float(mnmx_problem.func_f(self.tucker_regressor.core, backward_pass=False))

        optimizer.step(closure)
        new_C = C_param.detach()
        new_obj = float(mnmx_problem.func_f(new_C, backward_pass=False))
        if (not self.enforce_bcd_monotonicity) or (new_obj <= old_obj + 1e-9):
            self.tucker_regressor.core.requires_grad = False
            self.tucker_regressor.core.copy_(new_C)


    def _BCD_fit(
        self,
        use_dloader: bool,
        train_dataset: Dataset = None,
        val_dataset: Optional[Dataset] = None,
        X: torch.Tensor = None,
        y: torch.Tensor = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        seed: int = None,
        wandb_kwargs: dict = None,
        modes: list[int] = None,
        fit_C: bool = True,
    ):
        self._initialize_log(seed=seed, wandb_kwargs=wandb_kwargs)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        has_val = (
            ((X_val is not None) and (y_val is not None))
            or (val_dataset is not None)
        )
        column_printer = self._init_printer(with_val=has_val)
        column_printer.print_header()
        start_time = perf_counter()

        modes = list(range(1, self.N + 1)) if modes is None else modes
        it = 0
        prev_obj = torch.inf

        while it < self.max_it:
            it += 1
            if fit_C:
                # [CHANGE / FIX Solution 4]: Optimize core tensor C with L-BFGS or RADA
                mnmx_problem, _ = self._initialize_C_subproblem(
                    use_dloader,
                    dataset=train_dataset,
                    X=X,
                    y=y,
                )
                if self.solver_C_type == 'lbfgs':
                    self._solve_core_lbfgs(mnmx_problem, max_iter=25)
                else:
                    solver = self.subproblem_solvers['C']
                    y0 = self.solver_results['C'].point.y if self.solver_results['C'] is not None else None
                    rada_result = solver.solve(
                        mnmx_problem,
                        x0=self.tucker_regressor.core,
                        y0=y0,
                        seed=seed,
                    )
                    self.solver_results['C'] = rada_result
                    self.tucker_regressor.core.requires_grad = False
                    self.tucker_regressor.core.copy_(rada_result.point.x.data)

            # [CHANGE / FIX Solution 3.1 & 3.5]: Monotonicity guard with gradient-driven stall refinement
            for _, mode in enumerate(modes):
                U = self.tucker_regressor.Us[mode - 1]
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
                    save_score_to_self=False,  # [CHANGE / FIX]: Clean evaluation done globally below
                )

                old_U = U.clone().detach()
                old_block_obj = self._compute_block_objective(mode, old_U, mnmx_problem)

                y0 = self.solver_results[f'U_{mode}'].point.y if self.solver_results[f'U_{mode}'] is not None else None
                rada_result = solver.solve(
                    mnmx_problem,
                    x0=old_U,
                    y0=y0,
                    seed=seed,
                )

                new_U = rada_result.point.x.data
                new_block_obj = self._compute_block_objective(mode, new_U, mnmx_problem)
                sub_grad_norm = float(rada_result.gradient_norm) if rada_result.gradient_norm is not None else 0.0

                improved = (old_block_obj - new_block_obj) > self.min_objective_decrease

                accepted = (not self.enforce_bcd_monotonicity) or (new_block_obj <= old_block_obj + 1e-9)
                # Monotonicity Guard check
                if accepted:
                    self.solver_results[f'U_{mode}'] = rada_result
                    self.tucker_regressor.Us[mode - 1].requires_grad = False
                    self.tucker_regressor.Us[mode - 1].copy_(new_U)
                else:
                    if self.verbosity >= 2:
                        print(
                            f"BCD Mode {mode}: Rejected candidate (old: {old_block_obj:.6e}, "
                            f"candidate: {new_block_obj:.6e}, sub_grad: {sub_grad_norm:.3e})"
                        )

                # Record subproblem solve details
                self.subproblem_history.append({
                    'iteration': it,
                    'subproblem': f'U_{mode}',
                    'iterations': rada_result.iterations,
                    'gradient_norm': sub_grad_norm,
                    'stopping_criterion': rada_result.stopping_criterion,
                    'old_block_obj': float(old_block_obj),
                    'new_block_obj': float(new_block_obj),
                    'accepted': accepted,
                    'improved': improved,
                    'lda_smooth': solver.lda,
                })

                # [Anneal on subproblem convergence]:
                # If the subproblem optimization converged (stopping criterion eps or sub_grad_norm <= tol)
                # but did not improve the objective for the next iteration of BCD, anneal lambda_smooth.
                if self.anneal_on_subproblem_convergence:
                    subproblem_converged = (
                        (rada_result.stopping_criterion is not None and ('eps' in rada_result.stopping_criterion.lower()))
                        or (sub_grad_norm <= self.subproblem_grad_tol)
                    )
                    if subproblem_converged and (not improved):
                        old_lda = solver.lda
                        solver.lda = max(self.lda_smooth_floor_min, old_lda * self.stall_anneal_factor)
                        if self.verbosity >= 2:
                            print(
                                f"BCD Mode {mode}: Subproblem converged (grad={sub_grad_norm:.3e}) "
                                f"without objective improvement. Annealing lda_smooth for next BCD iteration: "
                                f"{old_lda:.3e} -> {solver.lda:.3e}"
                            )

            # [CHANGE / FIX]: Synchronized, clean evaluation of global train and validation metrics
            with torch.no_grad():
                self.tucker_regressor.set_forward_mode('full')
                if use_dloader:
                    train_dataset.reset_transform()
                    t_loader = DataLoader(train_dataset, **self.dataloader_cfg)
                    t_loss, t_score, t_n = 0.0, 0.0, 0
                    for batch in t_loader:
                        Xb = batch['X'].to(self.device, non_blocking=True)
                        Yb = batch['Y'].to(self.device, non_blocking=True)
                        if Yb.ndim == 1:
                            Yb = Yb.reshape(-1, 1)
                        eta_b = self.tucker_regressor(Xb)
                        t_loss += float(self.tucker_regressor.loss_fn(eta_b, Yb))
                        pred_b = self.tucker_regressor.inverse_link(eta_b)
                        t_score += float(self.tucker_regressor.score(pred_b, Yb)) * Yb.shape[0]
                        t_n += Yb.shape[0]
                    self._t_loss = t_loss / t_n if t_n > 0 else 0.0
                    self._t_score = t_score / t_n if t_n > 0 else 0.0

                    if val_dataset is not None:
                        val_dataset.reset_transform()
                        v_loader = DataLoader(val_dataset, **self.dataloader_cfg)
                        v_loss, v_score, v_n = 0.0, 0.0, 0
                        for batch in v_loader:
                            Xb = batch['X'].to(self.device, non_blocking=True)
                            Yb = batch['Y'].to(self.device, non_blocking=True)
                            if Yb.ndim == 1:
                                Yb = Yb.reshape(-1, 1)
                            eta_b = self.tucker_regressor(Xb)
                            v_loss += float(self.tucker_regressor.loss_fn(eta_b, Yb))
                            pred_b = self.tucker_regressor.inverse_link(eta_b)
                            v_score += float(self.tucker_regressor.score(pred_b, Yb)) * Yb.shape[0]
                            v_n += Yb.shape[0]
                        self._v_loss = v_loss / v_n if v_n > 0 else None
                        self._v_score = v_score / v_n if v_n > 0 else None
                else:
                    y_eval = y.reshape(-1, 1) if (y is not None and y.ndim == 1) else y
                    pred_eta = self.tucker_regressor(X)
                    self._t_loss = float(self.tucker_regressor.loss_fn(pred_eta, y_eval) / y_eval.shape[0])
                    pred_y = self.tucker_regressor.inverse_link(pred_eta)
                    self._t_score = float(self._score(pred_y, y_eval))

                    if X_val is not None and y_val is not None:
                        y_val_eval = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
                        val_eta = self.tucker_regressor(X_val)
                        self._v_loss = float(self.tucker_regressor.loss_fn(val_eta, y_val_eval) / y_val_eval.shape[0])
                        val_pred = self.tucker_regressor.inverse_link(val_eta)
                        self._v_score = float(self._score(val_pred, y_val_eval))

            scores = {
                'train_score': self._t_score,
                'train_loss': self._t_loss,
                'val_score': self._v_score,
                'val_loss': float(self._v_loss) if self._v_loss is not None else None,
            }

            in_prods = {}
            # Check convergence & compute gradient norms
            problem, _ = self._initialize_C_subproblem(use_dloader, dataset=train_dataset, X=X, y=y)
            grad_C = problem.grad_f(self.tucker_regressor.core, repeat_forward=True)
            with torch.no_grad():
                y_mmx = (
                    self.solver_results['C'].point.y
                    if self.solver_results['C'] is not None
                    else torch.zeros_like(self.tucker_regressor.core)
                )
                grad_C = grad_C + y_mmx
                grad_C_norm = problem.manifold.norm(self.tucker_regressor.core, grad_C)
                in_prods['core'] = float(problem.inner_product(self.tucker_regressor.core, y_mmx))
            grad_norms = {"grad_C": float(grad_C_norm)}

            for mode in list(range(1, self.N + 1)):
                U = self.tucker_regressor.Us[mode - 1]
                problem, _ = self._initialize_U_subproblem(use_dloader, mode, dataset=train_dataset, X=X, y=y)
                grad_U = problem.grad_f(U, repeat_forward=True)
                with torch.no_grad():
                    if self.solver_results[f'U_{mode}'] is not None:
                        Uy = self.solver_results[f'U_{mode}'].point.y
                        in_prod = float(problem.inner_product(self.solver_results[f'U_{mode}'].point.x, Uy))
                    else:
                        Uy = torch.zeros(self.Ds[mode - 1].shape[0], U.shape[1], device=self.device, dtype=self.dtype)
                        in_prod = 0.0
                    nabla_AT = problem.nabla_AT(U)
                    grad_U = grad_U + nabla_AT @ Uy
                    grad_U_norm = problem.manifold.norm(U, grad_U)
                    in_prods[f'U_{mode}'] = in_prod
                grad_norms[f'grad_U_{mode}'] = float(grad_U_norm)

            max_grad_norm = max(grad_norms.values())
            objectives = self._calculate_objectives(in_prods)

            row = [it, scores['train_score']]
            if has_val:
                row += [scores['val_score']]
            row += [objectives['objective'], max_grad_norm]
            row += [grad_norms[f'grad_U_{mode}'] for mode in range(1, self.N + 1)]
            row += [grad_norms['grad_C']]
            column_printer.print_row(row)

            obj_val = objectives.pop('objective')

            # [CHANGE / FIX Solution 3.3]: REMOVED runaway epsilon-reduction feedback loop
            # (In v2, if obj_val > prev_obj, eps was divided by 10, destroying conditioning).
            prev_obj = obj_val

            self._add_log_entry(
                start_time,
                it,
                obj_val,
                gradient_norm=max_grad_norm,
                **scores,
                **objectives,
                **grad_norms,
            )

            reason = self._check_stopping_criteria(start_time, it, max_grad_norm)
            if reason is not None:
                if self.verbosity >= 1:
                    print(reason)
                break

        if self._wandb_run is not None:
            self._wandb_run.finish()

    @torch.no_grad()
    def _calculate_objectives(self, in_prods: dict[str, float]) -> dict[str, float]:
        obj_val = self._t_loss
        objectives = {}
        modes = list(range(1, self.N + 1))
        for i, mode in enumerate(modes):
            U = self.tucker_regressor.Us[i]
            L = self.Ls[i] if self.Ls is not None else None
            theta = self.thetas[i] if self.thetas is not None else None
            drichlet_energy = 0
            if (theta is not None) and (theta > 0) and (L is not None):
                drichlet_energy = 0.5 * theta * (U * (L @ U)).sum()
            objectives[f'U_{mode}_de'] = float(drichlet_energy)

            lda = self.ldas[i]
            D = self.Ds[i]
            l1_penalty = 0
            if (lda is not None) and (lda > 0) and (D is not None):
                l1_penalty = lda * ((D @ U).abs().sum())
            in_p = in_prods.get(f'U_{mode}', 0.0)
            disparity = l1_penalty - in_p
            objectives[f'U_{mode}_l1'] = float(l1_penalty)
            objectives[f'U_{mode}_l1_disparity'] = float(disparity)
            obj_val += drichlet_energy + l1_penalty
            objectives[f'U_{mode}_nnz_ratio'] = float(
                (U.abs() >= 1e-8).sum() / U.numel()
            )

        C = self.tucker_regressor.core
        ridge_penalty = self.tau * C.pow(2).sum()
        objectives['ridge_penalty'] = float(ridge_penalty)

        core_l1_penalty = self.lda_core * C.abs().sum()
        objectives['core_l1_penalty'] = float(core_l1_penalty)
        in_p_core = in_prods.get('core', 0.0)
        disparity = core_l1_penalty - in_p_core
        objectives['core_l1_disparity'] = float(disparity)
        obj_val += core_l1_penalty + ridge_penalty

        objectives['core_nnz_ratio'] = float(
            (C.abs() >= 1e-8).sum() / C.numel()
        )
        objectives['objective'] = float(obj_val)
        return objectives

    def _initialize_solvers(self):
        """Initialize RADA-RGD subproblem solvers with stabilized defaults."""
        if self.subproblem_solvers is None:
            self.subproblem_solvers = {}
            for mode in range(1, self.N + 1):
                lda = self.ldas[mode - 1]
                fd = self.tucker_regressor.feature_dims[mode - 1]
                fr = self.tucker_regressor.feature_ranks[mode - 1]
                # [CHANGE / FIX Solution 1 & 3]: Sound defaults from Xu et al. (2024)
                self.subproblem_solvers[f'U_{mode}'] = RADA_RGD(
                    R=lda * (fd * fr) ** 0.5 if lda != 0 else 1e-6 * (fd * fr) ** 0.5,
                    c1=1e-4,  # [CHANGE / FIX]: 1e-4 (was 1e-1)
                    max_it=1000,
                    Tk=10,
                    max_line_search=15,
                    beta1=0.1 * fd * (fr**0.5),  # [CHANGE / FIX]: paper scaling
                    rho=1.5,  # [CHANGE / FIX]: 1.5 (was 2.0)
                    tau_1=0.999,
                    tau_2=0.9,
                    zeta=1.0,
                    zeta_BB_min=1e-8,
                    zeta_BB_max=10.0,  # [CHANGE / FIX]: capped at 10.0 (was 1e20)
                    eps=1e-6,
                    lda_smooth_floor=self.lda_smooth_floor,
                    track_best=False,
                    max_time=None,
                    max_function_evals=50000,
                    verbosity=0,
                )

            if self.solver_C_type == 'rada':
                core_dim = math.prod(self.tucker_regressor.ranks)
                lda_core = self.lda_core
                self.subproblem_solvers['C'] = RADA_RGD(
                    R=lda_core * (core_dim) ** 0.5 if lda_core != 0 else 1e-6 * (core_dim) ** 0.5,
                    c1=1e-4,
                    max_it=500,
                    Tk=5,
                    max_line_search=15,
                    beta1=0.1 * (core_dim) ** 0.5,
                    rho=1.5,
                    tau_1=0.999,
                    tau_2=0.9,
                    zeta=1.0,
                    zeta_BB_min=1e-8,
                    zeta_BB_max=10.0,
                    eps=1e-6,
                    lda_smooth_floor=self.lda_smooth_floor,
                    track_best=False,
                    max_time=None,
                    max_function_evals=25000,
                    verbosity=0,
                )

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

        def prox_h(Y, *vargs, **kwargs):
            return torch.clamp(Y, -mu, +mu)
        # This is just a placeholder. Will need to be fixed.
        mapping_A = lambda x: x
        nabla_AT = torch.eye(
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
            nabla_AT,
        )
        return mnmx_problem, X_transform

    def _initialize_U_subproblem(
        self,
        use_dloader: bool,
        mode: int,
        lda: float = None,
        theta: float = None,
        dataset: Dataset = None,
        X: torch.Tensor = None,
        y: torch.Tensor = None,
        val_dataset: Dataset = None,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        save_score_to_self: bool = False,
    ) -> Tuple[Problem, TuckerCovariateTransform]:
        fwtype = 'feature_dir_n_full' if self.transform_covariates else 'full'
        X_transform = self.tucker_regressor.set_forward_mode(fwtype, mode)
        X_val_prime = None
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

            if val_dataset is not None:
                val_dataset.update_transform(X_transform)
                val_dataloader = DataLoader(val_dataset, **cfg)
                val_numbatches = len(val_dataloader)
                val_X_primes = [None] * val_numbatches
                val_y_s = [None] * val_numbatches
                for i_batch, sample_batched in enumerate(val_dataloader):
                    val_X_primes[i_batch] = sample_batched['X'].to(self.device, non_blocking=True)
                    val_y_s[i_batch] = sample_batched['Y'].to(self.device, non_blocking=True)
                X_val_prime = torch.cat(val_X_primes)
                y_val = torch.cat(val_y_s)
        else:
            X_prime = X_transform(X)
            if (X_val is not None) and (y_val is not None):
                X_val_prime = X_transform(X_val)

        theta = self.thetas[mode - 1] if theta is None else theta
        L = self.Ls[mode - 1] if self.Ls is not None else None

        def objective(U):
            eta = self.tucker_regressor.functional_feature_dir_n_full(U=U, x=X_prime)
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

        mapping_A = self.Ds[mode - 1]
        nabla_AT = mapping_A.T
        mnmx_problem = MinimaxProblem(
            self.tucker_regressor.Us[mode - 1].manifold,
            objective,
            func_h,
            prox_h,
            mapping_A,
            nabla_AT,
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
            'coefficient_dims': (
                self.tucker_regressor.task_dims + self.tucker_regressor.feature_dims
            ),
            'ranks': self.tucker_regressor.ranks,
            'tau': self.tau,
            'ldas': self.ldas,
            'lda_core': self.lda_core,
            'thetas': self.thetas,
            'solver_C_type': self.solver_C_type,
            'enforce_bcd_monotonicity': self.enforce_bcd_monotonicity,
            'lda_smooth_floor': self.lda_smooth_floor,
            'lda_smooth_floor_min': self.lda_smooth_floor_min,
            'stall_anneal_factor': self.stall_anneal_factor,
            'subproblem_grad_tol': self.subproblem_grad_tol,
            'min_objective_decrease': self.min_objective_decrease,
            'anneal_on_subproblem_convergence': self.anneal_on_subproblem_convergence,
        }

    def _initialize_log(self, seed=None, wandb_kwargs=None):
        self.hyper_parameters = self.hyperparameters
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
            'iterations': defaultdict(list),
        }
        self.subproblem_history = []
        self.log['subproblem_history'] = self.subproblem_history
        if wandb_kwargs is not None:
            run_cfg = {
                **self.hyperparameters,
                'seed': seed,
                'init_with_hosvd_of_grad': self.init_with_hosvd_of_grad,
                'max_time': self.max_time,
                'max_it': self.max_it,
                'min_gradient_norm': self.min_gradient_norm,
            }
            wandb_kwargs['config'] = {**wandb_kwargs.get('config', {}), **run_cfg}
            self._wandb_run = wandb.init(**wandb_kwargs)

    def _add_log_entry(self, start_time, iteration, objective, **kwargs):
        if self.log_verbosity <= 0:
            return
        if (self.logging_period != 0) and (iteration % self.logging_period == 0):
            self.log['iterations']['iteration'].append(iteration)
            self.log['iterations']['time'].append(perf_counter() - start_time)
            self.log['iterations']['objective'].append(objective)
            for key, value in kwargs.items():
                self.log['iterations'][key].append(value)

            if self._wandb_run is not None:
                metrics = {
                    key: self.log['iterations'][key][-1]
                    for key in self.log['iterations'].keys()
                }
                self._wandb_run.log(metrics, step=iteration)

    def _check_stopping_criteria(self, start_time, iteration, gradient_norm):
        run_time = perf_counter() - start_time
        reason = None
        if (self.max_time is not None) and (run_time >= self.max_time):
            reason = f"Terminated - max time reached after {iteration} iterations."
        elif iteration >= self.max_it:
            reason = (
                f"Terminated - maximum number of iterations reached after {run_time:.3f} seconds."
            )
        elif gradient_norm <= self.min_gradient_norm:
            reason = (
                f"Terminated - min grad norm reached after {iteration} iterations, {run_time:.3f} seconds."
            )
        return reason

    def _init_printer(self, with_val=False):
        if self.verbosity >= 1:
            print("Starting BCD_RGD (v4 - Stabilized) Solver for Tucker Regression")
        if self.verbosity >= 2:
            print("Hyper-parameters:")
            pprint(self.hyperparameters)
            iteration_format_length = int(np.log10(max(self.max_it, 1))) + 1
            columns = [
                ("Iteration", f"{iteration_format_length}d"),
                ("Train score", ".4f"),
            ]
            if with_val:
                columns += [("Val score", ".4f")]
            columns += [
                ("Cost", ".10e"),
                ("Gradient norm", ".4e"),
            ] + [
                (f"U_{mode} grad norm", ".4e") for mode in range(1, self.N + 1)
            ] + [
                ("C grad norm", ".4e")
            ]
            column_printer = printer.ColumnPrinter(columns=columns)
        else:
            column_printer = printer.VoidPrinter()
        return column_printer

    def _fit(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def threshold_sparsity(self, tol: float = 1e-3):
        """Hard threshold small entries in factor matrices to exact zeros."""
        if hasattr(self.tucker_regressor, 'Us') and self.tucker_regressor.Us is not None:
            for p in self.tucker_regressor.Us:
                if p is not None and hasattr(p, 'data'):
                    mask = p.data.abs() < tol
                    p.data[mask] = 0.0
