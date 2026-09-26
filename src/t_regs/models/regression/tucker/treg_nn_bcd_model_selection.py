"""Hyper-parameter search and regularization path selection for TuckerRegressorBCD."""

from copy import deepcopy
from time import perf_counter
from typing import Optional, Sequence, Dict, Any

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import wandb

from t_regs.utils import printer
from t_regs.solvers.manifold.rada import RADA_RGD
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor


class TuckerBCDRegularizerSearch:
    r"""Sequential Regularization Path Search for Sparse, Low-Rank Tucker GLMs.

    Optimizes the hyper-parameters of :class:`TuckerRegressorBCD`:
      - :math:`\tau`: Ridge penalty on core tensor :math:`\frac{\tau}{2} \|\mathcal{C}\|_F^2`
      - :math:`\lambda_m, m=1,\dots,N`: Sparsity penalties on factor matrices :math:`\sum_m \lambda_m \|D_m \mathbf{U}_m\|_1`
      - :math:`\theta_m, m=1,\dots,N`: Smoothness penalties on factor matrices (optional)

    The search proceeds along a warm-started path:
      1. Baseline unregularized model: :math:`\tau=0, \boldsymbol{\lambda}=\mathbf{0}` (`Tucker`).
      2. Core ridge parameter search: Sweeps :math:`\tau \in \text{tau\_range}` with warm-starting (`RidgeTucker`).
      3. Factor sparsity parameter search: Sequentially sweeps :math:`\lambda_m \in \text{ldas\_ranges}[m-1]`
         for :math:`m=1,\dots,N`, carrying forward the optimal :math:`\lambda_k^*` from prior modes (`SparseRidgeTucker`).

    Parameters
    ----------
    tau_range: Sequence[float]
        Candidate values for core ridge penalty parameter :math:`\tau`.
    ldas_ranges: Sequence[Sequence[float]]
        Candidate values for factor sparsity parameters :math:`\lambda_m` for each mode :math:`m=1,\dots,N`.
    rada_cfg: Optional[dict]
        Configuration dictionary or DictConfig for factor subproblem solver :class:`RADA_RGD`.
    lbfgs_config: Optional[dict]
        Configuration dictionary for core tensor L-BFGS solver.
    thetas_ranges: Optional[Sequence[Sequence[float]]]
        Candidate values for factor smoothness parameters :math:`\theta_m` (optional).
    cross_validate: bool = False
        Whether to perform K-fold cross-validation. If False, uses a hold-out validation split.
    k_fold: int = 5
        Number of folds for cross-validation when `cross_validate=True`.
    val_ratio: float = 0.2
        Validation split fraction when `cross_validate=False` and separate validation set is not provided.
    selection_metric: str = 'val_score'
        Metric to maximize during hyperparameter selection ('val_score' or 'val_loss').
    verbosity: int = 0
        Verbosity level (0: silent, 1: summary, 2: detailed per-step).
    """

    def __init__(
        self,
        tau_range: Sequence[float],
        ldas_ranges: Sequence[Sequence[float]],
        rada_cfg: Optional[Dict[str, Any]] = None,
        lbfgs_config: Optional[Dict[str, Any]] = None,
        thetas_ranges: Optional[Sequence[Sequence[float]]] = None,
        cross_validate: bool = False,
        k_fold: int = 5,
        val_ratio: float = 0.2,
        selection_metric: str = 'val_loss',
        verbosity: int = 0,
    ):
        self.tau_range = list(tau_range)
        self.ldas_ranges = [list(r) for r in ldas_ranges]
        self.thetas_ranges = [list(r) for r in thetas_ranges] if thetas_ranges is not None else None
        self.rada_cfg = dict(rada_cfg) if rada_cfg is not None else {}
        self.lbfgs_config = dict(lbfgs_config) if lbfgs_config is not None else {
            'max_iter': 100,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'line_search_fn': 'strong_wolfe',
        }
        self.cross_validate = cross_validate
        self.k_fold = k_fold
        self.val_ratio = val_ratio
        self.selection_metric = selection_metric
        self.verbosity = verbosity

        self.best_models: Dict[str, Any] = {}
        self.summary_df: Optional[pd.DataFrame] = None
        self._wandb_run = None
        self._wandb_table = None
        self._control_vars: Dict[str, Any] = {}
        self._columns: list[str] = []

    def _initialize_log(
        self,
        control_vars: Optional[dict] = None,
        wandb_kwargs: Optional[dict] = None,
        eval_gt: bool = False,
    ):
        if control_vars is not None:
            self._control_vars = dict(control_vars)
            control_keys = list(control_vars.keys())
        else:
            self._control_vars = {}
            control_keys = []

        n_modes = len(self.ldas_ranges)
        hparam_cols = ['tau'] + [f'lda_{i+1}' for i in range(n_modes)] + ['best_C']
        metric_cols = [
            'test_score', 'test_loss',
            'mean_val_score', 'std_val_score',
            'mean_val_loss', 'std_val_loss',
            'mean_train_score', 'std_train_score',
            'mean_train_loss', 'std_train_loss',
            'time',
        ]
        sparsity_cols = [f'U_{i+1}_nnz_ratio' for i in range(n_modes)] + ['B_nnz_ratio']
        gt_cols = ['rel_coef_error'] + [f'U_{i+1}_subspace_dist' for i in range(n_modes)] if eval_gt else []

        self._columns = control_keys + ['Model'] + hparam_cols + metric_cols + sparsity_cols + gt_cols
        self.summary_df = pd.DataFrame(columns=self._columns)

        if wandb_kwargs is not None and wandb.run is None:
            w_kwargs = deepcopy(wandb_kwargs)
            if 'tags' not in w_kwargs:
                w_kwargs['tags'] = []
            if 'TuckerBCDRegularizerSearch' not in w_kwargs['tags']:
                w_kwargs['tags'].append('TuckerBCDRegularizerSearch')
            w_kwargs['tags'] = [x for x in w_kwargs['tags'] if x!='lv1']
            w_kwargs['tags'].append('lv2')
            self._wandb_run = wandb.init(**w_kwargs)
            self._wandb_table = wandb.Table(columns=self._columns,
                                            log_mode="MUTABLE")

    def _add_log_entry(self, model_name: str, tau: float, ldas: Sequence[float], **kwargs):
        row = {
            'Model': model_name,
            'tau': float(tau),
            **{f'lda_{i+1}': float(lda) for i, lda in enumerate(ldas)},
            'best_C': np.nan,
            **self._control_vars,
            **kwargs,
        }
        clean_row = {col: row.get(col, np.nan) for col in self._columns}
        self.summary_df.loc[len(self.summary_df)] = clean_row

        if self._wandb_run is not None:
            try:
                self._wandb_table.add_data(*[clean_row[col] for col in self._columns])
                self._wandb_run.log({'summary_table': self._wandb_table})
            except Exception:
                pass

    def _create_model(
        self,
        base_model: TuckerRegressorBCD,
        tau: float,
        ldas: Sequence[float],
        thetas: Optional[Sequence[float]] = None,
        warm_start_from: Optional[TuckerRegressorBCD] = None,
    ) -> TuckerRegressorBCD:
        """Create and configure a TuckerRegressorBCD instance with warm start."""
        if warm_start_from is not None:
            model = deepcopy(warm_start_from)
            # model.init_with_hosvd_of_grad = False
        else:
            model = deepcopy(base_model)

        model.tau = float(tau)
        model.ldas = [float(lda) for lda in ldas]
        if thetas is not None:
            model.thetas = [float(theta) for theta in thetas]

        model.lbfgs_config = dict(self.lbfgs_config)

        N = model.tucker_regressor.covariant_degree
        for mode in range(1, N + 1):
            lda = model.ldas[mode-1]
            fd = model.tucker_regressor.feature_dims[mode-1]
            fr = model.tucker_regressor.feature_ranks[mode-1]

            cfg = deepcopy(self.rada_cfg)
            cfg['R'] = float(lda * (fd * fr) ** 0.5) if lda != 0 else 0.0
            if 'beta1' not in cfg:
                cfg['beta1'] = 0.1 * fd * (fr ** 0.5)

            if 'verbosity' not in cfg:
                cfg['verbosity'] = max(0, self.verbosity - 1)

            model.subproblem_solvers[f'U_{mode}'] = RADA_RGD(**cfg)

            if (
                warm_start_from is not None
                and hasattr(warm_start_from, 'solver_results')
                and warm_start_from.solver_results.get(f'U_{mode}') is not None
            ):
                res = deepcopy(warm_start_from.solver_results[f'U_{mode}'])
                if hasattr(res, 'point') and hasattr(res.point, 'y') and res.point.y is not None:
                    if lda > 0:
                        res.point.y = torch.clamp(res.point.y, -lda, lda)
                    else:
                        res.point.y = torch.zeros_like(res.point.y)
                model.solver_results[f'U_{mode}'] = res

        return model

    @torch.no_grad()
    def _evaluate_metrics(
        self,
        model: TuckerRegressorBCD,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        B_gt: Optional[torch.Tensor] = None,
        Us_gt: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute test score, loss, sparsity, and ground truth recovery."""
        metrics: Dict[str, float] = {}

        if X_test is not None and y_test is not None:
            pred_test = model.predict(X_test)
            test_score = model._score(pred_test, y_test)
            eta_test = model.tucker_regressor._fw_full(X_test)
            test_loss = float(
                model.tucker_regressor.loss_fn(
                    eta_test,
                    y_test.reshape((-1,) + model.tucker_regressor.task_dims)
                ) / y_test.shape[0]
            )
            metrics['test_score'] = float(test_score)
            metrics['test_loss'] = float(test_loss)

        N = model.tucker_regressor.covariant_degree
        for m in range(1, N + 1):
            U = model.tucker_regressor.Us[m - 1]
            nnz = float((U.abs() >= 1e-5).sum() / U.numel())
            metrics[f'U_{m}_nnz_ratio'] = nnz

        B_hat = model.tucker_regressor.expanded_form()
        metrics['B_nnz_ratio'] = float((B_hat.abs() >= 1e-5).sum() / B_hat.numel())

        if B_gt is not None:
            B_gt_dev = torch.as_tensor(B_gt, device=B_hat.device, dtype=B_hat.dtype)
            err = torch.linalg.norm(B_hat - B_gt_dev) / torch.linalg.norm(B_gt_dev) # pylint: disable=not-callable
            metrics['rel_coef_error'] = float(err.item())

        if Us_gt is not None:
            for m in range(1, N + 1):
                U_hat = model.tucker_regressor.Us[m - 1]
                U_true = torch.as_tensor(Us_gt[m - 1], device=U_hat.device, dtype=U_hat.dtype)
                P_hat = U_hat @ U_hat.T
                P_true = U_true @ U_true.T
                r = U_hat.shape[1]
                chordal_dist = torch.linalg.norm(P_hat - P_true) / ((2.0 * r) ** 0.5)   # pylint: disable=not-callable
                metrics[f'U_{m}_subspace_dist'] = float(chordal_dist.item())

        return metrics

    def search(
        self,
        base_model: TuckerRegressorBCD,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        B_gt: Optional[torch.Tensor] = None,
        Us_gt: Optional[Sequence[torch.Tensor]] = None,
        control_vars: Optional[dict] = None,
        wandb_kwargs: Optional[dict] = None,
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """Run the complete sequential regularization path search."""
        eval_gt = B_gt is not None
        self._initialize_log(control_vars=control_vars, wandb_kwargs=wandb_kwargs, eval_gt=eval_gt)
        
        n_samples = X.shape[0]
        if X_val is not None and y_val is not None:
            splits = [((np.arange(n_samples), np.arange(n_samples)), X, y, X_val, y_val)]
        elif not self.cross_validate:
            np_rng = np.random.default_rng(seed)
            perm = np_rng.permutation(n_samples)
            n_val = max(1, int(n_samples * self.val_ratio))
            train_idx, val_idx = perm[n_val:], perm[:n_val]
            splits = [((train_idx, val_idx), X[train_idx], y[train_idx], X[val_idx], y[val_idx])]
        else:
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=seed)
            splits = []
            for tr_idx, val_idx in kf.split(np.arange(n_samples)):
                splits.append(((tr_idx, val_idx), X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]))

        N = base_model.tucker_regressor.covariant_degree
        zero_ldas = [0.0] * N

        # =========================================================================
        # Stage 1: Unregularized Base Model (Tucker)
        # =========================================================================
        if self.verbosity >= 1:
            print("\n[TuckerBCDRegularizerSearc] Step 1/3: Fitting Unregularized Tucker Baseline...")

        tucker_results = []
        tucker_models = []
        start_time = perf_counter()

        for i, split_info in enumerate(splits):
            _, X_tr, y_tr, X_v, y_v = split_info
            model = self._create_model(base_model, tau=0.0, ldas=zero_ldas, warm_start_from=None)

            if wandb_kwargs is not None:
                pwandb_kwargs = deepcopy(wandb_kwargs)
                pwandb_kwargs['name'] = wandb_kwargs['name'] + f'-Tucker-f{i}'
                if pwandb_kwargs.get('tags') is None:
                    pwandb_kwargs['tags'] = []
                pwandb_kwargs['tags'] = [x for x in pwandb_kwargs['tags'] if x != "lv2"]
                pwandb_kwargs['tags'].append('lv3')
            else:
                pwandb_kwargs = None
            model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v, seed=seed,
                      wandb_kwargs=pwandb_kwargs
                      )
            tucker_models.append(model)
            tucker_results.append({
                'train_score': float(model._t_score),
                'train_loss': float(model._t_loss),
                'val_score': float(model._v_score),
                'val_loss': float(model._v_loss),
            })

        t_elapsed = perf_counter() - start_time
        best_split_idx = int(np.argmax([r['val_score'] for r in tucker_results]))
        best_tucker_model = tucker_models[best_split_idx]
        self.best_models['Tucker'] = best_tucker_model

        test_metrics = self._evaluate_metrics(best_tucker_model, X_test, y_test, B_gt, Us_gt)
        self._add_log_entry(
            model_name='Tucker',
            tau=0.0,
            ldas=zero_ldas,
            mean_train_score=float(np.mean([r['train_score'] for r in tucker_results])),
            std_train_score=float(np.std([r['train_score'] for r in tucker_results])),
            mean_train_loss=float(np.mean([r['train_loss'] for r in tucker_results])),
            std_train_loss=float(np.std([r['train_loss'] for r in tucker_results])),
            mean_val_score=float(np.mean([r['val_score'] for r in tucker_results])),
            std_val_score=float(np.std([r['val_score'] for r in tucker_results])),
            mean_val_loss=float(np.mean([r['val_loss'] for r in tucker_results])),
            std_val_loss=float(np.std([r['val_loss'] for r in tucker_results])),
            time=t_elapsed,
            **test_metrics,
        )

        if self.verbosity >= 1:
            print(f"  -> Best Tucker: Val Score = {np.mean([r['val_score'] for r in tucker_results]):.4f}, "
                  f"Test Score = {test_metrics.get('test_score', np.nan):.4f}")

        # =========================================================================
        # Stage 2: Core Ridge Path Search (RidgeTucker)
        # =========================================================================
        if self.verbosity >= 1:
            print(f"\n[TuckerBCDRegularizerSearch] Step 2/3: Searching Core Ridge Path ({len(self.tau_range)} tau values)...")

        ridge_val_scores = np.zeros(len(self.tau_range))
        ridge_val_losses = np.zeros(len(self.tau_range))
        ridge_train_scores = np.zeros(len(self.tau_range))
        ridge_train_losses = np.zeros(len(self.tau_range))
        ridge_fitted_models = [[] for _ in range(len(self.tau_range))]

        start_time = perf_counter()
        for s_idx, split_info in enumerate(splits):
            _, X_tr, y_tr, X_v, y_v = split_info
            current_model = tucker_models[s_idx]

            for t_idx, tau in enumerate(self.tau_range):
                model = self._create_model(
                    base_model,
                    tau=tau,
                    ldas=zero_ldas,
                    warm_start_from=None,
                )

                if wandb_kwargs is not None:
                    pwandb_kwargs = deepcopy(wandb_kwargs)
                    pwandb_kwargs['name'] = wandb_kwargs['name'] + f'-TuckerRidge-f{i}' + f"-tau{t_idx}"
                    if pwandb_kwargs.get('tags') is None:
                        pwandb_kwargs['tags'] = []
                    pwandb_kwargs['tags'] = [x for x in pwandb_kwargs['tags'] if x != "lv2"]
                    pwandb_kwargs['tags'].append('lv3')
                else:
                    pwandb_kwargs = None
                model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v, seed=seed,
                          wandb_kwargs=pwandb_kwargs)
                ridge_fitted_models[t_idx].append(model)
                ridge_val_scores[t_idx] += float(model._v_score)
                ridge_val_losses[t_idx] += float(model._v_loss)
                ridge_train_scores[t_idx] += float(model._t_score)
                ridge_train_losses[t_idx] += float(model._t_loss)
                current_model = model

        n_splits = len(splits)
        ridge_val_scores /= n_splits
        ridge_val_losses /= n_splits
        ridge_train_scores /= n_splits
        ridge_train_losses /= n_splits
        t_elapsed = perf_counter() - start_time

        if self.selection_metric == 'val_loss':
            best_tau_idx = int(np.argmin(ridge_val_losses))
        else:
            best_tau_idx = int(np.argmax(ridge_val_scores))

        best_tau = float(self.tau_range[best_tau_idx])
        best_ridge_model = ridge_fitted_models[best_tau_idx][best_split_idx]
        self.best_models['RidgeTucker'] = best_ridge_model

        test_metrics = self._evaluate_metrics(best_ridge_model, X_test, y_test, B_gt, Us_gt)
        self._add_log_entry(
            model_name='RidgeTucker',
            tau=best_tau,
            ldas=zero_ldas,
            mean_train_score=float(ridge_train_scores[best_tau_idx]),
            std_train_score=0.0,
            mean_train_loss=float(ridge_train_losses[best_tau_idx]),
            std_train_loss=0.0,
            mean_val_score=float(ridge_val_scores[best_tau_idx]),
            std_val_score=0.0,
            mean_val_loss=float(ridge_val_losses[best_tau_idx]),
            std_val_loss=0.0,
            time=t_elapsed,
            **test_metrics,
        )

        if self.verbosity >= 1:
            print(f"  -> Best RidgeTucker (tau={best_tau:g}): Val Score = {ridge_val_scores[best_tau_idx]:.4f}, "
                  f"Test Score = {test_metrics.get('test_score', np.nan):.4f}")

        # =========================================================================
        # Stage 3: Sequential Factor Sparsity Search (SparseRidgeTucker)
        # =========================================================================
        current_ldas = [0.0] * N
        current_models = [ridge_fitted_models[best_tau_idx][s_idx] for s_idx in range(n_splits)]

        for m in range(1, N + 1):
            lda_grid = self.ldas_ranges[m - 1]
            if self.verbosity >= 1:
                print(f"\n[TuckerBCDRegularizerSearch] Step 3.{m}/{N}: Sweeping mode-{m} sparsity ({len(lda_grid)} lda values)...")

            lda_val_scores = np.zeros(len(lda_grid))
            lda_val_losses = np.zeros(len(lda_grid))
            lda_train_scores = np.zeros(len(lda_grid))
            lda_train_losses = np.zeros(len(lda_grid))
            lda_fitted_models = [[] for _ in range(len(lda_grid))]

            start_time = perf_counter()
            for s_idx, split_info in enumerate(splits):
                _, X_tr, y_tr, X_v, y_v = split_info
                warm_model = current_models[s_idx]

                for l_idx, lda in enumerate(lda_grid):
                    eval_ldas = list(current_ldas)
                    eval_ldas[m - 1] = float(lda)

                    model = self._create_model(
                        base_model,
                        tau=best_tau,
                        ldas=eval_ldas,
                        warm_start_from=None,
                    )
                    if wandb_kwargs is not None:
                        pwandb_kwargs = deepcopy(wandb_kwargs)
                        pwandb_kwargs['name'] = wandb_kwargs['name'] + f'-SparseTuckerRidge-{m}-f{s_idx}' + f"-lda{l_idx}"
                        if pwandb_kwargs.get('tags') is None:
                            pwandb_kwargs['tags'] = []
                        pwandb_kwargs['tags'] = [x for x in pwandb_kwargs['tags'] if x != "lv2"]
                        pwandb_kwargs['tags'].append('lv3')
                    else:
                        pwandb_kwargs = None
                    model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v, seed=seed,
                              wandb_kwargs=pwandb_kwargs)
                    lda_fitted_models[l_idx].append(model)
                    lda_val_scores[l_idx] += float(model._v_score)
                    lda_val_losses[l_idx] += float(model._v_loss)
                    lda_train_scores[l_idx] += float(model._t_score)
                    lda_train_losses[l_idx] += float(model._t_loss)
                    warm_model = model

            lda_val_scores /= n_splits
            lda_val_losses /= n_splits
            lda_train_scores /= n_splits
            lda_train_losses /= n_splits
            t_elapsed = perf_counter() - start_time

            if self.selection_metric == 'val_loss':
                best_lda_idx = int(np.argmin(lda_val_losses))
            else:
                best_lda_idx = int(np.argmax(lda_val_scores))

            best_lda = float(lda_grid[best_lda_idx])
            current_ldas[m - 1] = best_lda
            current_models = [lda_fitted_models[best_lda_idx][s_idx] for s_idx in range(n_splits)]

            best_mode_model = current_models[best_split_idx]
            model_tag = f'SparseRidgeTucker-{m}'
            self.best_models[model_tag] = best_mode_model

            test_metrics = self._evaluate_metrics(best_mode_model, X_test, y_test, B_gt, Us_gt)
            self._add_log_entry(
                model_name=model_tag,
                tau=best_tau,
                ldas=current_ldas,
                mean_train_score=float(lda_train_scores[best_lda_idx]),
                std_train_score=0.0,
                mean_train_loss=float(lda_train_losses[best_lda_idx]),
                std_train_loss=0.0,
                mean_val_score=float(lda_val_scores[best_lda_idx]),
                std_val_score=0.0,
                mean_val_loss=float(lda_val_losses[best_lda_idx]),
                std_val_loss=0.0,
                time=t_elapsed,
                **test_metrics,
            )

            if self.verbosity >= 1:
                print(f"  -> Best {model_tag} (lda_{m}={best_lda:g}): Val Score = {lda_val_scores[best_lda_idx]:.4f}, "
                      f"Test Score = {test_metrics.get('test_score', np.nan):.4f}")

        # Record final SparseRidgeTucker
        self.best_models['SparseRidgeTucker'] = current_models[best_split_idx]

        if self._wandb_run is not None:
            self._wandb_run.finish()

        return self.summary_df
