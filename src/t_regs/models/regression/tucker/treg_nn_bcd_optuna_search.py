"""Optuna-based Hyper-parameter Search for TuckerRegressorBCD

Optimizes:
  - `feature_ranks`: Tuple of Tucker factor ranks (R_1, ..., R_N)
  - `tau`: Core tensor Frobenius ridge penalty (tau / 2) * ||C||_F^2
  - `alpha`: (Optional) Graph Laplacian scaling factor W_m = I + alpha_m * L_m
             for GeneralizedStiefel feature manifolds.
"""

from copy import deepcopy
from collections import defaultdict
from time import perf_counter
from typing import Optional, Sequence, Union, Dict, Any, Tuple, List
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import optuna

from t_regs.manifolds.steifel import Steifel
from t_regs.manifolds.generalized_steifel import GeneralizedSteifel
from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.solvers.manifold.rada import RADA_RGD


class TuckerBCDOptunaSearch:
    r"""Bayesian and Multi-parameter Optimization for TuckerRegressorBCD using Optuna.

    Jointly searches for:
      - `feature_ranks`: `(R_1, ..., R_N)` integer factor ranks.
      - `tau`: Continuous log-uniform core ridge penalty parameter.
      - `alpha`: Continuous log-uniform graph Laplacian scaling parameters
        $\mathbf{W}_m = (\mathbf{I} + \alpha_m \mathbf{L}_m)$ on :class:`GeneralizedSteifel` manifolds.

    Parameters
    ----------
    feature_ranks_ranges : Sequence[Tuple[int, int] | Sequence[int]]
        Rank search ranges for each feature mode $m=1,\dots,N$.
        Can be specified as `(min_rank, max_rank)` tuples or lists of candidate integers.
    tau_range : Tuple[float, float] | Sequence[float], default=(1e-5, 1e2)
        Candidate range or sequence for core ridge penalty $\tau$.
    log_tau : bool, default=True
        Whether to sample `tau` on a logarithmic scale.
    laplacians : Optional[Sequence[Optional[Any]]], default=None
        Optional sequence of graph Laplacian matrices $\mathbf{L}_m$ for each feature mode.
        Can be dense numpy arrays, scipy sparse matrices, or torch Tensors.
    weights : Optional[Sequence[Optional[Any]]], default=None
        Optional precomputed symmetric positive definite weight matrices $\mathbf{W}_m$.
    alphas : Optional[Sequence[Optional[float]]], default=None
        Optional fixed $\alpha_m$ values for modes where $\mathbf{L}_m$ is provided.
        Defaults to `0.001` if $\mathbf{L}_m$ is provided and no range/value is given.
    alpha_ranges : Optional[Sequence[Optional[Tuple[float, float] | Sequence[float]]]], default=None
        Search range for $\alpha_m$ for each feature mode. If provided for mode $m$,
        Optuna will sample $\alpha_m$ in addition to ranks and $\tau$.
    log_alpha : bool, default=True
        Whether to sample `alpha` on a logarithmic scale.
    retraction : str, default='qr_with_inv_R'
        Retraction algorithm for :class:`GeneralizedSteifel`.
        Options are `'qr_with_inv_R'` and `'qr_with_inv_sqrt_G'`.
    use_sparse_w : bool, default=False
        Whether to convert $\mathbf{W}_m$ to torch sparse CSR format for memory efficiency.
    regression_type : str, default='logistic'
        GLM regression family ('logistic', 'linear', 'multinomial', 'poisson').
    n_trials : int, default=50
        Number of Optuna optimization trials to evaluate.
    timeout : Optional[float], default=None
        Time limit in seconds for the Optuna study.
    sampler : Optional[optuna.samplers.BaseSampler], default=None
        Optuna sampler (defaults to `TPESampler(seed=seed)`).
    pruner : Optional[optuna.pruners.BasePruner], default=None
        Optuna pruner for early stopping unpromising trials.
    cross_validate : bool, default=False
        Whether to perform K-fold cross-validation. If False, uses a hold-out split.
    k_fold : int, default=5
        Number of folds for cross-validation when `cross_validate=True`.
    val_ratio : float, default=0.2
        Validation split fraction when `cross_validate=False` and no separate validation set is given.
    selection_metric : str, default='val_score'
        Metric to optimize ('val_score' for accuracy / R2, or 'val_loss' for loss).
    direction : Optional[str], default=None
        Optimization direction ('maximize' or 'minimize'). Inferred from `selection_metric` if None.
    base_model_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword arguments passed to :class:`TuckerRegressorBCD`
        (e.g., `max_it`, `min_gradient_norm`, `init_with_hosvd_of_grad`).
    rada_cfg : Optional[Dict[str, Any]], default=None
        Configuration dictionary for factor subproblem solver :class:`RADA_RGD`.
    lbfgs_config : Optional[Dict[str, Any]], default=None
        Configuration dictionary for core tensor L-BFGS solver.
    device : str, default='cuda' if torch.cuda.is_available() else 'cpu'
        PyTorch device.
    dtype : torch.dtype, default=torch.float64
        Tensor floating-point precision.
    verbosity : int, default=1
        Verbosity level (0: silent, 1: summary, 2: detailed per-trial).
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        feature_ranks_ranges: Sequence[Union[Tuple[int, int], Sequence[int]]],
        tau_range: Union[Tuple[float, float], Sequence[float]] = (1e-5, 1e2),
        log_tau: bool = True,
        laplacians: Optional[Sequence[Optional[Any]]] = None,
        weights: Optional[Sequence[Optional[Any]]] = None,
        alphas: Optional[Sequence[Optional[float]]] = None,
        alpha_ranges: Optional[Sequence[Optional[Union[Tuple[float, float], Sequence[float]]]]] = None,
        log_alpha: bool = True,
        retraction: str = 'qr_with_inv_R',
        use_sparse_w: bool = False,
        regression_type: str = 'logistic',
        n_trials: int = 50,
        timeout: Optional[float] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        cross_validate: bool = True,
        k_fold: int = 5,
        val_ratio: float = 0.2,
        selection_metric: str = 'val_score',
        direction: Optional[str] = None,
        max_it: Optional[int] = None,
        base_model_kwargs: Optional[Dict[str, Any]] = None,
        rada_cfg: Optional[Dict[str, Any]] = None,
        lbfgs_config: Optional[Dict[str, Any]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype: torch.dtype = torch.float64,
        verbosity: int = 1,
        seed: int = 42,
    ):
        self.feature_ranks_ranges = list(feature_ranks_ranges)
        self.tau_range = tau_range
        self.log_tau = log_tau
        self.laplacians = list(laplacians) if laplacians is not None else [None] * len(self.feature_ranks_ranges)
        self.weights = list(weights) if weights is not None else [None] * len(self.feature_ranks_ranges)
        self.alphas = list(alphas) if alphas is not None else [None] * len(self.feature_ranks_ranges)
        self.alpha_ranges = list(alpha_ranges) if alpha_ranges is not None else [None] * len(self.feature_ranks_ranges)
        self.log_alpha = log_alpha
        self.retraction = retraction
        self.use_sparse_w = use_sparse_w
        self.regression_type = regression_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        self.sampler = sampler or optuna.samplers.TPESampler(seed=seed)
        self.pruner = pruner
        self.cross_validate = cross_validate
        self.k_fold = k_fold
        self.val_ratio = val_ratio
        self.selection_metric = selection_metric

        if direction is not None:
            self.direction = direction
        else:
            self.direction = 'maximize' if selection_metric == 'val_score' else 'minimize'

        self.base_model_kwargs = dict(base_model_kwargs) if base_model_kwargs is not None else {}
        if max_it is not None:
            self.base_model_kwargs['max_it'] = max_it
        self.rada_cfg = dict(rada_cfg) if rada_cfg is not None else {}
        self.lbfgs_config = dict(lbfgs_config) if lbfgs_config is not None else {
            'max_iter': 100,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'line_search_fn': 'strong_wolfe',
        }
        self.device = device
        self.dtype = dtype
        self.verbosity = verbosity

        # Results state
        self.study: Optional[optuna.Study] = None
        self.best_model: Optional[TuckerRegressorBCD] = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float('-inf') if self.direction == 'maximize' else float('inf')
        self.best_trial: Optional[optuna.Trial] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self._test_metrics: Dict[str, float] = {}

    def _convert_to_tensor(self, mat: Any) -> torch.Tensor:
        """Convert numpy/scipy/torch matrix to tensor on target device/dtype."""
        if hasattr(mat, 'toarray'):
            mat_dense = mat.toarray()
            t = torch.from_numpy(mat_dense).to(self.device, self.dtype)
        elif isinstance(mat, np.ndarray):
            t = torch.from_numpy(mat).to(self.device, self.dtype)
        elif isinstance(mat, torch.Tensor):
            t = mat.to(self.device, self.dtype)
        else:
            t = torch.as_tensor(mat, device=self.device, dtype=self.dtype)
        return t

    def _sample_parameters(self, trial: optuna.Trial) -> Tuple[Tuple[int, ...], float, List[float]]:
        """Sample ranks, tau, and alphas for a given trial."""
        n_modes = len(self.feature_ranks_ranges)
        sampled_ranks = []
        for m in range(n_modes):
            r_range = self.feature_ranks_ranges[m]
            if isinstance(r_range, (tuple, list)) and len(r_range) == 2 and isinstance(r_range[0], int) and isinstance(r_range[1], int):
                rank_m = trial.suggest_int(f'rank_{m+1}', r_range[0], r_range[1])
            else:
                rank_m = trial.suggest_categorical(f'rank_{m+1}', list(r_range))
            sampled_ranks.append(rank_m)

        # Sample tau
        if isinstance(self.tau_range, (tuple, list)) and len(self.tau_range) == 2 and isinstance(self.tau_range[0], (int, float)):
            tau = trial.suggest_float('tau', float(self.tau_range[0]), float(self.tau_range[1]), log=self.log_tau)
        else:
            tau = trial.suggest_categorical('tau', [float(t) for t in self.tau_range])

        # Sample alphas
        sampled_alphas = []
        for m in range(n_modes):
            if self.alpha_ranges[m] is not None:
                a_range = self.alpha_ranges[m]
                if isinstance(a_range, (tuple, list)) and len(a_range) == 2:
                    alpha_m = trial.suggest_float(f'alpha_{m+1}', float(a_range[0]), float(a_range[1]), log=self.log_alpha)
                else:
                    alpha_m = trial.suggest_categorical(f'alpha_{m+1}', [float(a) for a in a_range])
            elif self.alphas[m] is not None:
                alpha_m = float(self.alphas[m])
            elif self.laplacians[m] is not None:
                alpha_m = 0.001
            else:
                alpha_m = 0.0
            sampled_alphas.append(alpha_m)

        return tuple(sampled_ranks), tau, sampled_alphas

    def _build_manifolds(
        self,
        feature_dims: Sequence[int],
        feature_ranks: Sequence[int],
        alphas: Sequence[float],
    ) -> List[Any]:
        """Construct Stiefel or GeneralizedStiefel manifolds for each feature mode."""
        manifolds = []
        for m, (dim, rank, alpha) in enumerate(zip(feature_dims, feature_ranks, alphas)):
            W_mat = None
            if self.weights[m] is not None:
                W_mat = self._convert_to_tensor(self.weights[m])
            elif self.laplacians[m] is not None:
                L_tensor = self._convert_to_tensor(self.laplacians[m])
                I_tensor = torch.eye(dim, device=self.device, dtype=self.dtype)
                W_mat = I_tensor + alpha * L_tensor
                W_mat = 0.5 * (W_mat + W_mat.T)

            if W_mat is not None:
                if self.use_sparse_w:
                    W_mat = W_mat.to_sparse_csr()
                manifold = GeneralizedSteifel(
                    n=dim,
                    p=rank,
                    G=W_mat,
                    retraction=self.retraction,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                manifold = Steifel(
                    n=dim,
                    p=rank,
                    device=self.device,
                    dtype=self.dtype,
                )
            manifolds.append(manifold)

        return manifolds

    def _instantiate_model(
        self,
        feature_dims: Sequence[int],
        feature_ranks: Sequence[int],
        tau: float,
        alphas: Sequence[float],
    ) -> TuckerRegressorBCD:
        """Create TuckerRegressorBCD instance with trial configuration."""
        manifolds = self._build_manifolds(feature_dims, feature_ranks, alphas)

        tregs = TuckerRegressor(
            regression_type=self.regression_type,
            feature_dims=feature_dims,
            feature_ranks=feature_ranks,
            feature_manifolds=manifolds,
        ).to(self.device, dtype=self.dtype)

        N = len(feature_dims)
        model_kwargs = {
            'tau': tau,
            'ldas': [0.0] * N,
            'thetas': [0.0] * N,
            'Ls': [None] * N,
            'device': self.device,
            'min_gradient_norm': 1e-5,
            'max_it': 100,
            'init_with_hosvd_of_grad': True,
            'lbfgs_config': dict(self.lbfgs_config),
            'verbosity': 0,
        }
        model_kwargs.update(self.base_model_kwargs)

        model = TuckerRegressorBCD(tucker_regressor=tregs, **model_kwargs)

        # Configure RADA_RGD solvers for each mode
        for mode in range(1, N + 1):
            fd = feature_dims[mode - 1]
            fr = feature_ranks[mode - 1]
            cfg = deepcopy(self.rada_cfg)
            cfg['R'] = cfg.get('R', 0.0)
            if 'beta1' not in cfg:
                cfg['beta1'] = 0.1 * fd * (fr ** 0.5)
            if 'verbosity' not in cfg:
                cfg['verbosity'] = max(0, self.verbosity - 2)
            model.subproblem_solvers[f'U_{mode}'] = RADA_RGD(**cfg)

        return model

    def search(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        X_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
        test_size: Optional[float] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> pd.DataFrame:
        """Execute the Optuna hyperparameter optimization study."""
        # Optional automatic train/test split
        if test_size is not None and test_size > 0 and X_test is None:
            from sklearn.model_selection import train_test_split
            X_np = X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()
            y_np = y if isinstance(y, np.ndarray) else y.detach().cpu().numpy()
            stratify = y_np if self.regression_type == 'logistic' and len(np.unique(y_np)) > 1 else None
            X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
                X_np, y_np, test_size=test_size, random_state=self.seed, stratify=stratify
            )
            X = self._convert_to_tensor(X_tr_np)
            y = self._convert_to_tensor(y_tr_np)
            X_test = self._convert_to_tensor(X_te_np)
            y_test = self._convert_to_tensor(y_te_np)
        else:
            X = self._convert_to_tensor(X)
            y = self._convert_to_tensor(y)
            if X_test is not None:
                X_test = self._convert_to_tensor(X_test)
            if y_test is not None:
                y_test = self._convert_to_tensor(y_test)

        if X_val is not None:
            X_val = self._convert_to_tensor(X_val)
        if y_val is not None:
            y_val = self._convert_to_tensor(y_val)

        n_samples = X.shape[0]
        feature_dims = tuple(X.shape[1:])

        # Set up data splits
        if X_val is not None and y_val is not None:
            splits = [((np.arange(n_samples), np.arange(n_samples)), X, y, X_val, y_val)]
        elif not self.cross_validate:
            np_rng = np.random.default_rng(self.seed)
            perm = np_rng.permutation(n_samples)
            n_val = max(1, int(n_samples * self.val_ratio))
            train_idx, val_idx = perm[n_val:], perm[:n_val]
            splits = [((train_idx, val_idx), X[train_idx], y[train_idx], X[val_idx], y[val_idx])]
        else:
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)
            splits = []
            for tr_idx, v_idx in kf.split(np.arange(n_samples)):
                splits.append(((tr_idx, v_idx), X[tr_idx], y[tr_idx], X[v_idx], y[v_idx]))

        if self.verbosity >= 1:
            print(f"[TuckerBCDOptunaSearch] Starting study (trials={self.n_trials}, metric={self.selection_metric}, direction={self.direction})...")

        def objective(trial: optuna.Trial) -> float:
            ranks, tau, alphas = self._sample_parameters(trial)

            fold_train_scores = []
            fold_train_losses = []
            fold_val_scores = []
            fold_val_losses = []

            t0 = perf_counter()
            for s_idx, (_, X_tr, y_tr, X_v, y_v) in enumerate(splits):
                try:
                    model = self._instantiate_model(feature_dims, ranks, tau, alphas)
                    model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v, seed=self.seed)

                    fold_train_scores.append(float(model._t_score))
                    fold_train_losses.append(float(model._t_loss))
                    fold_val_scores.append(float(model._v_score))
                    fold_val_losses.append(float(model._v_loss))
                except Exception as e:
                    if self.verbosity >= 2:
                        print(f"Trial {trial.number} failed on split {s_idx}: {e}")
                    raise optuna.exceptions.TrialPruned()

            elapsed = perf_counter() - t0

            mean_val_score = float(np.mean(fold_val_scores))
            mean_val_loss = float(np.mean(fold_val_losses))
            mean_tr_score = float(np.mean(fold_train_scores))
            mean_tr_loss = float(np.mean(fold_train_losses))

            # Record attributes
            trial.set_user_attr('mean_train_score', mean_tr_score)
            trial.set_user_attr('mean_train_loss', mean_tr_loss)
            trial.set_user_attr('mean_val_score', mean_val_score)
            trial.set_user_attr('mean_val_loss', mean_val_loss)
            trial.set_user_attr('time', elapsed)

            target_metric = mean_val_score if self.selection_metric == 'val_score' else mean_val_loss
            return target_metric

        # Create and run Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.direction,
            load_if_exists=True,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING if self.verbosity < 2 else optuna.logging.INFO)
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_trial = self.study.best_trial
        self.best_score = self.best_trial.value
        self.best_params = self.best_trial.params

        if self.verbosity >= 1:
            print(f"[TuckerBCDOptunaSearch] Study finished! Best Trial: #{self.best_trial.number} with {self.selection_metric}={self.best_score:.4f}")
            print(f"  Best Parameters: {self.best_params}")

        # Retrain best model on complete training data
        best_ranks = tuple(self.best_params[f'rank_{m+1}'] for m in range(len(feature_dims)))
        best_tau = float(self.best_params['tau'])
        best_alphas = [
            float(self.best_params[f'alpha_{m+1}']) if f'alpha_{m+1}' in self.best_params
            else (float(self.alphas[m]) if self.alphas[m] is not None else 0.001 if self.laplacians[m] is not None else 0.0)
            for m in range(len(feature_dims))
        ]

        if self.verbosity >= 1:
            print("[TuckerBCDOptunaSearch] Retraining best model on full training set...")

        self.best_model = self._instantiate_model(feature_dims, best_ranks, best_tau, best_alphas)
        self.best_model.fit(X, y, X_val=X_val, y_val=y_val, seed=self.seed)

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            pred_test = self.best_model.predict(X_test)
            test_score = float(self.best_model._score(pred_test, y_test))
            eta_test = self.best_model.tucker_regressor._fw_full(X_test)
            test_loss = float(
                self.best_model.tucker_regressor.loss_fn(
                    eta_test,
                    y_test.reshape((-1,) + self.best_model.tucker_regressor.task_dims)
                ) / y_test.shape[0]
            )
            self._test_metrics = {'test_score': test_score, 'test_loss': test_loss}
            if self.verbosity >= 1:
                print(f"[TuckerBCDOptunaSearch] Test Score = {test_score:.4f}, Test Loss = {test_loss:.4f}")

        # Construct summary dataframe
        df = self.study.trials_dataframe()
        clean_cols = {col: col.replace('params_', '').replace('user_attrs_', '') for col in df.columns}
        df = df.rename(columns=clean_cols)
        self.summary_df = df

        return self.summary_df

    def plot_search_results(self, save_dir: Optional[Union[str, Path]] = None):
        """Generate and optionally save Optuna parameter slices and optimization history plots."""
        import matplotlib.pyplot as plt

        if self.study is None:
            raise ValueError("No study found. Run search() first.")

        save_path = Path(save_dir) if save_dir is not None else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        # Plot 1: Optimization History
        fig, ax = plt.subplots(figsize=(8, 5))
        trial_vals = [t.value for t in self.study.trials if t.value is not None]
        best_so_far = []
        curr = float('-inf') if self.direction == 'maximize' else float('inf')
        for v in trial_vals:
            curr = max(curr, v) if self.direction == 'maximize' else min(curr, v)
            best_so_far.append(curr)

        ax.plot(trial_vals, 'o-', alpha=0.5, label='Trial Value')
        ax.plot(best_so_far, 'r-', linewidth=2, label='Best So Far')
        ax.set_xlabel('Trial')
        ax.set_ylabel(self.selection_metric)
        ax.set_title(f'Optuna Optimization History ({self.selection_metric})')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path / 'optuna_history.png', dpi=200)
            plt.close(fig)
        else:
            plt.show()

        # Plot 2: Parameter slices
        params = list(self.best_params.keys())
        n_p = len(params)
        fig, axes = plt.subplots(1, n_p, figsize=(4 * n_p, 4), squeeze=False)
        for i, p_name in enumerate(params):
            p_vals = [t.params.get(p_name) for t in self.study.trials if t.value is not None]
            scores = [t.value for t in self.study.trials if t.value is not None]
            ax = axes[0, i]
            ax.scatter(p_vals, scores, c='tab:blue', edgecolors='k', alpha=0.7)
            ax.set_xlabel(p_name)
            ax.set_ylabel(self.selection_metric)
            if 'tau' in p_name or 'alpha' in p_name:
                ax.set_xscale('log')
            ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path / 'optuna_param_slices.png', dpi=200)
            plt.close(fig)
        else:
            plt.show()
