"""Optuna-based Hyper-parameter Searcher for Sparse TuckerRegressorBCD.

Optimizes:
  - Feature factor ranks: (r_1, ..., r_N) from r_{min, i} <= r_i <= r_{max, i}
  - Sparsity regularizers: (lambda_1, ..., lambda_N) (`ldas`)
  - Core ridge penalty: tau

Supports:
  - Input X, y as either np.ndarray or torch.Tensor.
  - Cross-validation with K-Fold, StratifiedKFold, StratifiedShuffleSplit, or hold-out split.
  - Multi-GPU parallel worker execution across specified CUDA devices.
  - Periodic and best-so-far model parameter checkpointing into .npz archive.
  - Best model restoration from disk via `.load_best_model()`.
"""

from copy import deepcopy
from collections import defaultdict
import json
from pathlib import Path
import queue
import threading
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from t_regs.manifolds.steifel import Steifel
from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.solvers.manifold.rada import RADA_RGD


def fused_lasso_matrix_torch(
    n: int, dtype: torch.dtype = torch.float64, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """Constructs the (n-1) x n first-order difference matrix for 1D fused lasso

    as a sparse PyTorch COO tensor.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    m = n - 1
    row_indices = torch.repeat_interleave(torch.arange(m, device=device), 2)
    col_indices = torch.stack(
        [
            torch.arange(m, device=device),
            torch.arange(1, n, device=device),
        ],
        dim=1,
    ).flatten()
    indices = torch.stack([row_indices, col_indices])
    values = torch.tensor([-1.0, 1.0], dtype=dtype, device=device).repeat(m)
    return torch.sparse_coo_tensor(
        indices, values, size=(m, n), dtype=dtype, device=device
    ).coalesce()


def sparse_eye(
    n: int, dtype: torch.dtype = torch.float64, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """Constructs an n x n identity matrix as a sparse PyTorch COO tensor."""
    indices = torch.arange(n, device=device).repeat(2, 1)
    values = torch.ones(n, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=(n, n), dtype=dtype, device=device
    ).coalesce()


class SparseTRegsBCDTuner:
    r"""Optuna-based Hyper-parameter Searcher for Sparse TuckerRegressorBCD.

    Jointly searches for:
      - `feature_ranks`: Tuple of integer factor ranks :math:`(r_1, \dots, r_N)`
        with :math:`r_{\min, i} \le r_i \le r_{\max, i}` for each mode :math:`i \in \{1, \dots, N\}`.
      - `ldas`: Sparsity regularizers :math:`(\lambda_1, \dots, \lambda_N)` for factor matrices.
      - `tau`: Continuous log-uniform core tensor Frobenius ridge penalty :math:`\frac{\tau}{2} \|\mathcal{C}\|_F^2`.

    Parameters
    ----------
    feature_ranks_ranges : Sequence[Tuple[int, int] | Sequence[int]]
        Rank search ranges for each feature mode m=1,...,N.
        Can be (min_rank, max_rank) tuples or sequences of candidate integers.
    ldas_ranges : Sequence[Tuple[float, float] | Sequence[float] | float | None]
        Search ranges or candidate sequences for sparsity penalties lambda_m.
        Can be (lda_min, lda_max) tuples, lists of candidate floats, or a single float for fixed lambda.
    tau_range : Tuple[float, float] | Sequence[float], default=(1e-6, 1e2)
        Candidate range or sequence for core ridge penalty tau.
    log_ldas : bool, default=True
        Whether to sample ldas on a logarithmic scale.
    log_tau : bool, default=True
        Whether to sample tau on a logarithmic scale.
    thetas : Optional[Sequence[float]], default=None
        Fixed smoothness regularizers theta_m (no search over smoothness penalties).
        Defaults to [0.0] * N.
    Ds : Optional[Sequence[Optional[torch.Tensor]]], default=None
        Sparsity / analysis lasso penalty matrices for each feature mode (e.g. sparse_eye, fused_lasso).
        Defaults to identity sparse matrices.
    Ls : Optional[Sequence[Optional[torch.Tensor]]], default=None
        Graph Laplacian matrices for each feature mode (optional, defaults to None).
    devices : Optional[Sequence[Union[str, torch.device]]], default=None
        GPU devices to parallelize trials across (e.g. ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']).
        If None, uses `device`.
    device : str, default='cuda' if torch.cuda.is_available() else 'cpu'
        Default PyTorch device when `devices` is not specified.
    dtype : torch.dtype, default=torch.float64
        Tensor floating-point precision.
    regression_type : str, default='logistic'
        GLM regression family ('logistic', 'linear', 'multinomial', 'poisson').
    task_dims : Sequence[int], default=(1,)
        Dimensions of the regression task.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    cross_validate : bool, default=True
        Whether to perform cross-validation. If False, uses a hold-out split.
    k_fold : int, default=5
        Number of folds for cross-validation when `cross_validate=True`.
    cv : Optional[Any], default=None
        Optional scikit-learn cross-validation splitter (e.g. StratifiedKFold, StratifiedShuffleSplit).
        Overrides `k_fold` if provided.
    val_ratio : float, default=0.2
        Validation split fraction when `cross_validate=False` and no separate validation set is given.
    stratify : bool, default=True
        Whether to stratify splits for classification tasks.
    selection_metric : str, default='val_score'
        Metric to optimize ('val_score' for accuracy / R2, or 'val_loss' for loss).
    direction : Optional[str], default=None
        Optimization direction ('maximize' or 'minimize'). Inferred from `selection_metric` if None.
    n_trials : int, default=50
        Number of Optuna optimization trials to evaluate.
    timeout : Optional[float], default=None
        Time limit in seconds for the Optuna study.
    n_jobs : Optional[int], default=None
        Number of parallel workers. Defaults to len(devices) if devices is provided, else 1.
    sampler : Optional[optuna.samplers.BaseSampler], default=None
        Optuna sampler (defaults to `TPESampler(seed=seed)`).
    pruner : Optional[optuna.pruners.BasePruner], default=None
        Optuna pruner for early stopping unpromising trials.
    checkpoint_path : Union[str, Path], default="best_sparse_tregs_model.npz"
        Disk path where model parameters of the best trial so far are saved as a .npz file.
    base_model_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword arguments passed to :class:`TuckerRegressorBCD`.
    rada_cfg : Optional[Dict[str, Any]], default=None
        Configuration dictionary for factor subproblem solver :class:`RADA_RGD`.
    lbfgs_config : Optional[Dict[str, Any]], default=None
        Configuration dictionary for core tensor L-BFGS solver.
    min_gradient_norm : float, default=1e-5
        Gradient norm stopping criterion for BCD solver.
    max_it : int, default=100
        Maximum iterations for BCD solver.
    subproblem_solvers : Optional[Dict[str, Any]], default=None
        Optional custom dictionary of subproblem solvers.
    verbosity : int, default=1
        Verbosity level (0: silent, 1: summary, 2: detailed per-trial).
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        feature_ranks_ranges: Sequence[Union[Tuple[int, int], Sequence[int]]],
        ldas_ranges: Sequence[Union[Tuple[float, float], Sequence[float], float, None]],
        tau_range: Union[Tuple[float, float], Sequence[float]] = (1e-6, 1e2),
        log_ldas: bool = True,
        log_tau: bool = True,
        thetas: Optional[Sequence[float]] = None,
        Ds: Optional[Sequence[Optional[torch.Tensor]]] = None,
        Ls: Optional[Sequence[Optional[torch.Tensor]]] = None,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float64,
        regression_type: str = "logistic",
        task_dims: Sequence[int] = (1,),
        fit_intercept: bool = True,
        cross_validate: bool = True,
        k_fold: int = 5,
        cv: Optional[Any] = None,
        val_ratio: float = 0.2,
        stratify: bool = True,
        selection_metric: str = "val_score",
        direction: Optional[str] = None,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        n_jobs: Optional[int] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        checkpoint_path: Union[str, Path] = "best_sparse_tregs_model.npz",
        base_model_kwargs: Optional[Dict[str, Any]] = None,
        rada_cfg: Optional[Dict[str, Any]] = None,
        lbfgs_config: Optional[Dict[str, Any]] = None,
        min_gradient_norm: float = 1e-5,
        max_it: int = 100,
        subproblem_solvers: Optional[Dict[str, Any]] = None,
        verbosity: int = 1,
        seed: int = 42,
    ):
        self.feature_ranks_ranges = list(feature_ranks_ranges)
        self.ldas_ranges = list(ldas_ranges)
        self.tau_range = tau_range
        self.log_ldas = log_ldas
        self.log_tau = log_tau

        self.N = len(self.feature_ranks_ranges)
        self.thetas = list(thetas) if thetas is not None else [0.0] * self.N
        self.Ds = list(Ds) if Ds is not None else None
        self.Ls = list(Ls) if Ls is not None else [None] * self.N

        self.devices = list(devices) if devices is not None else None
        self.device = str(device)
        self.dtype = dtype
        self.regression_type = regression_type
        self.task_dims = tuple(task_dims)
        self.fit_intercept = fit_intercept

        self.cross_validate = cross_validate
        self.k_fold = k_fold
        self.cv = cv
        self.val_ratio = val_ratio
        self.stratify = stratify

        self.selection_metric = selection_metric
        if direction is not None:
            self.direction = direction
        else:
            self.direction = (
                "minimize"
                if selection_metric in ("val_loss", "loss", "train_loss", "mean_val_loss")
                else "maximize"
            )

        self.n_trials = n_trials
        self.timeout = timeout
        if n_jobs is not None:
            self.n_jobs = n_jobs
        elif self.devices is not None and len(self.devices) > 0:
            self.n_jobs = len(self.devices)
        else:
            self.n_jobs = 1

        self.seed = seed
        self.sampler = sampler or optuna.samplers.TPESampler(seed=seed)
        self.pruner = pruner
        self.checkpoint_path = Path(checkpoint_path)

        self.base_model_kwargs = dict(base_model_kwargs) if base_model_kwargs is not None else {}
        self.rada_cfg = dict(rada_cfg) if rada_cfg is not None else {}
        self.lbfgs_config = dict(lbfgs_config) if lbfgs_config is not None else {
            "max_iter": 400,
            "tolerance_grad": 1e-7,
            "tolerance_change": 1e-8,
            "line_search_fn": "strong_wolfe",
        }
        self.min_gradient_norm = min_gradient_norm
        self.max_it = max_it
        self.subproblem_solvers = subproblem_solvers
        self.verbosity = verbosity

        # Results state
        self.study: Optional[optuna.Study] = None
        self.best_model: Optional[TuckerRegressorBCD] = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float("-inf") if self.direction == "maximize" else float("inf")
        self.best_trial: Optional[optuna.Trial] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self._test_metrics: Dict[str, float] = {}

        # Concurrency & file locks
        self._checkpoint_lock = threading.Lock()

    def _convert_to_tensor(
        self,
        mat: Any,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert numpy/scipy/torch array to torch.Tensor on target device/dtype."""
        dev = device or self.device
        dt = dtype or self.dtype
        if hasattr(mat, "toarray"):
            mat_dense = mat.toarray()
            t = torch.from_numpy(mat_dense).to(device=dev, dtype=dt)
        elif isinstance(mat, np.ndarray):
            t = torch.from_numpy(mat).to(device=dev, dtype=dt)
        elif isinstance(mat, torch.Tensor):
            t = mat.to(device=dev, dtype=dt)
        else:
            t = torch.as_tensor(mat, device=dev, dtype=dt)
        return t

    def _convert_target_to_tensor(
        self,
        y: Any,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert target array/tensor to torch.Tensor with shape matching (n_samples, *task_dims)."""
        t = self._convert_to_tensor(y, device=device, dtype=dtype)
        expected_shape = (-1,) + self.task_dims
        if t.ndim != len(expected_shape):
            t = t.reshape(expected_shape)
        return t

    def _compute_classification_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute classification accuracy, F1 score, AU-ROC, and AU-PRC.

        Handles both binary logistic regression and multiclass multinomial regression,
        with graceful handling of edge cases (e.g. single-class folds).
        """
        if self.regression_type not in ("logistic", "multinomial"):
            return {
                "accuracy": float("nan"),
                "f1": float("nan"),
                "au_roc": float("nan"),
                "au_prc": float("nan"),
            }

        yt = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
        yp = y_prob.detach().cpu().numpy() if isinstance(y_prob, torch.Tensor) else np.asarray(y_prob)

        metrics: Dict[str, float] = {}

        if self.regression_type == "logistic":
            yt_flat = yt.ravel().astype(int)
            yp_flat = yp.ravel().astype(float)
            y_pred = (yp_flat >= 0.5).astype(int)

            metrics["accuracy"] = float(accuracy_score(yt_flat, y_pred))
            metrics["f1"] = float(f1_score(yt_flat, y_pred, average="binary", zero_division=0))

            unique_classes = np.unique(yt_flat)
            if len(unique_classes) > 1:
                try:
                    metrics["au_roc"] = float(roc_auc_score(yt_flat, yp_flat))
                except Exception:
                    metrics["au_roc"] = float("nan")
                try:
                    metrics["au_prc"] = float(average_precision_score(yt_flat, yp_flat))
                except Exception:
                    metrics["au_prc"] = float("nan")
            else:
                metrics["au_roc"] = float("nan")
                metrics["au_prc"] = float("nan")

        elif self.regression_type == "multinomial":
            yt_flat = yt.ravel().astype(int)
            if yp.ndim == 1:
                yp = yp.reshape(-1, 1)
            y_pred = np.argmax(yp, axis=1)

            metrics["accuracy"] = float(accuracy_score(yt_flat, y_pred))
            metrics["f1"] = float(f1_score(yt_flat, y_pred, average="macro", zero_division=0))

            unique_classes = np.unique(yt_flat)
            if len(unique_classes) > 1:
                try:
                    metrics["au_roc"] = float(roc_auc_score(yt_flat, yp, multi_class="ovr", average="macro"))
                except Exception:
                    metrics["au_roc"] = float("nan")
                try:
                    from sklearn.preprocessing import label_binarize

                    n_classes = yp.shape[1] if yp.ndim > 1 else len(unique_classes)
                    Y_bin = label_binarize(yt_flat, classes=list(range(n_classes)))
                    metrics["au_prc"] = float(average_precision_score(Y_bin, yp, average="macro"))
                except Exception:
                    metrics["au_prc"] = float("nan")
            else:
                metrics["au_roc"] = float("nan")
                metrics["au_prc"] = float("nan")

        return metrics

    def _sample_parameters(self, trial: optuna.Trial) -> Tuple[Tuple[int, ...], List[float], float]:
        """Sample feature factor ranks, sparsity regularizers (ldas), and core ridge penalty (tau)."""
        # 1. Sample factor ranks (r_1, ..., r_N)
        sampled_ranks = []
        for m in range(self.N):
            r_range = self.feature_ranks_ranges[m]
            if (
                isinstance(r_range, (tuple, list))
                and len(r_range) == 2
                and isinstance(r_range[0], int)
                and isinstance(r_range[1], int)
            ):
                rank_m = trial.suggest_int(f"rank_{m+1}", int(r_range[0]), int(r_range[1]))
            else:
                rank_m = trial.suggest_categorical(f"rank_{m+1}", [int(r) for r in r_range])
            sampled_ranks.append(rank_m)

        # 2. Sample sparsity penalties (lambda_1, ..., lambda_N)
        sampled_ldas = []
        for m in range(self.N):
            l_range = self.ldas_ranges[m]
            if l_range is None:
                lda_m = 0.0
            elif isinstance(l_range, (int, float)):
                lda_m = float(l_range)
            elif (
                isinstance(l_range, (tuple, list))
                and len(l_range) == 2
                and isinstance(l_range[0], (int, float))
                and isinstance(l_range[1], (int, float))
            ):
                min_l = float(l_range[0])
                max_l = float(l_range[1])
                lda_m = trial.suggest_float(f"lda_{m+1}", min_l, max_l, log=self.log_ldas)
            else:
                lda_m = trial.suggest_categorical(f"lda_{m+1}", [float(l) for l in l_range])
            sampled_ldas.append(lda_m)

        # 3. Sample core ridge penalty tau
        if (
            isinstance(self.tau_range, (tuple, list))
            and len(self.tau_range) == 2
            and isinstance(self.tau_range[0], (int, float))
            and isinstance(self.tau_range[1], (int, float))
        ):
            tau = trial.suggest_float(
                "tau", float(self.tau_range[0]), float(self.tau_range[1]), log=self.log_tau
            )
        else:
            tau = trial.suggest_categorical("tau", [float(t) for t in self.tau_range])

        return tuple(sampled_ranks), sampled_ldas, tau

    def _instantiate_model(
        self,
        feature_dims: Sequence[int],
        feature_ranks: Sequence[int],
        tau: float,
        ldas: Sequence[float],
        thetas: Optional[Sequence[float]] = None,
        regression_type: Optional[str] = None,
        fit_intercept: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TuckerRegressorBCD:
        """Create a TuckerRegressorBCD instance from tuner configuration."""
        dev = device or self.device
        dt = dtype or self.dtype
        reg_type = regression_type or self.regression_type
        f_intercept = self.fit_intercept if fit_intercept is None else fit_intercept
        th = self.thetas if thetas is None else thetas

        manifolds = [
            Steifel(n=dim, p=rank, device=dev, dtype=dt)
            for dim, rank in zip(feature_dims, feature_ranks)
        ]

        tregs = TuckerRegressor(
            regression_type=reg_type,
            feature_dims=feature_dims,
            feature_ranks=feature_ranks,
            task_dims=self.task_dims,
            task_ranks=self.task_dims,
            feature_manifolds=manifolds,
            seed=self.seed,
        ).to(dev, dtype=dt)

        N = len(feature_dims)
        if self.Ds is not None:
            Ds_dev = [D.to(device=dev, dtype=dt) if isinstance(D, torch.Tensor) else D for D in self.Ds]
        else:
            Ds_dev = [sparse_eye(feature_dims[i], dtype=dt, device=dev) for i in range(N)]

        if self.Ls is not None:
            Ls_dev = [L.to(device=dev, dtype=dt) if isinstance(L, torch.Tensor) else L for L in self.Ls]
        else:
            Ls_dev = [None] * N

        model_kwargs = {
            "tau": tau,
            "ldas": list(ldas),
            "thetas": list(th),
            "Ds": Ds_dev,
            "Ls": Ls_dev,
            "device": dev,
            "dtype": dt,
            "min_gradient_norm": self.min_gradient_norm,
            "max_it": self.max_it,
            "init_with_hosvd_of_grad": False,
            "init_with_lbfgs_full_glm": True,
            "fit_intercept": True,
            "lbfgs_config": dict(self.lbfgs_config),
            "verbosity": max(0, self.verbosity - 1),
        }
        model_kwargs.update(self.base_model_kwargs)

        model = TuckerRegressorBCD(tucker_regressor=tregs, **model_kwargs)

        if self.subproblem_solvers is not None:
            model.subproblem_solvers = deepcopy(self.subproblem_solvers)
        else:
            model._initialize_solvers()
            if self.rada_cfg:
                for solver in model.subproblem_solvers.values():
                    if isinstance(solver, RADA_RGD):
                        for k, v in self.rada_cfg.items():
                            if hasattr(solver, k):
                                setattr(solver, k, v)

        return model

    def _save_checkpoint(
        self,
        filepath: Union[str, Path],
        model: TuckerRegressorBCD,
        hyperparameters: Dict[str, Any],
        trial_metrics: Optional[Dict[str, float]] = None,
    ):
        """Save model parameters and hyperparameter configuration to .npz file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {}

        # 1. Factor matrices model.tucker_regressor.Us
        for m, U in enumerate(model.tucker_regressor.Us):
            save_dict[f"U_{m}"] = U.detach().cpu().numpy()
        save_dict["n_factors"] = np.array(len(model.tucker_regressor.Us))

        # 2. Core tensor model.tucker_regressor.core
        save_dict["core"] = model.tucker_regressor.core.detach().cpu().numpy()

        # 3. Intercept model.tucker_regressor.intercept
        if hasattr(model.tucker_regressor, "intercept") and model.tucker_regressor.intercept is not None:
            save_dict["intercept"] = model.tucker_regressor.intercept.detach().cpu().numpy()
            save_dict["has_intercept"] = np.array(True)
        else:
            save_dict["intercept"] = np.array([])
            save_dict["has_intercept"] = np.array(False)

        # 4. Hyperparameter settings & metadata
        hp_data = dict(hyperparameters)
        if trial_metrics:
            hp_data["metrics"] = trial_metrics
        save_dict["hyperparameters_json"] = np.array(json.dumps(hp_data, default=str))

        for k, v in hp_data.items():
            if isinstance(v, (int, float, str, bool)):
                save_dict[f"hp_{k}"] = np.array(v)
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                save_dict[f"hp_{k}"] = np.array(v)

        np.savez_compressed(filepath, **save_dict)
        if self.verbosity >= 2:
            print(f"[SparseTRegsBCDTuner] Saved best model checkpoint to {filepath}")

    def load_best_model(
        self,
        filepath: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> TuckerRegressorBCD:
        """Load the saved best model so far from disk (.npz file).

        Parameters
        ----------
        filepath : Optional[Union[str, Path]]
            Path to the .npz checkpoint. Defaults to `self.checkpoint_path`.
        device : Optional[Union[str, torch.device]]
            Target PyTorch device. Defaults to `self.device`.

        Returns
        -------
        TuckerRegressorBCD
            Restored model with best hyperparameters and weights.
        """
        path = Path(filepath or self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        target_device = device or self.device
        archive = np.load(path, allow_pickle=True)

        if "hyperparameters_json" in archive:
            hp = json.loads(str(archive["hyperparameters_json"]))
        else:
            hp = {}

        feature_ranks = tuple(int(r) for r in hp.get("feature_ranks", []))
        feature_dims = tuple(int(d) for d in hp.get("feature_dims", []))
        tau = float(hp.get("tau", 1e-5))
        ldas = tuple(float(l) for l in hp.get("ldas", []))
        thetas = tuple(float(t) for t in hp.get("thetas", [0.0] * len(feature_dims)))
        regression_type = str(hp.get("regression_type", self.regression_type))
        fit_intercept = bool(hp.get("fit_intercept", self.fit_intercept))

        n_factors = int(archive["n_factors"]) if "n_factors" in archive else len(feature_ranks)
        Us_np = [archive[f"U_{m}"] for m in range(n_factors)]
        core_np = archive["core"]
        has_intercept = bool(archive["has_intercept"]) if "has_intercept" in archive else False
        intercept_np = archive["intercept"] if has_intercept else None

        resolved_dims = feature_dims or tuple(u.shape[0] for u in Us_np)
        resolved_ranks = feature_ranks or tuple(u.shape[1] for u in Us_np)
        model = self._instantiate_model(
            feature_dims=resolved_dims,
            feature_ranks=resolved_ranks,
            tau=tau,
            ldas=ldas,
            thetas=thetas,
            regression_type=regression_type,
            fit_intercept=fit_intercept,
            device=target_device,
        )

        for m, U_mat in enumerate(Us_np):
            U_tensor = torch.from_numpy(U_mat).to(device=target_device, dtype=self.dtype)
            model.tucker_regressor.Us[m].data.copy_(U_tensor)

        core_tensor = torch.from_numpy(core_np).to(device=target_device, dtype=self.dtype)
        model.tucker_regressor.core.data.copy_(core_tensor)

        if (
            intercept_np is not None
            and hasattr(model.tucker_regressor, "intercept")
            and model.tucker_regressor.intercept is not None
        ):
            ic_tensor = torch.from_numpy(intercept_np).to(device=target_device, dtype=self.dtype)
            model.tucker_regressor.intercept.data.copy_(ic_tensor)

        model._initialize_solvers()
        self.best_model = model
        self.best_params = hp
        if self.verbosity >= 1:
            print(f"[SparseTRegsBCDTuner] Successfully loaded best model from {path}")
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
        storage: Optional[str] = 'sqlite:///optuna_sptregs_tuner.db',
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Execute the Optuna hyperparameter optimization study for TuckerRegressorBCD.

        Parameters
        ----------
        X : Union[torch.Tensor, np.ndarray]
            Covariate tensor data of shape (n_samples, *feature_dims).
        y : Union[torch.Tensor, np.ndarray]
            Response vector/matrix of shape (n_samples,) or (n_samples, *task_dims).
        X_val : Optional[Union[torch.Tensor, np.ndarray]]
            Optional validation covariate tensor.
        y_val : Optional[Union[torch.Tensor, np.ndarray]]
            Optional validation response.
        X_test : Optional[Union[torch.Tensor, np.ndarray]]
            Optional test covariate tensor for final out-of-sample evaluation.
        y_test : Optional[Union[torch.Tensor, np.ndarray]]
            Optional test response for final out-of-sample evaluation.
        test_size : Optional[float]
            Optional train/test split fraction applied before tuning if X_test is not given.
        study_name : Optional[str]
            Name of the Optuna study.
        storage : Optional[str]
            Optuna storage URL (e.g. SQLite database path 'sqlite:///optuna.db').
        checkpoint_path : Optional[Union[str, Path]]
            Custom file path for the .npz checkpoint. Overrides self.checkpoint_path if given.

        Returns
        -------
        pd.DataFrame
            Summary dataframe of all evaluated trials.
        """
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)

        # Convert input data to numpy for flexible CV splitting
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        # Handle optional automatic test split
        if test_size is not None and test_size > 0 and X_test is None:
            stratify_arr = (
                y_np.ravel() if (self.stratify and self.regression_type == "logistic" and len(np.unique(y_np)) > 1) else None
            )
            X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
                X_np, y_np, test_size=test_size, random_state=self.seed, stratify=stratify_arr
            )
            X_np, y_np = X_tr_np, y_tr_np
            X_test = X_te_np
            y_test = y_te_np

        n_samples = X_np.shape[0]
        feature_dims = tuple(X_np.shape[1:])

        # Setup data splits
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        is_fixed_val = False

        if X_val is not None and y_val is not None:
            is_fixed_val = True
            X_v_np = X_val.detach().cpu().numpy() if isinstance(X_val, torch.Tensor) else np.asarray(X_val)
            y_v_np = y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else np.asarray(y_val)
            splits = [(np.arange(n_samples), np.arange(len(X_v_np)))]
        elif not self.cross_validate:
            # Holdout validation split
            if self.stratify and self.regression_type == "logistic" and len(np.unique(y_np)) > 1:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
                tr_idx, v_idx = next(sss.split(X_np, y_np.ravel()))
            else:
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(n_samples)
                n_val = max(1, int(n_samples * self.val_ratio))
                tr_idx, v_idx = perm[n_val:], perm[:n_val]
            splits = [(tr_idx, v_idx)]
        else:
            # Cross validation
            if self.cv is not None:
                splitter = self.cv
            elif self.stratify and self.regression_type == "logistic" and len(np.unique(y_np)) > 1:
                splitter = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)
            else:
                splitter = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)

            for tr_idx, v_idx in splitter.split(X_np, y_np.ravel()):
                splits.append((tr_idx, v_idx))

        # Setup device allocation queue for multi-GPU parallelization
        if self.devices is not None and len(self.devices) > 0:
            device_list = [str(d) for d in self.devices]
        else:
            device_list = [str(self.device)]

        device_pool = queue.Queue()
        for d in device_list:
            device_pool.put(d)

        if self.verbosity >= 1:
            print(
                f"[SparseTRegsBCDTuner] Starting study (trials={self.n_trials}, "
                f"splits={len(splits)}, metric={self.selection_metric}, direction={self.direction})..."
            )
            print(f"  Target Devices: {device_list} | Concurrent Workers: {self.n_jobs}")

        def objective(trial: optuna.Trial) -> float:
            target_device = device_pool.get()
            try:
                ranks, ldas, tau = self._sample_parameters(trial)

                fold_train_scores = []
                fold_train_losses = []
                fold_val_scores = []
                fold_val_losses = []
                fold_val_accs = []
                fold_val_f1s = []
                fold_val_aurocs = []
                fold_val_auprcs = []
                fold_tr_accs = []
                fold_tr_f1s = []
                fold_tr_aurocs = []
                fold_tr_auprcs = []

                t0 = perf_counter()
                last_fold_model = None

                for s_idx, (tr_idx, v_idx) in enumerate(splits):
                    if is_fixed_val:
                        X_tr_t = self._convert_to_tensor(X_np, device=target_device)
                        y_tr_t = self._convert_target_to_tensor(y_np, device=target_device)
                        X_v_t = self._convert_to_tensor(X_v_np, device=target_device)
                        y_v_t = self._convert_target_to_tensor(y_v_np, device=target_device)
                    else:
                        X_tr_t = self._convert_to_tensor(X_np[tr_idx], device=target_device)
                        y_tr_t = self._convert_target_to_tensor(y_np[tr_idx], device=target_device)
                        X_v_t = self._convert_to_tensor(X_np[v_idx], device=target_device)
                        y_v_t = self._convert_target_to_tensor(y_np[v_idx], device=target_device)

                    try:
                        model = self._instantiate_model(
                            feature_dims=feature_dims,
                            feature_ranks=ranks,
                            tau=tau,
                            ldas=ldas,
                            device=target_device,
                        )

                        model.fit(X_tr_t, y_tr_t, X_val=X_v_t, y_val=y_v_t, seed=self.seed)
                        last_fold_model = model

                        fold_train_scores.append(float(model._t_score))
                        fold_train_losses.append(float(model._t_loss))
                        fold_val_scores.append(float(model._v_score))
                        fold_val_losses.append(float(model._v_loss))

                        # Evaluate classification metrics on validation fold
                        prob_v = model.predict(X_v_t).detach().cpu().numpy()
                        y_v_arr = y_v_t.detach().cpu().numpy()
                        val_cls = self._compute_classification_metrics(y_v_arr, prob_v)
                        fold_val_accs.append(val_cls["accuracy"])
                        fold_val_f1s.append(val_cls["f1"])
                        fold_val_aurocs.append(val_cls["au_roc"])
                        fold_val_auprcs.append(val_cls["au_prc"])

                        # Evaluate classification metrics on training fold
                        prob_tr = model.predict(X_tr_t).detach().cpu().numpy()
                        y_tr_arr = y_tr_t.detach().cpu().numpy()
                        tr_cls = self._compute_classification_metrics(y_tr_arr, prob_tr)
                        fold_tr_accs.append(tr_cls["accuracy"])
                        fold_tr_f1s.append(tr_cls["f1"])
                        fold_tr_aurocs.append(tr_cls["au_roc"])
                        fold_tr_auprcs.append(tr_cls["au_prc"])

                    except Exception as e:
                        if self.verbosity >= 1:
                            print(f"[SparseTRegsBCDTuner] Trial {trial.number} failed on split {s_idx}: {e}")
                        raise optuna.exceptions.TrialPruned()

                elapsed = perf_counter() - t0

                mean_val_score = float(np.mean(fold_val_scores))
                mean_val_loss = float(np.mean(fold_val_losses))
                mean_tr_score = float(np.mean(fold_train_scores))
                mean_tr_loss = float(np.mean(fold_train_losses))

                mean_val_acc = float(np.nanmean(fold_val_accs)) if fold_val_accs else float("nan")
                mean_val_f1 = float(np.nanmean(fold_val_f1s)) if fold_val_f1s else float("nan")
                mean_val_auroc = float(np.nanmean(fold_val_aurocs)) if fold_val_aurocs else float("nan")
                mean_val_auprc = float(np.nanmean(fold_val_auprcs)) if fold_val_auprcs else float("nan")

                mean_tr_acc = float(np.nanmean(fold_tr_accs)) if fold_tr_accs else float("nan")
                mean_tr_f1 = float(np.nanmean(fold_tr_f1s)) if fold_tr_f1s else float("nan")
                mean_tr_auroc = float(np.nanmean(fold_tr_aurocs)) if fold_tr_aurocs else float("nan")
                mean_tr_auprc = float(np.nanmean(fold_tr_auprcs)) if fold_tr_auprcs else float("nan")

                # Store classification metrics in user attributes of the Optuna trial
                trial.set_user_attr("accuracy", mean_val_acc)
                trial.set_user_attr("f1", mean_val_f1)
                trial.set_user_attr("au_roc", mean_val_auroc)
                trial.set_user_attr("au_prc", mean_val_auprc)
                trial.set_user_attr("AU-ROC", mean_val_auroc)
                trial.set_user_attr("AU-PRC", mean_val_auprc)

                trial.set_user_attr("val_accuracy", mean_val_acc)
                trial.set_user_attr("val_f1", mean_val_f1)
                trial.set_user_attr("val_au_roc", mean_val_auroc)
                trial.set_user_attr("val_au_prc", mean_val_auprc)
                trial.set_user_attr("val_AU-ROC", mean_val_auroc)
                trial.set_user_attr("val_AU-PRC", mean_val_auprc)

                trial.set_user_attr("mean_val_accuracy", mean_val_acc)
                trial.set_user_attr("mean_val_f1", mean_val_f1)
                trial.set_user_attr("mean_val_auroc", mean_val_auroc)
                trial.set_user_attr("mean_val_auprc", mean_val_auprc)

                trial.set_user_attr("mean_train_accuracy", mean_tr_acc)
                trial.set_user_attr("mean_train_f1", mean_tr_f1)
                trial.set_user_attr("mean_train_auroc", mean_tr_auroc)
                trial.set_user_attr("mean_train_auprc", mean_tr_auprc)

                trial.set_user_attr("mean_train_score", mean_tr_score)
                trial.set_user_attr("mean_train_loss", mean_tr_loss)
                trial.set_user_attr("mean_val_score", mean_val_score)
                trial.set_user_attr("mean_val_loss", mean_val_loss)
                trial.set_user_attr("time", elapsed)
                trial.set_user_attr("device", target_device)

                if self.selection_metric in ("val_score", "score"):
                    target_metric = mean_val_score
                elif self.selection_metric in ("val_accuracy", "accuracy"):
                    target_metric = mean_val_acc if not np.isnan(mean_val_acc) else mean_val_score
                elif self.selection_metric in ("val_f1", "f1"):
                    target_metric = mean_val_f1
                elif self.selection_metric in ("val_auroc", "auroc", "val_au_roc", "au_roc", "AU-ROC"):
                    target_metric = mean_val_auroc
                elif self.selection_metric in ("val_auprc", "auprc", "val_au_prc", "au_prc", "AU-PRC"):
                    target_metric = mean_val_auprc
                elif self.selection_metric in ("val_loss", "loss"):
                    target_metric = mean_val_loss
                else:
                    target_metric = mean_val_score

                # Real-time checkpointing of the best model so far
                with self._checkpoint_lock:
                    is_new_best = False
                    if self.direction == "maximize":
                        if target_metric > self.best_score:
                            self.best_score = target_metric
                            is_new_best = True
                    else:
                        if target_metric < self.best_score:
                            self.best_score = target_metric
                            is_new_best = True

                    if is_new_best and last_fold_model is not None:
                        trial_params = {
                            "feature_ranks": list(ranks),
                            "ldas": list(ldas),
                            "tau": float(tau),
                            "thetas": list(self.thetas),
                            "feature_dims": list(feature_dims),
                            "task_dims": list(self.task_dims),
                            "regression_type": self.regression_type,
                            "fit_intercept": self.fit_intercept,
                            "trial_number": trial.number,
                            "best_score": float(target_metric),
                            "selection_metric": self.selection_metric,
                        }
                        self._save_checkpoint(
                            filepath=self.checkpoint_path,
                            model=last_fold_model,
                            hyperparameters=trial_params,
                            trial_metrics={
                                "mean_val_score": mean_val_score,
                                "mean_val_loss": mean_val_loss,
                                "accuracy": mean_val_acc,
                                "f1": mean_val_f1,
                                "au_roc": mean_val_auroc,
                                "au_prc": mean_val_auprc,
                                "mean_tr_score": mean_tr_score,
                                "mean_tr_loss": mean_tr_loss,
                            },
                        )

                return target_metric

            finally:
                device_pool.put(target_device)

        # Create and execute Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.direction,
            load_if_exists=True,
        )

        optuna.logging.set_verbosity(
            optuna.logging.WARNING if self.verbosity < 2 else optuna.logging.INFO
        )
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
        )

        self.best_trial = self.study.best_trial
        self.best_score = self.best_trial.value
        self.best_params = self.best_trial.params

        if self.verbosity >= 1:
            print(
                f"[SparseTRegsBCDTuner] Study completed! Best Trial: #{self.best_trial.number} "
                f"with {self.selection_metric}={self.best_score:.4f}"
            )
            print(f"  Best Parameters: {self.best_params}")

        # Retrain best model on complete training set
        best_ranks = tuple(int(self.best_params[f"rank_{m+1}"]) for m in range(len(feature_dims)))
        best_ldas = [
            float(self.best_params[f"lda_{m+1}"])
            if f"lda_{m+1}" in self.best_params
            else (float(self.ldas_ranges[m]) if isinstance(self.ldas_ranges[m], (int, float)) else 0.0)
            for m in range(len(feature_dims))
        ]
        best_tau = float(self.best_params["tau"])

        if self.verbosity >= 1:
            print("[SparseTRegsBCDTuner] Retraining best model on full training set...")

        primary_device = device_list[0]
        self.best_model = self._instantiate_model(
            feature_dims=feature_dims,
            feature_ranks=best_ranks,
            tau=best_tau,
            ldas=best_ldas,
            device=primary_device,
        )

        X_full_t = self._convert_to_tensor(X_np, device=primary_device)
        y_full_t = self._convert_target_to_tensor(y_np, device=primary_device)
        X_val_t = self._convert_to_tensor(X_val, device=primary_device) if X_val is not None else None
        y_val_t = self._convert_target_to_tensor(y_val, device=primary_device) if y_val is not None else None

        self.best_model.fit(X_full_t, y_full_t, X_val=X_val_t, y_val=y_val_t, seed=self.seed)

        # Save final retrained model to checkpoint
        self._save_checkpoint(
            filepath=self.checkpoint_path,
            model=self.best_model,
            hyperparameters={
                "feature_ranks": list(best_ranks),
                "ldas": list(best_ldas),
                "tau": float(best_tau),
                "thetas": list(self.thetas),
                "feature_dims": list(feature_dims),
                "task_dims": list(self.task_dims),
                "regression_type": self.regression_type,
                "fit_intercept": self.fit_intercept,
                "best_score": float(self.best_score),
                "trial_number": self.best_trial.number,
                "selection_metric": self.selection_metric,
            },
        )

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            X_test_t = self._convert_to_tensor(X_test, device=primary_device)
            y_test_t = self._convert_target_to_tensor(y_test, device=primary_device)

            pred_test = self.best_model.predict(X_test_t)
            test_prob_np = pred_test.detach().cpu().numpy()
            test_y_np = y_test_t.detach().cpu().numpy()
            test_cls = self._compute_classification_metrics(test_y_np, test_prob_np)

            test_score = float(self.best_model.score(pred_test, y_test_t))
            eta_test = self.best_model.tucker_regressor._fw_full(X_test_t)
            y_target = y_test_t.reshape((-1,) + self.best_model.tucker_regressor.task_dims)
            test_loss = float(self.best_model.tucker_regressor.loss_fn(eta_test, y_target) / y_test_t.shape[0])

            self._test_metrics = {
                "test_score": test_score,
                "test_loss": test_loss,
                "test_accuracy": test_cls["accuracy"],
                "test_f1": test_cls["f1"],
                "test_au_roc": test_cls["au_roc"],
                "test_au_prc": test_cls["au_prc"],
            }
            if self.verbosity >= 1:
                print(
                    f"[SparseTRegsBCDTuner] Test Metrics: Score={test_score:.4f}, Loss={test_loss:.4f}, "
                    f"Acc={test_cls['accuracy']:.4f}, F1={test_cls['f1']:.4f}, "
                    f"AU-ROC={test_cls['au_roc']:.4f}, AU-PRC={test_cls['au_prc']:.4f}"
                )

        # Summary dataframe
        df = self.study.trials_dataframe()
        clean_cols = {col: col.replace("params_", "").replace("user_attrs_", "") for col in df.columns}
        df = df.rename(columns=clean_cols)
        self.summary_df = df

        return self.summary_df

    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Predict responses using the fitted best model."""
        if self.best_model is None:
            raise ValueError("No fitted model found. Run search() or load_best_model() first.")
        X_t = self._convert_to_tensor(X, device=self.best_model.device)
        return self.best_model.predict(X_t)

    def score(self, X: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> float:
        """Score predictions against true targets using the fitted best model."""
        if self.best_model is None:
            raise ValueError("No fitted model found. Run search() or load_best_model() first.")
        X_t = self._convert_to_tensor(X, device=self.best_model.device)
        y_t = self._convert_target_to_tensor(y, device=self.best_model.device)
        pred = self.best_model.predict(X_t)
        return float(self.best_model.score(pred, y_t))

    def plot_search_results(self, save_dir: Optional[Union[str, Path]] = None):
        """Generate and optionally save Optuna optimization history and parameter slice plots."""
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
        curr = float("-inf") if self.direction == "maximize" else float("inf")
        for v in trial_vals:
            curr = max(curr, v) if self.direction == "maximize" else min(curr, v)
            best_so_far.append(curr)

        ax.plot(trial_vals, "o-", alpha=0.5, label="Trial Value")
        ax.plot(best_so_far, "r-", linewidth=2, label="Best So Far")
        ax.set_xlabel("Trial")
        ax.set_ylabel(self.selection_metric)
        ax.set_title(f"Optuna Optimization History ({self.selection_metric})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path / "optuna_history.png", dpi=200)
            plt.close(fig)
        else:
            plt.show()

        # Plot 2: Parameter slices
        params = list(self.best_params.keys())
        n_p = len(params)
        if n_p > 0:
            fig, axes = plt.subplots(1, n_p, figsize=(4 * n_p, 4), squeeze=False)
            for i, p_name in enumerate(params):
                p_vals = [t.params.get(p_name) for t in self.study.trials if t.value is not None]
                scores = [t.value for t in self.study.trials if t.value is not None]
                ax = axes[0, i]
                ax.scatter(p_vals, scores, c="tab:blue", edgecolors="k", alpha=0.7)
                ax.set_xlabel(p_name)
                ax.set_ylabel(self.selection_metric)
                if "tau" in p_name or "lda" in p_name:
                    ax.set_xscale("log")
                ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path / "optuna_param_slices.png", dpi=200)
                plt.close(fig)
            else:
                plt.show()

    def plot_best_factors(
        self,
        freqs: Optional[np.ndarray] = None,
        info: Optional[Any] = None,
        ch_names: Optional[Sequence[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """Plot Mode-1 spatial scalp topomap and Mode-2 spectral factor loadings for the best model."""
        import matplotlib.pyplot as plt

        if self.best_model is None:
            raise ValueError("Best model not fitted. Run search() or load_best_model() first.")

        save_path = Path(save_dir) if save_dir is not None else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        u = self.best_model.tucker_regressor.Us[0].detach().cpu().numpy()
        v = self.best_model.tucker_regressor.Us[1].detach().cpu().numpy()
        r1, r2 = u.shape[1], v.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Mode 1: Spatial factors (topomap if MNE info provided, else line plot)
        if info is not None:
            try:
                import mne

                mne.viz.plot_topomap(
                    u[:, 0],
                    pos=info,
                    axes=axes[0],
                    names=ch_names,
                    sphere="eeg",
                    sensors=True,
                    show=False,
                )
                axes[0].set_title(f"Spatial Factor $U_1$ (Comp 1/{r1})", fontsize=12)
            except Exception as e:
                print(f"Topomap plot fallback to line plot: {e}")
                axes[0].plot(u)
                axes[0].set_xlabel("Channel Index")
                axes[0].set_ylabel("Loading")
                axes[0].set_title(f"Spatial Factor $U_1$ ({r1} components)", fontsize=12)
                axes[0].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[0].plot(u)
            axes[0].set_xlabel("Channel Index")
            axes[0].set_ylabel("Loading")
            axes[0].set_title(f"Spatial Factor $U_1$ ({r1} components)", fontsize=12)
            axes[0].grid(True, linestyle="--", alpha=0.5)

        # Mode 2: Spectral factors
        if freqs is not None:
            axes[1].plot(freqs, v)
            axes[1].set_xlabel("Frequency (Hz)")
        else:
            axes[1].plot(v)
            axes[1].set_xlabel("Feature Index")
        axes[1].set_ylabel("Loading")
        axes[1].set_title(f"Spectral Factor $U_2$ ({r2} components)", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path / "best_model_factors.png", dpi=200)
            plt.close(fig)
        else:
            plt.show()
