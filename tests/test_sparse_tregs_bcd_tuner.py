"""Test suite for SparseTRegsBCDTuner."""

import os
import sys
import tempfile
from pathlib import Path

# Add src to sys.path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import unittest
import torch
import numpy as np

from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.models.regression.tucker.sparse_tregs_bcd_tuner import (
    SparseTRegsBCDTuner,
    fused_lasso_matrix_torch,
    sparse_eye,
)


class TestSparseTRegsBCDTuner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float64

        # Small synthetic 2-mode tensor dataset:
        # N=30 samples, shape (8, 12)
        self.n_samples = 30
        self.feature_dims = (8, 12)
        self.X_np = np.random.randn(self.n_samples, *self.feature_dims).astype(np.float64)
        # Binary labels for classification
        self.y_np = (np.random.rand(self.n_samples) > 0.5).astype(np.float64)

    def test_helpers(self):
        """Test fused_lasso_matrix_torch and sparse_eye helpers."""
        eye = sparse_eye(5, dtype=self.dtype, device="cpu")
        self.assertEqual(eye.shape, (5, 5))
        self.assertTrue(eye.is_sparse)

        D = fused_lasso_matrix_torch(6, dtype=self.dtype, device="cpu")
        self.assertEqual(D.shape, (5, 6))
        self.assertTrue(D.is_sparse)

    def test_search_numpy_and_cross_validation(self):
        """Test search with numpy arrays, k_fold cross validation, and hyperparameter sampling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_best_model.npz"

            tuner = SparseTRegsBCDTuner(
                feature_ranks_ranges=[(1, 2), (1, 2)],
                ldas_ranges=[(1e-4, 1e-2), (1e-4, 1e-2)],
                tau_range=(1e-6, 1e-2),
                cross_validate=True,
                k_fold=2,
                n_trials=3,
                max_it=5,
                checkpoint_path=ckpt_path,
                device="cpu",
                dtype=self.dtype,
                verbosity=0,
                seed=42,
            )

            df = tuner.search(self.X_np, self.y_np)
            self.assertIsNotNone(df)
            self.assertGreaterEqual(len(df), 3)
            self.assertIsNotNone(tuner.best_model)
            self.assertTrue(ckpt_path.exists())

            # Verify predictions and score
            preds = tuner.predict(self.X_np)
            self.assertEqual(preds.shape[0], self.n_samples)
            score = tuner.score(self.X_np, self.y_np)
            self.assertIsInstance(score, float)

    def test_search_torch_tensor_and_difference_matrices(self):
        """Test search with torch tensors and custom difference matrices (Ds)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_torch_ckpt.npz"

            D1 = sparse_eye(self.feature_dims[0], dtype=self.dtype, device="cpu")
            D2 = fused_lasso_matrix_torch(self.feature_dims[1], dtype=self.dtype, device="cpu")

            tuner = SparseTRegsBCDTuner(
                feature_ranks_ranges=[(1, 2), (1, 2)],
                ldas_ranges=[(1e-5, 1e-2), 1e-4],  # Mode 1 sampled, Mode 2 fixed
                tau_range=(1e-5, 1e-1),
                Ds=[D1, D2],
                cross_validate=False,
                val_ratio=0.3,
                n_trials=2,
                max_it=5,
                checkpoint_path=ckpt_path,
                device="cpu",
                dtype=self.dtype,
                verbosity=0,
                seed=42,
            )

            X_torch = torch.from_numpy(self.X_np).to("cpu", dtype=self.dtype)
            y_torch = torch.from_numpy(self.y_np).to("cpu", dtype=self.dtype)

            df = tuner.search(X_torch, y_torch)
            self.assertIsNotNone(tuner.best_model)
            self.assertTrue(ckpt_path.exists())

    def test_load_best_model_checkpoint(self):
        """Test that saved .npz checkpoint accurately restores model parameters and predictions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.npz"

            tuner = SparseTRegsBCDTuner(
                feature_ranks_ranges=[(1, 2), (1, 2)],
                ldas_ranges=[1e-4, 1e-4],
                tau_range=(1e-5, 1e-3),
                cross_validate=False,
                val_ratio=0.25,
                n_trials=2,
                max_it=5,
                checkpoint_path=ckpt_path,
                device="cpu",
                dtype=self.dtype,
                verbosity=0,
                seed=42,
            )

            tuner.search(self.X_np, self.y_np)
            preds_original = tuner.predict(self.X_np)

            # Load model into a new tuner instance
            tuner_loader = SparseTRegsBCDTuner(
                feature_ranks_ranges=[(1, 2), (1, 2)],
                ldas_ranges=[1e-4, 1e-4],
                device="cpu",
                dtype=self.dtype,
            )
            restored_model = tuner_loader.load_best_model(filepath=ckpt_path, device="cpu")

            self.assertIsNotNone(restored_model)
            self.assertEqual(len(restored_model.tucker_regressor.Us), 2)

            preds_restored = tuner_loader.predict(self.X_np)
            np.testing.assert_allclose(
                preds_original.detach().cpu().numpy(),
                preds_restored.detach().cpu().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_multi_device_pool(self):
        """Test multi-device queue allocation logic."""
        tuner = SparseTRegsBCDTuner(
            feature_ranks_ranges=[(1, 2), (1, 2)],
            ldas_ranges=[1e-4, 1e-4],
            devices=["cpu", "cpu"],
            n_jobs=2,
            cross_validate=False,
            val_ratio=0.25,
            n_trials=2,
            max_it=3,
            device="cpu",
            dtype=self.dtype,
            verbosity=0,
            seed=42,
        )
        df = tuner.search(self.X_np, self.y_np)
        self.assertIsNotNone(df)
        self.assertGreaterEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()

