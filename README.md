# T-Regs
Library of Structured Tensor Learning Algorithms including Tensor Regression, Robust Tensor Decomposition, Dimensionality Reduction and Generalized Multi-Linear Models and related tools such as optimization problem solvers, visualization tools, synthetic data generation scripts.

> ⚠️ For the experiment setup in related to the experiments in `Sparse and Low-Tucker Rank Generalized Multi-Linear Model` please visit [SRT-GMLM](https://github.com/indibi/SRT-GMLM.git). I intend to have everything up by Oct 5th. Apologies for the inconvenience.

<!-- ## Contents: -->
1. **Models:**
    - Sparse and Low-Tucker Rank Generalized Multi-Linear Model (`SRT-GMLM`) is a special configuration of the `TuckerRegressorBCD` class, found in `t_regs.models.regression.tucker_nn_bcd.py`.
2. **Solvers:**
    - The implementation of the Riemannian Alternating Descent Ascent (`RADA`) algorithm can be found in `t_regs.solvers.manifold.rada.py`.


> Many of the algorithms implemented are based on works of other authors, and their work is cited in the python script documentations.