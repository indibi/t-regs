# T-Regs Project To-Do:

## Demonstrations
Create model demonstrations for,
- [ ] Regression
- [ ] Tensor Recovery
- [ ] Anomaly Detection

## Documentation
- [ ] Writing a document showing the repository structure and available algorithms implemented
- [ ] Creating a better logical structure for the repository that makes a distinction between models and decomposition algorithms without overly complicating it.
- [ ] Creating artwork for the repository.
- [ ] Creating a library documentation.
- [ ] Converting the docstyle to NumPy and ensuring consistency.
- [ ] Re-structuring the `todo.md` file.

## Software
### Models
- `models/`
  - `matrix_decomp/`
    - `nmf/`: <span style="color:gray;">% Non-negative Matrix Factorizations</span>
    - `pmf/`: <span style="color:gray;">% Penalized Matrix Factorizations</span>
      - Sparse SVD - Power Algorithm
      - [ ] SPC
      - [ ] PMD(FL) <span style="color:gray;">% Penalized Matrix Decomposition with Fused Lasso penalty</span>
      - [ ] Implementing (generalized) fused-lasso subproblem
      - [ ] Sparse and Functional PCA (`rank1_sfpca`)
      - `deflation.py`
    - `gpmf/` <span style="color:gray;">% Generalized Penalized Matrix Factorizations</span>
    - `generalized_svd.py` <span style="color:gray;">% Generalized Least Squares Matrix Decomposition GMD (Allen et. al.) </span>
      - [ ] Tested
      - [ ] Fit score implemented
      <!--
      - Methods:
        - `.__init__()`
        - `.__call__()`
        - `.score()`
        - `.fit()`
      - Attributes:
        - `.asdf`
      -->
  - `tensor_decomp/`
    - `tucker/`
      - HoSVD: <span style="color:gray;">% Higher-order Singular Value Decomposition</span>
      - HooI: <span style="color:gray;">% Higher-order orthogonal Iteration</span>
      - Sparse HoSVD:
      - Sparse HooI:
    - `cp/`
      - CP-ALS: <span style="color:gray;">% Alternating Least Squares Algorithm</span>
        - [ ] Tested
        - [ ] Fit score implemented
      - CP-TPA: <span style="color:gray;">% Tensor Power Algorithm</span>
        - [ ] Tested
        - [ ] Fit score implemented
      - Sparse CP-TPA: <span style="color:gray;">% Sparse Tensor Power Algorithm</span>
        - [ ] Tested
        - [ ] Fit score implemented
  - `regression/`
    - RegressionBaseClass
    - LinearRegression
      - Unregularized
        - [x] Proximal Gradient
          - [x] Tested
        - [ ] ADMM
      - Lasso
      - Generalized Fused Lasso
    - LogisticRegression
      - Unregularized
        - [x] Proximal Gradient
          - [x] Tested
    - `tensor_regression/`
      - MatrixRegression
        - Generalized inner product
        - 
  - `pca/`
    Algorithms that find subspaces, and can transform data into low-dimensional representatins.
    - [ ] `base.py` <span style="color:gray;">% PCABaseClass </span>
      <!--
      - Methods:
        - `.__init__()`:
            initialize the input data with correct dtype and move to desired device.
        - `.__call__()`:
            fit with given hyper-parameters
        - `.transform()`: Transform data to new coordinates
        - `.project()`: Project given data onto the subspace spanned by principal directions
        - `.score()`: Score the fit of the model using Bayesian Information Criterion or a similar statistical measure.
        - `.fit()`: Search for the best hyper-parameters using the score
        - `.center_data()`:
      -->
    - [ ] `pca.py` <span style="color:gray;">% PCA </span>
    - [ ] `sparse_pca.py` <span style="color:gray;">% SparsePCA </span>
    - [ ] `spc.py` <span style="color:gray;">% SPC (Witten et. al.) </span>
    - [ ] `sfpca.py` <span style="color:gray;">% SFPCA (Allen et. al.) </span>
      - [ ] Tested
      - [ ] Fit score
        - [ ] Factor sub-problem fit score
        - [ ] Rank fit score
  - `robust_pca/`
    Algorithms that decompose a tensor into Low-rank + Sparse parts.
    - RobustPCA
    - OnlineRPCA
    - HoRPCA
    - ...
  - `distributions/`
    - [ ] `multi_linear_normal.py`
- `synthetic_data/`
  - [x] `generate_low_tucker_rank.py`
  - `generate_anomaly.py`
  - `generate_sythetic_regression.py`
  - [x] `qmult.py` <span style="color:gray;">% Generate orthogonal matrix from uniform, AKA Haar, distribution </span>
  - 
- `utils/`
### Solvers
  - `solvers/`
    - [ ] `manifold/`
      - [x] `line_searcher.py`
      - [x] `gradient_descent.py`
      - [ ] `radmm.py`
      - [ ] `riemannian_ada.py`
        - [ ] Tested
    - [ ] `genlasso/`
      - [ ] 
    - [x] `proximal_gradient_base.py`
      - [ ] Restructure it to not require inheritence from base class.
      - [ ] Implement accelerated proximal gradient as well.
    - [x] `admm_base_class.py`
      - [ ] Restructure to not require inheritance from base class
### Operators
- `proximal_ops/`
  - [ ] `proj_l1_ball.py`
  - [ ] `prox_lp_lq.py`
    - [ ] `prox_l21()`
    - [ ] `prox_grouped_l2_l1()`
  - `singular_value_thresholding.py`
    - [x] `soft_svt()`
    - [x] `mode_n_soft_svt()`
    - [ ] `hard_svt()`
    - [ ] Implementing statistical fit scores
  - `prox_grouped_l21.py`
- `multilinear_ops/`
  - [x] matricization
  - `matrix_products.py`
    - [ ] Khatri-Rao Product
    - [ ] 
  - `tensor_products.py`
    - `mode_n_product()`
    - `multi_mode_product()`
    - [ ] Implement T-Product
  - `tucker.py`
    - [ ] TuckerTensor
      - [ ] Inner Product
      - [ ] Norm
      - [ ] Mode product
      - [ ] Sparse Tucker Tensor?
    - [x] TuckerOperator
    - [x] SumTuckerOperator
  - `cp.py` <span style="color:gray;">% Multi-linear operations and representations related to CP decomposition </span>
    - [ ] CPTensor
      - [ ] Inner Product
      - [ ] Norm
      - [ ] Mode product
      - [ ] Sparse CP Tensor?
    - [ ] mttkrp <span style="color:gray;">% Matricized tensor times khatri-rao product </span>
    - [ ] expand_kruskal