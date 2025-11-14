# T-Regs 
Library of Tensor Regression algorithms that solve regularized inverse problems in the variational estimator form

$$
\begin{aligned} \mathrm{m}&\mathrm{inimize} &F(\beta, \mathcal{D}) + J(\beta) \\
& \beta \in \mathcal{C}\subset \mathbb{R}^p &
\end{aligned}
$$

where $\beta$ is the vector of model coefficients constrained to a set $\mathcal{C}$, $\mathcal{D}$ is the observed data, $J$ is a regularizer function, promoting a desired structure.


The applications include,
1. Regression problems
    a. Regularized Linear Regression
    b. Generalized Linear Model regressions:
        - Logistic Regression,
        - Sparse Logistic Regression
        - Tensor Generalized Linear Model
2. Matrix/Tensor Recovery problems
    a. Graph signal inpainting,
    b. Collaborative filtering
3. Anomaly Detection
    a. Spatio-temporal anomaly detection
    b. Grouped anomaly detection
4. Dimensionality reduction
    - Functional PCA
    - Sparse PCA