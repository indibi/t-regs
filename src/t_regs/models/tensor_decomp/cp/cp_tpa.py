from collections import defaultdict

import torch
from torch.nn.functional import softshrink
import numpy as np

# from ....models.matrix_decomp.generalized_svd import generalized_svd as gsvd
from ....multilinear_ops.tensor_products import multi_mode_product
from ....multilinear_ops.cp import expand_kruskal
# from ....multilinear_ops.matricization import matricize, tensorize
from ....proximal_ops.weighted_prox_l1 import icd_weighted_prox_step_l1 as weighted_prox_l1

def weighted_inner_product(u: torch.Tensor,
                           v: torch.Tensor,
                           Q: torch.Tensor | None) -> torch.Tensor:
    """Compute the weighted inner product of vectors u and v with weight matrix Q.
    
    
    Parameters:
    -----------
    u : torch.Tensor
        first input vector
    v : torch.Tensor
        second input vector
    Q : torch.Tensor | None
        weight matrix. If None, standard inner product is computed.
    """
    if Q is None:
        return torch.dot(u, v)
    elif Q.is_sparse:
        Qu = torch.sparse.mm(Q, u.unsqueeze(1)).squeeze(1)
        return torch.dot(Qu, v)
    elif Q.ndim == 2:
        Qu = torch.matmul(Q, u)
        return torch.dot(Qu, v)
    elif Q.ndim == 1:
        return torch.dot(Q*u, v)
    else:
        raise ValueError("Invalid weight matrix Q.")

def weighted_norm(u: torch.Tensor,
                  Q: torch.Tensor) -> torch.Tensor:
    pass

def tensor_weighted_inner_product(X: torch.Tensor,
                                  Y: torch.Tensor,
                                  Qs: list[torch.Tensor]) -> torch.Tensor:
    modes = [mode for mode in range(1, X.ndim+1)]
    skip_modes = [mode for mode in range(1, X.ndim+1) if Qs[mode-1] is None]
    XQ = multi_mode_product(X, Qs, modes=modes, skip_modes=skip_modes)
    return torch.tensordot(XQ, Y, dims=X.ndim)

def tensor_weighted_norm(X: torch.Tensor,
                         Qs: list[torch.Tensor]) -> torch.Tensor:
    pass




class CP_TPA:
    """Regularized CP-Tensor Power Algorithm"""

    def __init__(self,
                    X: torch.Tensor,
                    rank: int = 1,
                    Qs: list[torch.Tensor | None] | None = None,
                    factor_penalties: list[str | None] | None = None,
                    max_iters: int = 100,
                    err_tol: float = 1e-6,
                    verbose: bool | int = False,
                    seed: int = 0,
                    **kwargs):
        self.M = X.ndim
        self.X = X
        self.modes = list(range(1, self.M+1))
        self.dims = X.shape
        self.rank = rank

        if Qs is None:
            self.Qs = [None]*self.M
            self._Q_diags = [None]*self.M
            self._is_Q_diag = [True]*self.M
        else:
            self.Qs = Qs
            if len(self.Qs) != self.M:
                raise ValueError("Length of Qs must match number of modes of X.")
        self._Q_diags = [None]*self.M
        self._is_Q_diag = [False]*self.M
        
        for m, mode in enumerate(self.modes):
            Q = self.Qs[m]
            n = self.dims[m]
            if Q is not None:
                if Q.is_sparse:
                    I = torch.sparse.spdiags(torch.ones(n, device='cpu', dtype=Q.dtype),
                            torch.tensor([0], device='cpu'),
                            (n, n)).to(Q.device)
                    self._Q_diags[m] = (Q * I).coalesce().values()
                    self._is_Q_diag[m] = False
                elif Q.ndim == 2:
                    self._Q_diags[m] = torch.diag(Q)
                    self._is_Q_diag[m] = False
                elif Q.ndim == 1:
                    self._Q_diags[m] = Q
                    self._is_Q_diag[m] = True
                    self.Qs[m] = torch.sparse.spdiags(Q.cpu(),
                            torch.tensor([0], device='cpu'),
                            (n, n)).to(Q.device).to_sparse_csr()
                else:
                    raise ValueError(f"Invalid Q for mode {mode}.")
            else:
                self._Q_diags[m] = None
                self._is_Q_diag[m] = True

        self.factor_penalties = factor_penalties
        if self.factor_penalties is None:
            self.factor_penalties = [None]*self.M
        
        self.var_estimate = None
        self.max_iters = max_iters
        self.err_tol = err_tol
        self.verbose = verbose
        self.seed = seed
        self.kwargs = kwargs
        self.rng = torch.Generator(self.X.device)
        self.rng.manual_seed(self.seed)

        self.Us = [torch.zeros((dim, rank), device=X.device, dtype=X.dtype) for dim in self.dims]
        self.ds = torch.zeros(rank, device=X.device, dtype=X.dtype)


    def _factor_update(self,
                      y: torch.Tensor,
                      lda: float,
                      mode: int,
                      u_init: torch.Tensor | None = None,
                      **kwargs: dict):# -> [torch.Tensor, float, float]:
        """Return unnormalized factor update, its norm and degrees of freedom estimate."""
        # Weighted TPA update (lda=0, Q!=I) :checkmark:
        # Sparse CP update (lda>0, Q=I) :checkmark:
        # Sparse + Weighted CP update (lda>0, Q!=I) :checkmark:
        # Functional TPA update (Q=0, ) # TODO: Functional CP-TPA not implemented yet
        Q = self.Qs[mode-1]
        Q_diag = self._Q_diags[mode-1]
        if self.factor_penalties[mode-1] is None: # No penalty
            u_new = y.clone()
            dof = (u_new.abs()>0.0).sum().item()
        elif self.factor_penalties[mode-1] == 'l1':
            if lda == 0:
                u_new = y.clone()
            else:
                if Q is None:
                    u_new = softshrink(y, lda)
                elif self._is_Q_diag[mode-1]:
                    y_abs = torch.abs(y).squeeze(1)
                    thresholds = lda / Q_diag
                    u_new = torch.sign(y) * torch.clamp(y_abs - thresholds,
                                                        min=0.0).reshape(-1,1)
                else:
                    u_new = weighted_prox_l1(y, lda, Q, Q_diag,
                                             x_init=u_init,
                                             **self.kwargs.get('icd_weighted_l1_args', 
                                                               {})
                                            ).reshape(-1,1)
            dof = (u_new.abs()>0.0).sum().item()
        elif self.factor_penalties[mode-1] == 'generalized_lasso':
            raise NotImplementedError("Generalized lasso penalty not implemented yet.")
        elif self.factor_penalties[mode-1] == 'functional':
            raise NotImplementedError("Functional penalty not implemented yet.")
        else:
            raise ValueError(f"Unknown penalty type {self.factor_penalties[mode-1]}"
                             f"for mode {mode}.")

        u_norm = weighted_inner_product(u_new.squeeze(), u_new.squeeze(), Q).sqrt()
        return u_new, u_norm, dof

    def rank1_tpa(self, Xk: torch.Tensor,
                  ldas: tuple[float] | None = None,
                  u_init: list[torch.Tensor | None] | None = None,
                  **kwargs):
        """Compute rank-1 CP decomposition of tensor Xk using regularized TPA."""
        if ldas is None:
            ldas = tuple(0.0 for _ in range(self.M))
        if u_init is None:
            us = [None]*self.M
        else:
            us = u_init
        
        
        Qus = []
        dofs = [None]*self.M
        u_max = [None]*self.M
        for m, mode in enumerate(self.modes):
            if us[m] is not None:
                u = u_init[m].reshape(-1,1)
            else:
                u = torch.randn((self.dims[m],1),
                            generator=self.rng,
                            device=Xk.device, dtype=Xk.dtype)
                u_norm = weighted_inner_product(u.squeeze(1), u.squeeze(1), self.Qs[m]
                                            ).sqrt()
                us[m] = u / u_norm
            Qus.append(self._Q_u_multiply(us[m], mode))
            
        
        for it in range(self.max_iters):
            us_old = [u.clone() for u in us]

            for m, mode in enumerate(self.modes):
                y = multi_mode_product(Xk, Qus, modes=self.modes,
                                       skip_modes=[mode],
                                       transpose=True).reshape(-1,1)
                
                u_new, u_norm, dof = self._factor_update(y, ldas[m], mode,
                                                         u_init=us[m],
                                                        )
                u_max[m] = torch.max(torch.abs(u_new)).item()
                us[m] = u_new / u_norm if u_norm > 0 else u_new
                Qus[m] = self._Q_u_multiply(us[m], mode)
                dofs[m] = dof
            # Check convergence
            max_change = max((torch.norm(us[m]-us_old[m])
                             for m in range(self.M)))
            if self.verbose:
                print(f"  Iter {it+1}: max factor change = {max_change:.4e}")
            if max_change < self.err_tol:
                break
        d = multi_mode_product(Xk, Qus, modes=self.modes, transpose=True).item()
        return us, d, dofs, u_max

    def factor_subproblem_parameter_search(self, Xk: torch.Tensor,
                                           mode: int,
                                           var_estimate: float | None = None,
                                           ldas: tuple[float] | None = None,
                                           steps:int = 10,
                                           space: str = 'lin',
                                           lda_max: float | None = None,
                                           **kwargs):
        """Select penalty parameter for factor update subproblem using BIC."""
        if ldas is not None:
            ldas = list(ldas)
            ldas[mode-1] = 0.0
        else:
            ldas = [0.0 for _ in range(self.M)]

        us, d, dofs, u_max = self.rank1_tpa(Xk,
                                     ldas=ldas,
                                     u_init=None)
        dof = dofs[mode-1]
        
        rank1_component = expand_kruskal(us, ds=1.0) * d
        residual = Xk - rank1_component
        res_q_norm = tensor_weighted_inner_product(residual,
                                                   residual,
                                                   self.Qs).sqrt().item()
        Xk_norm = tensor_weighted_inner_product(Xk,
                                                Xk,
                                                self.Qs).sqrt().item()
        # rank1_component_norm = tensor_weighted_inner_product(rank1_component,
        #                                            rank1_component,
        #                                            self.Qs).sqrt().item()
        # if Xk_norm < 1e-8:
        #     return us, 0.0, defaultdict(lambda : np.zeros(steps)), 0.0
        if var_estimate is None:
            if self.var_estimate is None:
                var_estimate = (res_q_norm**2) / residual.numel()
                self.var_estimate = var_estimate
            else:
                var_estimate = self.var_estimate

        if lda_max is None:
            if self.Qs[mode-1] is None:
                u_max_scale = 1.25
            else:
                u_max_scale = self.Qs[mode-1].max().item()*1.25
            lda_max = u_max[mode-1]*u_max_scale
        
        if space == 'lin':
            lda_range = torch.linspace(0.0, lda_max,
                                       steps=steps,
                                        device=Xk.device, dtype=Xk.dtype)
        elif space == 'log':
            lda_range = [0.0] + list(np.logspace(-4,
                                                np.log10(lda_max),
                                                num=steps-1))
            lda_range = torch.tensor(lda_range,
                                     device=Xk.device, dtype=Xk.dtype)
        else:
            raise ValueError("space must be 'lin' or 'log'.")
        
        metrics = defaultdict(lambda : np.zeros(steps))
        metrics['lda_range'] = lda_range.cpu().numpy()
        metrics['dof'][0] = dof
        metrics['bic'][0] = (res_q_norm**2) / var_estimate + dof * np.log(residual.numel())
        best_us = us
        best_d = d
        best_lda = 0.0
        best_bic = metrics['bic'][0]
        for i, lda in enumerate(lda_range[1:], start=1):
            ldas[mode-1] = lda.item()
            us, d, dofs, u_max = self.rank1_tpa(Xk,
                                     ldas=ldas,
                                     u_init=None)
            
            dof = dofs[mode-1]
            rank1_component = expand_kruskal(us, ds=1)* d
            residual = Xk - rank1_component
            res_q_norm = tensor_weighted_inner_product(residual,
                                                   residual,
                                                   self.Qs).sqrt().item()
            # var_estimate = (res_q_norm**2) / residual.numel()
            bic = (res_q_norm**2) / (var_estimate) + dof * np.log(residual.numel())
            metrics['dof'][i] = dof
            metrics['bic'][i] = bic
            if bic < best_bic:
                if self.verbose > 1:
                    print(f"  Mode {mode}: lda={lda:.4e}, bic={bic:.4e}, dof={dof}")
                best_bic = bic
                best_lda = lda.item()
                best_us = us
                best_d = d
            metrics['var_estimate'] = var_estimate
        return best_us, best_d, metrics, best_lda

    def parameter_search(self, Xk: torch.Tensor,
                         var_estimate: float | None = None,
                         ldas: tuple[float] | None = None,
                         steps:int = 10,
                         space: str = 'lin',
                         lda_max: list[float | None] | None = None,
                         repeat: int = 1,
                         return_metrics: bool = False,
                         **kwargs):
        """Select penalty parameters for all factor update subproblems using BIC."""
        best_ldas = [0.0 for _ in range(self.M)]
        all_metrics = {}
        if lda_max is None:
            lda_max = [None for _ in range(self.M)]
        for r in range(repeat):
            ldas = [best_ldas[i] for i in range(self.M)]
            for m, mode in enumerate(self.modes):
                if self.factor_penalties[mode-1] is not None:
                    
                    best_us, best_d, metrics, best_lda = self.factor_subproblem_parameter_search(
                                                                Xk,
                                                                mode,
                                                                var_estimate=var_estimate,
                                                                ldas=ldas,
                                                                steps=steps,
                                                                space=space,
                                                                lda_max=lda_max[m],
                                                                **kwargs)
                    best_ldas[mode-1] = best_lda
                    all_metrics[mode] = metrics
                    if self.verbose:
                        print(f"Best lda for mode {mode}: {best_lda:.4e}")
        # if return_metrics:
            return best_ldas, all_metrics, best_us, best_d
        # else:
            # return best_ldas, best_us, best_d


    def _Q_u_multiply(self, u: torch.Tensor,
                      mode: int) -> torch.Tensor:
        """Multiply vector u of shape (n,1) by weight matrix Q for given mode."""
        Q = self.Qs[mode-1]
        u = u.reshape(-1,1)
        if Q is None:
            return u
        elif Q.is_sparse:
            return torch.sparse.mm(Q, u)
        elif Q.ndim == 2:
            return torch.matmul(Q, u)
        elif Q.ndim == 1:
            return (Q * u.squeeze(1)).reshape(-1,1)
        else:
            raise ValueError(f"Invalid weight matrix Q for mode {mode}.")

    
    def __call__(self, X: torch.Tensor,
                    ldas: tuple[float] | None = None,
                    Us_init: list[torch.Tensor | None] | None = None,
                    **kwargs):
        Xk = X.clone()
        
        if Us_init is None:
            Us_init = [None]*self.M
        if ldas is not None:
            for r in range(self.rank):
                u_init = []
                for m in range(self.M):
                    if Us_init[m] is not None:
                        u_init.append(Us_init[m][:,r].reshape(-1,1))
                    else:
                        u_init.append(None)
                us, d, dofs, u_max = self.rank1_tpa(Xk,
                                            ldas=ldas,
                                            u_init=u_init)
                self.ds[r] = d
                for m in range(self.M):
                    self.Us[m][:,r] = us[m].squeeze()
                rank1_component = expand_kruskal(us, ds=1.0) * d
                Xk = Xk - rank1_component
        else:
            for r in range(self.rank):
                
                best_ldas, metrics, best_us, best_d = self.parameter_search(Xk, **kwargs)
                self.ds[r] = best_d
                for m in range(self.M):
                    self.Us[m][:,r] = best_us[m].squeeze()
                rank1_component = expand_kruskal(best_us, ds=1.0) * best_d
                Xk = Xk - rank1_component
        return self.Us, self.ds
    