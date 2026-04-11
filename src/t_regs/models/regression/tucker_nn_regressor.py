from typing import Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from ...multilinear_ops.tensor_products import multi_mode_product as mmp
from ...multilinear_ops.tensor_products import mode_n_product
from ...manifolds import Manifold, ManifoldParameter
from ...manifolds import Steifel, Euclidean, GeneralizedSteifel



class TuckerCovariateTransform:
    r"""Transformation operator for tensor shaped covariates.
    
    The tensor covariate transforms are designed to work in conjunction with
    the TensorRegressor class forward modes. This allows for more efficient
    implementation of a Block Coordinate Descent algorithm and may help with
    fitting large tensor covariate datasets into memory.

    Parameters
    ----------
    transform_type: str
        Transformation to apply to the tensor covariates. Options:
            - `full`: No transformation is applied.
            - `core`:
                Reduce the covariate tensor `X`s size from :math:`(F_1,...,F_N)`
                to :math:`(r_{F_1},...,r_{F_N})`
            - `feature_dir_n`:
                Reduce the covariate tensor `X`s size from :math:`(F_1,...,F_N)`
                to :math:`(r_{F_1},...,r_{F_{n-1}},F_n,r_{F_{n+1}},r_{F_N})`
            - `feature_dir_n_full`:
                Reduce the covariate tensor `X`s size from :math:`(F_1,...,F_N)`
                to :math:`(r_{T_1},...,r_{T_M}, F_n, r_{F_n})`
            - `task_dirs`:
                Reduce the covariate tensor `X`s size from :math:`(F_1,...,F_N)`
                to :math:`(r_{T_1},...,r_{T_M})`
    Us: Sequence[torch.Tensor]
        Covariate mode direction matrices.
    core: Optional[torch.Tensor]
        Core tensor.
    mode_n: Optional[int] = None
        Skipped covariate mode for the `feature_dir_n` option.
    """
    __transform_types__ = ['full',
                           'core',
                           'feature_dir_n',
                           'feature_dir_n_full',
                           'task_dirs']
    def __init__(self,
                 transform_type:str,
                 Us: Sequence[torch.Tensor],
                 core: torch.Tensor,
                 mode_n: Optional[int] = None
        ):
        if transform_type in self.__transform_types__:
            self.type = transform_type
        else:
            raise ValueError(
                f"Covariate transform type {transform_type} is not known.")
        self.Us = [ U.detach().to('cpu').pin_memory() for U in Us]
        # for U in self.Us:
        #     U.requires_grad = False
        self.core = core.detach().to('cpu').pin_memory()
        # self.core.requires_grad = False
        self.mode_n = mode_n
        self._N = len(Us)
        self._M = core.ndim - self._N
        self.order = core.ndim
        self.in_prod_modes = [i+1 for i in range(self._N)]
        self._in_prod_modes_b = [i+1 for i in self.in_prod_modes]
        self.skip_modes = [] if mode_n is None else [mode_n]
        self._skip_modes_b = [i+1 for i in self.skip_modes]
        # -1 is to ensure i!= m_plus_n -1
        n = -1 if mode_n is None else mode_n
        m_plus_n = -1 if mode_n is None else mode_n + self._M
        self.C_dot_modes = [
            i for i in range(self._M, self.order) if i != m_plus_n-1
        ]
        self.xt_dot_modes = [i for i in range(self._N) if i != n-1]
        self._xt_dot_modes_b = [i+1 for i in self.xt_dot_modes]

    @torch.no_grad()
    def __call__(self, x):
        """Transform the input covariates and reduce their dimensions."""
        batched = x.ndim != self._N
        in_prod_modes = self._in_prod_modes_b if batched else self.in_prod_modes
        skip_modes = self._skip_modes_b if batched else self.skip_modes
        xt_dot_modes = self._xt_dot_modes_b if batched else self.xt_dot_modes
        if self.type == 'core':
            xt = mmp(x, self.Us, in_prod_modes, transpose=True)
        elif self.type == 'feature_dir_n':
            xt = mmp(x, self.Us, in_prod_modes,
                     skip_modes=skip_modes, transpose=True)
        elif self.type == 'feature_dir_n_full':
            xt = mmp(x, self.Us, in_prod_modes,
                     skip_modes=skip_modes, transpose=True)
            xt = torch.tensordot(xt, self.core,
                    dims=(xt_dot_modes, self.C_dot_modes))
            xt = torch.movedim(xt, source=batched*1, destination=-2)
        elif self.type == 'task_dirs':
            xt = mmp(x, self.Us, in_prod_modes,
                     skip_modes=skip_modes, transpose=True)
            xt = torch.tensordot(xt, self.core, 
                    dims=(xt_dot_modes, self.C_dot_modes))
        elif self.type == 'full':
            return x
        else:
            raise NotImplementedError(
                f"Covariate transform type {self.type} is not implemented yet."
                )
        return xt


class TuckerRegressor(nn.Module):
    r"""A tensor in Tucker decomposition format for tensor of type (M,N).

    Let :math:`\mathcal{X} \in \mathbb{R}^{F_1 \times ...\times F_M} \mathbb{X}`
    be a covariate tensor as a multi-dimensional array. And let :math:`
    \mathcal{Y} \in \mathbb{R}^{T_1 \times ... \times T_N} = \mathbb{Y}` be
    corresponding responses.

    A tucker regressor :math:`\mathcal{B} \in L(\mathbb{X, Y}): \mathbb{X} \to 
    \mathbb{Y}` is a tensor, mapping the tensor covariates to tensor responses.
    Furthermore, it has the following tucker decomposition,

    ..  math::
        \mathcal{B}=\mathcal{C} \times_1\mathbf{V}_1\cdots\times_{F_M} {V}_{M}
            \times_{F_M+1}\mathbf{U}_1\times_{F_M+2}\cdots\times_{F_M+T_N}{U}_{N}

    Where,
    -   :math:`\mathcal{C}\in \mathbb{R}^{r_{T_1}\times\cdots \times r_{T_M}
        \times r_{F_1}\times\cdots \times \times r_{F_N}}` is the core tensor of
        order :math:`M+N`.
    -   :math:`V_m \in \mathrm{St}{T_m \times r_{T_m}}` is the task directions
        for mode :math:`m` for :math:`m = 1,2,...,M`, and :math:`
        \mathrm{St}(T_m, r_{T_m})` is the Steifel manifold with dimension
        :math:`T_m,r_{T_m}`.
    -   :math:`U_n \in \mathrm{St}(F_n \times r_{F_n})` is the feature directions
        for mode :math:`M+n` for :math:`n = 1,2,...,N`, and :math:`
        \mathrm{St}(F_n, r_{F_n})` is the Steifel manifold with dimension
        :math:`F_n,r_{F_n}`.


    Parameters
    ----------
    regression_type: str
        The type of the regression model. The available options are 
        `'linear'`, `'logistic'`, 'multinomial', and 'poisson' regression.
    feature_dims: Sequence[int]
        Dimensions of the feature tensors.
    feature_ranks: Sequence[int]
        Multi-linear rank of the feature regression covariates, :math:`(r_{F_1},
        \cdots, r_{F_N})`.
    task_dims: Optional[Sequence[int]]
        The dimensions of the task for multi-task applications. :math:`(T_1,...,
        T_M). Defaults to univariate regression with value `(1,)`.
    task_ranks: Optional[Sequence[int]]
        Multi-linear rank corresponding to the task dimensions, i.e. :math:`
        (r_{T_1}, ..., r_{T_M})`. Defaults to full-rank, i.e. `task_dims`.
    core: Optional[torch.Tensor],
        Initial values for the core tensor with dimension :math:`(r_{T_1},
        r_{T_2},...,r_{T_M}, r_{F_1},...,r_{F_N}). Defaults to random
        initialization.
    feature_directions: Optional[Sequence[torch.Tensor]]
        Initial values for the tuple of `N` matrices whose columns are the
        directions in the space of the covariates :math:`\mathbb{X}` (features
        within the tensor).
    task_directions: Optional[Sequence[torch.Tensor]]
        Initial values for the tuple of `M` matrices whose columns are the
        directions in the space of the covariates :math:`\mathbb{Y}` (responses
        within the tensor).
    feature_manifolds: Optional[Sequence[Manifold]]
        Manifolds for the feature directions. Defaults to Steifel manifold of
        appropriate dimensions. Can also be `GeneralizedSteifel`.
    task_manifolds: Optional[Sequence[Manifold]]
        Manifolds for the task directions. Defaults to Steifel manifold of
        appropriate dimensions. Can also be `GeneralizedSteifel`.
    
    Notes
    -----
    ..  If `task_dims` and `task_ranks` are equal i.e. full task rank, the task
        directions `Vs` are ignored and the operations are absorbed to the
        appropirate core tensor.
    """
    __forward_modes = ['full',
                       'core',
                       'feature_dir_n',
                       'feature_dir_n_full',
                       'task_dirs']
    regression_types = ['linear', 'logistic', 'multinomial', 'poisson']
    def __init__(self,
        regression_type: str,
        feature_dims: Sequence[int],
        feature_ranks: Sequence[int],
        task_dims: Optional[Sequence[int]] = None,
        task_ranks: Optional[Sequence[int]] = None,
        core: Optional[torch.Tensor] = None,
        feature_directions: Optional[Sequence[torch.Tensor]] = None,
        task_directions: Optional[Sequence[torch.Tensor]] = None,
        feature_manifolds: Optional[Sequence[Manifold]] = None,
        task_manifolds: Optional[Sequence[Manifold]] = None,
        ):
        super().__init__()
        if task_dims is None:
            task_dims = (1,)
        if task_ranks is None:
            task_ranks = task_dims
        self.regression_type = regression_type
        self.feature_dims = tuple(feature_dims)
        self.feature_ranks = tuple(feature_ranks)
        self.task_dims = tuple(task_dims)
        self.task_ranks = tuple(task_ranks)
        self._M = len(self.task_dims)
        self._N = len(self.feature_dims)
        self.dims = self.task_dims + self.feature_dims
        self.ranks = self.task_ranks + self.feature_ranks
        self.order = self._M + self._N
        self._validate_dims()
        self._forward_mode = 'full'
        self._active_dir_n = None
        # First mode    is for samples
        # 2,...,M+1     is for tasks
        # M+2,...,M+N+1 is for input covariates
        self.modes = [i for i in range(1, self.order + 1)]
        self.out_prod_dims = [m for m in range(2, self._M+1)]
        self.in_prod_dims = [n for n in range(2+self._M, 2+self.order)]

        # Initialize the parameters
        self.core = ManifoldParameter(data=core,
                                      manifold=Euclidean(self.ranks))

        if feature_directions is None:
            feature_directions = [None for _ in range(self._N)]
        if feature_manifolds is None:
            feature_manifolds = [
                Steifel(fd, fr)
                for fd, fr in zip(self.feature_dims, self.feature_ranks)
            ]
        self.Us = [
            ManifoldParameter(data=fdir, manifold=fman)
                for fdir,fman in zip(feature_directions, feature_manifolds)
        ]
        for i,p in enumerate(self.Us):
            self.register_parameter(f"U_{i+1}", p)

        self._full_rank_task = self.task_dims == self.task_ranks
        if self._full_rank_task:
            self.Vs = None
        else:
            if task_directions is None:
                task_directions = [None for _ in range(self._M)]
            if task_manifolds is None:
                task_manifolds = [
                    Steifel(td, tr)
                    for td, tr in zip(self.task_dims, self.task_ranks)
                ]
            self.Vs = [
                ManifoldParameter(data=tdir, manifold=tman)
                    for tdir,tman in zip(task_directions, task_manifolds)
                ]
            for i,p in enumerate(self.Vs):
                self.register_parameter(f"V_{i+1}", p)

    @torch.no_grad()
    def predict(self, # pylint: disable=unused-argument
            X:torch.Tensor,
            y=None,
            fw_mode='full',
            mode_n=None,
            ) -> torch.Tensor:
        # TODO: Add docstring
        if fw_mode == 'full':
            eta = self._fw_full(X)
        elif self._forward_mode == 'feature_dir_n':
            eta = self._fw_feature_dir_n(X, mode_n)
        elif self._forward_mode == 'core':
            eta = self._fw_core(X)
        elif self._forward_mode == 'feature_dir_n_full':
            eta = self._fw_feature_dir_n_full(X, mode_n)
        # elif self._forward_mode == 'task_dirs':
            # pass
        else:
            raise ValueError(f"Forward mode {fw_mode} is not recognized")
        return self.inverse_link(eta)

    @torch.no_grad()
    def score(self, pred:torch.Tensor, Y:torch.Tensor) -> float:
        # TODO: Add docstring and perhaps other options for scores.
        if self.regression_type == 'linear':
            ss_total = torch.sum((Y - torch.mean(Y, dim=0))**2)
            ss_residual = torch.sum((Y - pred)**2)
            r2_score = 1.0 - (ss_residual / ss_total)
            return r2_score.item()
        elif self.regression_type == 'logistic':
            pred = pred >0.5
            accuracy = torch.sum(Y == pred).item()
            accuracy = accuracy/ Y.shape[0]
            return accuracy
        elif self.regression_type == 'multinomial':
            pred_idx = pred.argmax(dim=1)
            accuracy = torch.sum(Y == pred_idx).item()
            accuracy = accuracy / Y.shape[0]
            return accuracy
        elif self.regression_type == 'poisson':
            raise NotImplementedError
        else:
            raise ValueError


    def inverse_link(self, eta:torch.Tensor) -> torch.Tensor:
        """Calculate the systematic means (μ) from linear predictors (η)."""
        # TODO: improve the docstring of the inverse link functions.
        if self.regression_type == 'linear' or self.regression_type is None:
            return eta
        elif self.regression_type == 'logistic':
            return F.sigmoid(eta)
        elif self.regression_type == 'multinomial':
            return F.softmax(eta, dim=1 if eta.ndim == self._M else 0)
        elif self.regression_type == 'poisson':
            return torch.exp(eta)
        else:
            raise NotImplementedError

    def loss_fn(self,
                eta: torch.Tensor,
                Y: torch.Tensor,
                weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Loss function corresponding to the Generalized Linear Model"""
        # TODO: improve the docstring of the loss functions.
        if self.regression_type == 'linear':
            return F.mse_loss(eta, Y, weight=weights, reduction='sum')
        elif self.regression_type == 'logistic':
            return F.binary_cross_entropy_with_logits(eta,
                                                      Y,
                                                      weight=weights,
                                                      reduction='sum')
        elif self.regression_type == 'multinomial':
            return F.cross_entropy(eta, Y, weight=weights, reduction='sum')
        elif self.regression_type == 'poisson':
            return  F.poisson_nll_loss(eta, Y,
                                       log_input=True,
                                       reduction='sum')
        else:
            raise NotImplementedError


    def forward(self, x):
        if self._forward_mode == 'full':
            return self._fw_full(x)
        elif self._forward_mode == 'feature_dir_n':
            return self._fw_feature_dir_n(x)
        elif self._forward_mode == 'core':
            return self._fw_core(x)
        elif self._forward_mode == 'feature_dir_n_full':
            return self._fw_feature_dir_n_full(x)
        elif self._forward_mode == 'task_dirs':
            pass

    def _fw_full(self, x):
        b_dim = x.ndim - self._N
        if x.ndim == self._N:
            n_samp = 1
        elif x.ndim == self._N+1:
            n_samp = x.shape[0]
        else:
            raise ValueError("Input covariate order incompatible")
        # Reshape it to have dimensions with [n_samp, 1,...,1, *feature_dims]
        # x_v = x.view([n_samp] + [1]*self._M + list(self.feature_dims))

        # Multiply x_v in the feature modes with U_m^T
        x_v = mmp(x,
                  self.Us,
                  modes=[n+1 for n in range(b_dim, b_dim + self._N)],
                  transpose=True)

        eta = torch.tensordot(x_v, self.core,
                dims=(
                    [n for n in range(b_dim, b_dim+self._N)],
                    [n for n in range(self._M, self.order)]
                )
                )
        # eta = torch.tensordot(eta, self.core,
        #                       dims=([i-1 for i in self.in_prod_dims],
        #                             [i+self._M for i in range(self._N)]))
        if self._full_rank_task:
            return eta
        else:
            # TODO: This line may be wrong.
            return mmp(eta,
                       self.Vs,
                       modes=[m+1 for m in range(b_dim, b_dim+self._M)],
                       transpose=False)

    def _fw_core(self, x):
        b_dim = x.ndim - self._N
        if b_dim == 0:
            n_samp = 1
        elif b_dim == 1:
            n_samp = x.shape[0]
        else:
            raise ValueError("Input covariate order incompatible")
        
        # Reshape it to have dimensions with [n_samp, 1,...,1, *feature_ranks]
        # x_v = x.view([n_samp] + [1]*self._M + list(self.feature_ranks))
        eta = torch.tensordot(x, self.core,
                dims=(
                    [n for n in range(b_dim, b_dim+self._N)],
                    [n for n in range(self._M, self.order)]
                )
                )
        if self._full_rank_task:
            return eta
        else:
            return mmp(eta, self.Vs,
                       modes=[m+1 for m in range(b_dim, b_dim+self._M)],
                       transpose=False)


    def _fw_feature_dir_n(self, x, mode_n=None):
        if x.ndim == self._N:
            n_samp = 1
        elif x.ndim == self._N+1:
            n_samp = x.shape[0]
        else:
            raise ValueError("Input covariate order incompatible")
        # Reshape it to have dimensions with [n_samp, 1,...,1, *feature_ranks]
        n = self._active_dir_n if mode_n is None else mode_n
        fdims = list(self.feature_ranks)
        fdims[n-1] = self.feature_ranks[n-1]
        x_v = x.view([n_samp] + [1]*self._M + fdims)

        eta = mode_n_product(x_v,
                             self.Us[n-1],
                             mode=1+self._M+n,
                             transpose=True)

        eta = torch.tensordot(eta, self.core,
                            dims=([i-1 for i in self.in_prod_dims],
                                  [i+self._M for i in range(self._N)]))
        if self._full_rank_task:
            return eta
        else:
            return mmp(eta, self.Vs, modes=self.out_prod_dims, transpose=False)

    def _fw_feature_dir_n_full(self, x, mode_n=None):
        if x.ndim == self._M+2:
            n_samp = 1
        elif x.ndim == self._M+3:
            n_samp = x.shape[0]
        else:
            raise ValueError("Input covariate order incompatible")
        # Reshape it to have dimensions with [n_samp, *task_ranks, F_n, r_{F_n}]
        n = self._active_dir_n if mode_n is None else mode_n
        tdims = list(self.task_ranks)
        fn, r_fn = self.feature_dims[n-1], self.feature_ranks[n-1]
        x_v = x.view([n_samp] + tdims + [fn, r_fn])
        eta = torch.tensordot(x_v, self.Us[n-1], dims=2)
        if self._full_rank_task:
            return eta
        else:
            return mmp(eta, self.Vs, modes=self.out_prod_dims, transpose=False)

    def _fw_task_dirs(self, x):
        raise NotImplementedError(
            "Forward mode reduction for task dimensions is not implemented yet."
            )


    def set_forward_mode(self,
            forward_mode: str,
            dir_n: Optional[int] = None
            ) -> TuckerCovariateTransform:
        r"""Change the forward mode and return covariate transform to pre-apply.
        
        Parameters
        ----------
        forward_mode: str
            - `'full'`: Accepts inputs covariates in their natural tensor format
                and calculates gradients for all parameters.
            - `'feature_dir_n'`: Forward accepts dimension reduced tensors as
                input covariates. The inputs must have dimensions `(n_samples,
                r_{F_1}`,...,r_{F_{n-1}}, F_{n}, r_{F_{n+1}}, ..., r_{F_N}}.
            - `'feature_dir_n_full'`: Forward accepts dimension reduced tensors as
                input covariates. The inputs must have dimensions `(n_samples,
                r_{T_1},...,r_{T_M}, F_n, r_{F_n})`
            - `'core'`: Forward accepts dimension reduced tensors as
                input covariates. The inputs must have dimensions `(n_samples,
                r_{F_1}`,...,r_{F_N}}.
            - `'task_dirs'`: Forward accepts tensors containing the inner
                products of input covariates :math:`<\mathcal{X}, \mathcal{C}
                \times_{M+1} \mathbf{U}_1 \cdots \times_{M+N} \mathbf{U}_M`.
                The inputs must be of dimension `(n_samples, *task_ranks)`.
            All modes are compatible with :class:`TuckerCovariateTransform`.
        dir_n: Optional[int]
            Covariate mode :math:`n \in [1,...,N]` whose mode will be set active.
        """
        if forward_mode not in self.__forward_modes:
            raise ValueError(
                (f"forward_mode:{forward_mode} is not recognized "
                 f"valid options are among {self.__forward_modes}")
            )
        if ((forward_mode is not self._forward_mode) or 
                (dir_n !=self._active_dir_n)
                ):
            self._forward_mode = forward_mode
            self._active_dir_n = None
            # Disable gradients for all parameter groups and flush the gradients.
            for param in self.parameters(recurse=False):
                param.requires_grad = False
                param.grad = None

            if forward_mode == 'full':
                # Reactivate gradients for all parameter groups
                for param in self.parameters(recurse= False):
                    param.requires_grad = True

            elif forward_mode == 'feature_dir_n':
                if not dir_n in list(range(1, self._N+1)):
                    raise ValueError(
                        f"Provided feature direction n is {dir_n} is invalid."
                        )
                # Reactivate the gradients for U_n
                self.Us[dir_n-1].requires_grad = True
                self._active_dir_n = dir_n

            elif forward_mode == 'feature_dir_n_full':
                if not dir_n in range(1, self._N+1):
                    raise ValueError(
                        f"Provided feature direction n is {dir_n} is invalid."
                        )
                # Reactivate the gradients for U_n
                self.Us[dir_n-1].requires_grad = True
                self._active_dir_n = dir_n

            elif forward_mode == 'core':
                # Reactivate the gradients for the core tensor.
                self.core.requires_grad = True

            elif forward_mode == 'task_dirs':
                self._active_dir_n = None
                if self.Vs is not None:
                    for v in self.Vs:
                        v.requires_grad = True
        return TuckerCovariateTransform(
                transform_type = forward_mode,
                Us = self.Us,
                core = self.core,
                mode_n = self._active_dir_n
            )

    def functional_core(self, core, x):
        # TODO: Add docstring
        b_dim = x.ndim - self._N
        if b_dim == 0:
            n_samp = 1
        elif b_dim == 1:
            n_samp = x.shape[0]
        else:
            raise ValueError("Input covariate order incompatible")
        
        # Reshape it to have dimensions with [n_samp, 1,...,1, *feature_ranks]
        # x_v = x.view([n_samp] + [1]*self._M + list(self.feature_ranks))
        eta = torch.tensordot(x, core,
                dims=(
                    [n for n in range(b_dim, b_dim+self._N)],
                    [n for n in range(self._M, self.order)]
                )
                )
        if self._full_rank_task:
            return eta
        else:
            return mmp(eta, self.Vs,
                       modes=[m+1 for m in range(b_dim, b_dim+self._M)],
                       transpose=False)

    def functional_feature_dir_n_full(self, U, x):
        # TODO: Add docstring
        if x.ndim == self._M+2:
            n_samp = 1
        elif x.ndim == self._M+3:
            n_samp = x.shape[0]
        else:
            raise ValueError("Input covariate order incompatible")
        # Reshape it to have dimensions with [n_samp, *task_ranks, F_n, r_{F_n}]
        # n = self._active_dir_n
        tdims = list(self.task_ranks)
        fn, r_fn = U.shape[0], U.shape[1]
        x_v = x.view([n_samp] + tdims + [fn, r_fn])
        eta = torch.tensordot(x_v, U, dims=2)
        if self._full_rank_task:
            return eta
        else:
            return mmp(eta, self.Vs, modes=self.out_prod_dims, transpose=False)

    @property
    def covariant_degree(self):
        """Covariant degree (order) of the tensor, i.e number of feature modes.
        """
        return self._N

    @property
    def contravariant_degree(self):
        """Covariant degree (order) of the tensor, i.e number of task modes."""
        return self._M

    @property
    def feature_directions(self):
        r"""Matrices of directions for the space of covariates :math:`\mathbb{X}`.
        """
        return self.Us

    @property
    def task_directions(self) -> Union[Sequence[torch.Tensor], None]:
        r"""Matrices of directions for the space of responses :math:`\mathbb{Y}`.
        """# TODO: Consider returning identity tensors for the full rank case.
        return self.Vs


    def _validate_dims(self):
        pass    # TODO: Check dimensions

    def _apply(self, fn, recurse=True):
        # 1. Let the standard PyTorch logic move all Parameters and Buffers
        super()._apply(fn, recurse=recurse)

        # 2. Recursively move tensors inside your custom _manifold attributes
        for param in self.parameters():
            if hasattr(param, '_manifold') and param._manifold is not None:
                # We use the internal 'fn' (which is the move/cast function)
                # to update any tensors inside the manifold object.
                self._move_manifold_tensors(param._manifold, fn)
        return self

    def _move_manifold_tensors(self, obj, fn):
        # If the manifold itself is a Tensor, move it
        if isinstance(obj, torch.Tensor):
            return fn(obj)

        # If it's an object, look for Tensors in its __dict__
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(obj, key, fn(value))
                # Optional: recurse if your manifold has nested objects
                elif hasattr(value, '__dict__'):
                    self._move_manifold_tensors(value, fn)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name, param in self._parameters.items():
            if (isinstance(param, ManifoldParameter) and
                param._manifold is not None
                ):
                # Store the manifold metadata under a unique key
                destination[prefix + name + '._manifold_type'] = type(
                    param._manifold).__name__
                destination[prefix + name + '._manifold_parameters'
                            ] = param._manifold.get_properties()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                            strict, missing_keys, unexpected_keys, error_msgs)
        for name, param in self.named_parameters(recurse=False):
            if isinstance(param, ManifoldParameter):
                type_key = prefix + name + '._manifold_type'
                parameter_key = prefix + name + '._manifold_parameters'

                if type_key in state_dict:
                    manifold_type = state_dict[type_key]
                    manifold_params = state_dict[parameter_key]

                    if manifold_type == 'Euclidean':
                        mantype = Euclidean
                    elif manifold_type == 'Steifel':
                        mantype = Steifel
                    elif manifold_type == 'GeneralizedSteifel':
                        mantype = GeneralizedSteifel
                    else:
                        continue
                    param._manifold = mantype(**manifold_params)
