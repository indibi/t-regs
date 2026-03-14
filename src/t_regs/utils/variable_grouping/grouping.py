"""Classes organizing the variable groupings for grouped sparsity reguarizers"""
# TODO: Adding nodelist, edgelist support to map variable indices to variable
# names
# TODO: Adding methods to map representations (v_1,...,v_{N_G}) <-> v_tilde
# efficiently

import torch

class Grouping:
    r"""Class for Variable Groupings.
    
    Used in grouped variable selection and structured sparsity regularized models.
    Parameters
    ----------
    G_ind:
        Group membership indicator matrix with size `num_groups` x 
        `num_vars`.
    weights:
        Vector of weights for each group. If not provided, weights are set with
        `weighing`. Must be of length `num_groups`.
    weighing: str = `sqrt_group_size`
        Group weight selection strategy. Defaults to `sqrt_group_size`
        which sets the weights to the square root of the group sizes.
    device: str | torch.device = 'cuda'.
        Torch device where the tensors reside.
    dtype: torch.dtype = torch.double.
        Data type of the variables.


    Attributes
    ----------
    expander : torch.sparse_coo_tensor
        Hybrid sparse coo tensor with shape (`num_groups` x `num_vars` x 1)
        used in proximal operators and grouped norm calculations.
    weights : torch.Tensor
        Tensor of weights for each group.
    w : torch.Tensor
        Tensor of weights of each group with shape = (`num_groups` x 1 x 1).
    D_G : torch.Tensor
        Number of total group membership for each variable. Dense tensor 
        with shape (1 x `num_vars` x 1).
    num_groups : int
        Number of groups.
    num_vars : int
        Number of variables.
    num_latent_vars : int
        Number of latent variables.
    group_sizes : torch.Tensor
        Number of variables in each group.
    is_overlapping : bool
        `True` if any of two of the groups are overlapping.
    is_covering : bool
        `False` if any of the variables does not belong to a group.
    E : torch.sparse_coo_tensor
        Alias for `expander`
    nov : int
        Alias for `num_vars`.
    nog : int
        Alias for `num_groups`.
    nolv : int
        Alias for `num_latent_vars`.


    Notes
    -----
    G_ind can is equivalent to the transpose of the incidence matrix of the
    hypergraph :math:`\mathcal{H} = (V, E)` where :math:`V = {1,..., p}` is
    the set of the indices of `p` variables and :math:`E` is the hyper-edge
    set of size :math:`|E| = N_G` where :math:`N_G` is `num_groups`.
    """
    def __init__(
    self,
    G_ind: torch.Tensor,# | torch.sparse_csr_tensor,  # pylint: disable=invalid-name
    weights: torch.Tensor | None = None,
    weighing: str = 'sqrt_group_size',
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.double,
    ):
        self.device = device
        self.dtype = dtype
        G_ind = G_ind.to_sparse_coo().to(device=self.device, dtype=self.dtype)

        self.nog = G_ind.shape[0]
        self.nov = G_ind.shape[1]
        self.weighing = weighing
        # Expander: (|Groups| x |Vertices| x 1) Hybrid Sparse COO tensor
        ind = G_ind.indices()
        val = G_ind.to_sparse_coo().values().reshape((-1,1))
        self.E = torch.sparse_coo_tensor(ind,   # pylint: disable=invalid-name
                                        val,
                                        size=(*G_ind.shape,1)
                                        ).coalesce()
        # D_G: (1 x |Vertices| x 1) Dense tensor
        self.D_G = torch.sum(self.E,     # pylint: disable=invalid-name
                             dim=0,
                             keepdim=True).coalesce().to_dense()
        self.group_sizes = torch.sum(self.E, dim=1).coalesce().to_dense()
        if weights is None:
            weights = torch.sum(G_ind, dim=1, keepdims=True).to_dense()
            if self.weighing in ['size_normalized', 'sqrt_group_size']:
                weights = weights.sqrt()
            elif self.weighing == 'size_normalized_inv':
                weights = 1/weights.sqrt()
            elif self.weighing == 'uniform':
                weights = torch.ones_like(weights)
        else:
            weights = weights.to(device=self.device, dtype=self.device)
            if weights.shape[0] != self.nog:
                raise ValueError("The custom weighting tensor must have"+\
                                 " the same length as the number of groups")
        # weights: (|Groups| x 1 x 1) Dense
        self.w = weights.reshape((self.nog,1,1))

    @property
    def expander(self): # pylint: disable=missing-function-docstring
        return self.E

    @property
    def num_vars(self): # pylint: disable=missing-function-docstring
        return self.nov

    @property
    def num_groups(self): # pylint: disable=missing-function-docstring
        return self.nog

    @property
    def weights(self): # pylint: disable=missing-function-docstring
        return self.w.reshape((self.nog,))

    @property
    def is_overlapping(self): # pylint: disable=missing-function-docstring
        return (self.D_G > 1).any()

    @property
    def is_covering(self): # pylint: disable=missing-function-docstring
        return (self.D_G > 0).all()


class LatentGrouping(Grouping):
    r"""Class for Latent Variable Grouping.
    
    Used in grouped variable selection and structured sparsity regularized models.
    Parameters
    ----------
    G_ind:
        Group membership indicator matrix with size `num_groups` x 
        `num_vars`.
    weights: torch.Tensor | None = None
        Vector of weights for each group. If not provided, weights are set with
        `weighing`. Must be of length `num_groups`.
    weighing: str = `sqrt_group_size`
        Group weight selection strategy. Defaults to `sqrt_group_size`
        which sets the weights to the square root of the group sizes.
    device: str | torch.device = 'cuda'.
        Torch device where the tensors reside.
    dtype: torch.dtype = torch.double.
        Data type of the variables.
    
    Attributes
    ----------
    expander : torch.sparse_coo_tensor
        Hybrid sparse coo tensor with shape (`num_groups` x `num_vars` x 1)
        used in proximal operators and grouped norm calculations.
    expander_latent : torch.sparse_coo_tensor
        Hybrid sparse coo tensor with shape (`num_groups`x`num_latent_vars`x 1)
        used in proximal operators and grouped norm calculations.
    H_l : torch.sparse_csr_tensor
        H_l with shape (num_var x num_latent_var) is the transpose of the
        incidence matrix of the hypergraph :math:`\mathcal{H=(V_l, E_l)}` where
        :math:`\mathcal{V}={1,2,..., \tilde{p}}` is the hypergraph vertices.
        Specifically, if :math:`x \in \mathbb{R}^p` is the original variables
        and :math:`(v_1,...,v_{N_G})\in\mathbb{R}^{p \times N_G} are the latent
        variables. If :math:`\tilde{v} \in \mathbb{R}^{\tilde{p}} is the stacked
        version of the latent variables; then :math:`x = H_l \tilde{v} =
        \sum_{g=1}^{N_G} v_g` is satisfied.        
    weights : torch.Tensor
        Tensor of weights for each group.
    w : torch.Tensor
        Tensor of weights of each group with shape = (`num_groups` x 1 x 1).
    D_G : torch.Tensor
        Number of total group membership for each variable. Dense tensor 
        with shape (1 x `num_vars` x 1).
    num_groups : int
        Number of groups.
    num_vars : int
        Number of variables.
    num_latent_vars : int
        Number of latent variables.
    group_sizes : torch.Tensor
        Number of variables in each group.
    is_overlapping : bool
        `True` if any of two of the groups are overlapping.
    is_covering : bool
        `False` if any of the variables does not belong to a group.
    E : torch.sparse_coo_tensor
        Alias for `expander`
    E_l : torch.sparse_coo_tensor
        Alias for `expander_latent`
    nov : int
        Alias for `num_vars`.
    nog : int
        Alias for `num_groups`.
    nolv : int
        Alias for `num_latent_vars`.
    
    
    Notes
    -----
    ..  G_ind can is equivalent to the transpose of the incidence matrix of the
        hypergraph :math:`\mathcal{H} = (V, E)` where :math:`V = {1,..., p}` is
        the set of the indices of `p` variables and :math:`E` is the hyper-edge
        set of size :math:`|E| = N_G` where :math:`N_G` is `num_groups`.
    ..  Every column of the matrix H_l should have exactly one non-zero entry
        equal to 1. :math:\tilde{v}_j is a latent variable belonging to ith
        original variable, if and only if H_{i,j} = 1.
    """
    def __init__(
    self,
    G_ind: torch.Tensor, # | torch.sparse_csr_tensor,                              # pylint: disable=invalid-name
    weights: torch.Tensor | None = None,
    weighing: str = 'sqrt_group_size',
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.double,
    ):
        super().__init__(G_ind, weights, weighing, device, dtype)
        self.nolv = int(torch.sum(self.group_sizes))
        # ---------------------------------------------------------------
        # Constructing matrix H: num_var x num_latent_var
        # Every column of the matrix H should have exactly one non-zero entry
        # equal to 1.
        # If v_j is a latent variable belonging to ith original variable,
        # Then H_{i,j} = 1
        values = torch.ones(self.nolv, device=self.device, dtype=self.dtype)
        row_indices = torch.arange(self.nov,
                                   device=self.device,
                                   dtype=self.dtype).reshape(1,self.nov,1)
        csc = (self.expander*row_indices).to_sparse_csc()
        row_indices = csc.values().ravel().to(dtype=torch.int64)
        ccol_indices = torch.arange(self.nolv+1, device=self.device)
        # H_l is the transpose of the incidence matrix of the hypergraph
        # where {1,2,..., `tilde{p}`} is the hypergraph vertices.
        self.H_l = torch.sparse_csc_tensor(ccol_indices,                        # pylint: disable=invalid-name
                                        row_indices,
                                        values
                                        ).to_sparse_csr()
        # ---------------------------------------------------------------
        # Constructing the derived group indicator matrix `G_ind_l for the
        # stacked latent variables :math:`v \in \mathbb{R}^{\tilde{p}}`.
        # \tilde{v} = [ v1,...,v_{|G_1|},  v_{|G_1|+1},...,v_{|G_1|+|G_2|},   ...,  v_{\sum_{k<g} |G_k|+1},...,v_{\sum_{k<=g} |G_k|},...]^T \in R^{\tilde{p}} # pylint: disable=line-too-long
        #               |<  Group  1  >|   |<         Group  2          >|          |<               Group  g                     >| ...                      # pylint: disable=line-too-long
        # s = H_l @ \tilde{v} = \sum_{g=1}^{N_G} v_g
        gs = self.group_sizes.ravel().to(dtype=torch.int64)
        ccol_indices = torch.cumsum(
            torch.cat(
                [torch.tensor([0], device=self.device, dtype=torch.int64), gs]
                ),
                dim=0
            )
        row_indices = torch.arange(self.nolv, device=self.device)
        G_ind_latent = torch.sparse_csc_tensor(
            ccol_indices,                    # pylint: disable=invalid-name
            row_indices,
            values).to_sparse_coo().coalesce()
        # Expander: (|num_groups| x |num_latent_vars| x 1) Hybrid Sparse COO tensor
        ind = G_ind_latent.indices()
        val = G_ind_latent.values().reshape((-1,1))
        self.E_l = torch.sparse_coo_tensor(ind,                                 # pylint: disable=invalid-name
                                        val,
                                        size=(*G_ind_latent.shape,1)
                                        ).coalesce()
        self.grouping_l = G_ind_latent.transpose(0,1).to_sparse_csr()

    @property
    def expander_latent(self): # pylint: disable=missing-function-docstring
        return self.E_l

    @property
    def num_latent_vars(self): # pylint: disable=missing-function-docstring
        return self.nolv
