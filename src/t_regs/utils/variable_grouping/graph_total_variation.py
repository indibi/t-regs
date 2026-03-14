"""Graph Total Variation Regularization initialization helper functions.

Author: Mert Indibi (indibimert2@gmail.com)

"""
from typing import Union

import numpy as np
import scipy as sp
import torch
import networkx as nx


def init_gtv_regularization(
    G: nx.Graph,
    edge_group_weighing: str = 'rms_out_degree',
    normalization: str = 'none',
    p: int =2,
    q: int =1,
    nodelist: list | None = None,
    edgelist: list | None = None,
    ):
    """Graph variation regularization operators for the given graph.

    Graph Total Variation
    GTV_{p,q}(x) = [
        sum_{u in V} [sum_{v in N(u)} (w_{u,v}^(1/p) |x_v - x_u|^p)]^(q/p)]
    
    where N(u) is the set of neighbors of u, w_{uv} is the weight of the
    edge (u,v).
    
    Parameters
    ----------
    G: nx.Graph | nx.DiGraph
        Graph for which the variation operator is to be computed. The graph
        can be weighted or unweighted.
    edge_group_weighing: str = 'rms_out_degree'
        Whether to group the outgoing edges of the vertices and separately 
        weighing the ell_2 norm of the local differences.
    normalization: str = 'none'
    p: int=2
        Power of the variation operator. Only 1 and 2 is supported.
    
    Returns
    -------
    Bt:
        csr matrix of size |E| x |V|
    group_indicator:
        Sparse group indicator matrix of size |V| x |E| grouing the edges
        with the same origin together
    group_w:
        group weights of size |V| x 1
    """
    if q != 1:
        raise NotImplementedError(("q!=1 is not yet supported for GTV "
                                "regularization."))
    if not G.is_directed() and p == 2:
        G = G.to_directed()
    if nodelist is None:
        nodelist = list(G.nodes)
    if edgelist is None:
        edgelist = list(G.edges)
    
    if nx.is_weighted(G):
        edge_w = np.array([G[u][v]['weight'] for u,v in edgelist])
    else:
        edge_w = np.ones(len(edgelist))
    B = nx.incidence_matrix(G, oriented=True,
                            nodelist=nodelist, edgelist=edgelist)

    if normalization == 'out_degree':
        edge_w = edge_w/np.array([np.array([G.out_degree(u) for u,v in edgelist])])
    elif normalization == 'in_degree':
        edge_w = edge_w/np.array([np.array([G.in_degree(v) for u,v in edgelist])])
    elif normalization == 'none':
        pass

    # Bt : |E| x |V|
    Bt = B.multiply(np.power(edge_w.reshape(1,-1), 1/p)).tocsc().T
    # group indicator: |V| x |E|
    if p==2:
        group_indicator = ((B<0)*1.0).tocsr()
    elif p==1:
        group_indicator = sp.sparse.eye(B.shape[1], format='csr')
    else:
        raise NotImplementedError("Only p=1 and 2 is implemented for GTV")

    # group_w: |V| x 1
    if edge_group_weighing == 'rms_out_degree':
        edge_w = edge_w.reshape((-1,1))
        group_w = np.power(
            (group_indicator @ np.power(edge_w, p)),
            1/p
            )
    elif (edge_group_weighing == 'none') or edge_group_weighing is None:
        group_w = np.ones((group_indicator.shape[0],1))
    else:
        raise NotImplementedError(("Only `rms_out_degree` or `none` is "
                                "implemented for GTV group weighing."))
    return Bt, group_indicator, group_w

class GraphTotalVariationRegularizer:
    """Graph Total Variation Regularization Helper Class

    Organizes linear operators and the indexing of the Graph Total Variation
    functional used as regularizer in statistical learning and graph signal
    processing. Specifically, represents the following penalty:

    .. math::
        GTV_{p,q}(x) = [
        sum_{u in V} [sum_{v in N(u)} d_u (w_{u,v}^(1/p) |x_v - x_u|^p)]^(q/p)]
    
    where N(u) is the set of neighbors of u, w_{uv} is the weight of the
    edge (u,v).

    Parameters
    ----------
    G: nx.Graph | nx.DiGraph
        Graph for which the variation operator is to be computed. The graph
        can be weighted or unweighted.
    edge_group_weighing: str | None = 'lp_degree'
        Weighing strategy to choose :math:`d_u`. Defaults to `'lp_degree'`
        which sets weights to the :math:`\ell_p` norm of the edge weights
        within group. `None` or `'none'` sets all groups weights to 1.
    edge_weight_normalization: str = 'none'
        Whether to normalize the edge weights of the graph. Options are
        `'none'`, `'out_degree'`, `'in_degre'`.
            - `'out_degree'`: Sum of the edge weights originating from a node
            equal to 1.
            - `'in_degree'`: Sum of the edge weights terminating on the same a
            node equal to 1.
    p: int = 2
        First exponent of the differences. For total variation norm, :math:`p=1`
        corresponds to the anisotropic TV and and :math:`p=2` corresponds to
        the isotropic TV. Currently only 1 or 2 is supported.
    q: int = 1
        Exponent of the grouped differences. Only q=1 is currently supported.
    nodelist: list
        The list mapping the variable indices to the graph nodes. Defaults to
        the ordering produced by `G.nodes()` consistent with `networkx`.
    edgelist: list
        The list, mapping the graph to the graph incidence matrix. Defaults to
        the ordering produced by `G.edges()` consistent with `networkx`.
    device: str | torch.device = 'cuda'.
        Torch device where the grouping tensors reside.
    dtype: torch.dtype = torch.double.
        Data type of the variables.
    
    Attributes
    ----------
    D: torch.Tensor
        Differentiation operator, (Oriented, weighted incidence matrix) of 
        size :math:`|E| \times |V|` in csr format.
    DTD: torch.Tensor
        Matrix :math:`D^T D` of size :math:`|V| \times |V|`, also corresponds to
        the graph laplacian.
    grouping: Grouping
        `Grouping` object representing the grouping of the variables after the
        differentiation operation. If :math:`p\neq 1`, it groups the local
        differences on the :math:`|E|` edges, together according to which
        node the edges originate from. Used in calling the proximal operators.
    p: int
        First exponent of the differences.
    q: int
        Exponent of the grouped differences.
    edge_weight_normalization: str
    nodelist: list
        The list mapping the variable indices to the graph nodes. Defaults to
        the ordering produced by `G.nodes()` consistent with `networkx`.
    edgelist: list
        The list, mapping the graph to the graph incidence matrix. Defaults to
        the ordering produced by `G.edges()` consistent with `networkx`.
    
    """
    # TODO: Add p-laplacian regularizer and other GSP, TV references to the docstring
    def __init__(
    self,
    G: Union[nx.Graph,nx.DiGraph],
    edge_group_weighing: str = 'l2_degree',
    edge_normalization: str = 'none',
    p: int =2,
    q: int =1,
    nodelist: list | None = None,
    edgelist: list | None = None,
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.double,
    ):
        self.device = device
        self.dtype = dtype
        self.p = p
        self.q = q
        self.edge_group_weighing = edge_group_weighing
        self.edge_normalization = edge_normalization
        
        if q != 1:
            raise NotImplementedError(("q!=1 is not yet supported for GTV "
                                        "regularization."))
        if not G.is_directed() and p == 2:
            G = G.to_directed()
        if nodelist is None:
            nodelist = list(G.nodes)
        if edgelist is None:
            edgelist = list(G.edges)
        
        if nx.is_weighted(G):
            edge_w = np.array([G[u][v]['weight'] for u,v in edgelist])
        else:
            edge_w = np.ones(len(edgelist))
        B = nx.incidence_matrix(G, oriented=True,
                                nodelist=nodelist, edgelist=edgelist)

        if edge_normalization == 'out_degree':
            edge_w = edge_w/np.array([np.array([G.out_degree(u) for u,v in edgelist])])
        elif edge_normalization == 'in_degree':
            edge_w = edge_w/np.array([np.array([G.in_degree(v) for u,v in edgelist])])
        elif edge_normalization == 'none' or None:
            pass

        # Bt : |E| x |V|
        Bt = B.multiply(np.power(edge_w.reshape(1,-1), 1/p)).tocsc().T
        # group indicator: |V| x |E|
        if p==2:
            group_indicator = ((B<0)*1.0).tocsr()
        elif p==1:
            group_indicator = sp.sparse.eye(B.shape[1], format='csr')
        else:
            raise NotImplementedError("Only p=1 and 2 is implemented for GTV")
        
        # group_w: |V| x 1
        if edge_group_weighing in 'lp_degree':
            edge_w = edge_w.reshape((-1,1))
            group_w = np.power(
                (group_indicator @ np.power(edge_w, p)),
                1/p
                )
        elif (edge_group_weighing == 'none') or edge_group_weighing is None:
            group_w = np.ones((group_indicator.shape[0],1))
        else:
            raise NotImplementedError(("Only `lp_degree` or `none` is "
                                        "implemented for GTV group weighing."))
        
        BBt = Bt.T @ Bt
        
        self.D = torch.sparse_csr_tensor(
            Bt.indptr,
            Bt.indices,
            Bt.data,
            device=device,
            dtype=dtype)
        self.DTD = torch.sparse_csr_tensor(
            BBt.indptr,
            BBt.indices,
            BBt.data,
            device=device,
            dtype=dtype)
        self.G_ind = torch.sparse_csr_tensor(
            group_indicator.indptr,
            group_indicator.indices,
            group_indicator.data,
            device=device,
            dtype=dtype)
        self.g_weight = torch.tensor(group_w, device='cuda')
    
    def __call__(self, x:torch.Tensor):
        """Evaluate Graph Total Variation of graph signal `x`"""
        raise NotImplementedError("GTV evaluation not yet implemented")
        # TODO: Implement GTV evaluation.