"""Graph based grouping initialization helper functions"""

import numpy as np
import scipy as sp
import torch
import networkx as nx

from .grouping import LatentGrouping

def init_neighborhood_grouping(
    G: nx.Graph | nx.DiGraph,   # pylint: disable=invalid-name
    weighing: str = 'sqrt_group_size',
    r_hop: int = 1,
    nodelist: list | None = None,
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.double,
    ) -> LatentGrouping:
    """Construct variable grouping with graph neighborhood
    
    Parameters
    ----------
    G: nx.Graph | nx.DiGraph,
        Networkx graph :math:`G = (V, E)` with the vertex set :math:`V`
        representing the variables to be grouped according to the edge
        set :math:`E`
    weighing: str = 'sqrt_group_size`,
        Weiging strategy for each group within the grouping. Defaults to the
        square root of the group size.
    r_hop: int = 1
        Group size in `r_hop` radius, i.e. all nodes within `r_hop` distance
        from center vertices are included in the groups making up the grouping.
    nodelist: list
        The list mapping the variable indices to the graph nodes. Defaults to
        the ordering produced by `G.nodes()` consistent with `networkx`.
    device: str | torch.device = 'cuda'.
        Torch device where the grouping tensors reside.
    dtype: torch.dtype = torch.double.
        Data type of the variables.
    
    Returns
    -------
    LatentGrouping
    """
    A = nx.adjacency_matrix(G, nodelist=nodelist)                   # pylint: disable=invalid-name
    I = sp.sparse.diags(np.ones(G.number_of_nodes()), format='csr') # pylint: disable=invalid-name
    if r_hop == 0:
        G_ind = torch.sparse.spdiags(   # pylint: disable=not-callable,invalid-name
            diagonals = torch.ones(G.number_of_nodes(),
                                   device = device, dtype = dtype),
            offsets = torch.tensor([0]),
            shape = (G.number_of_nodes(), G.number_of_nodes()),
            layout = torch.sparse_csr
            )
    elif r_hop == 1:
        A_r = A + I # pylint: disable=invalid-name
        G_ind = torch.sparse_csr_tensor(A_r.indptr, # pylint: disable=invalid-name
                                        A_r.indices,
                                        A_r.data,
                                        device=device,
                                        dtype=dtype).to_sparse_csr()
    else:
        tmp_A = A.copy()    # pylint: disable=invalid-name
        A_r = A.copy() + I  # pylint: disable=invalid-name
        for _ in range(r_hop-1):
            tmp_A = tmp_A @ A   # pylint: disable=invalid-name
            A_r = A_r + tmp_A   # pylint: disable=invalid-name
        A_r = A_r>0             # pylint: disable=invalid-name
        G_ind = torch.sparse_csr_tensor(A_r.indptr,     # pylint: disable=invalid-name
                                        A_r.indices,
                                        A_r.data,
                                        device=device,
                                        dtype=dtype)
    return LatentGrouping(G_ind, weighing=weighing, device=device, dtype=dtype)

def init_edge_grouping(
    G: nx.Graph | nx.DiGraph,   # pylint: disable=invalid-name
    weighing: str = 'sqrt_group_size',
    nodelist: list | None = None,
    edgelist: list | None = None,
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.double,
    ) -> LatentGrouping:
    """Construct variable grouping with graph edge set
    
    Parameters
    ----------
    G: nx.Graph | nx.DiGraph,
        Networkx graph :math:`G = (V, E)` with the vertex set :math:`V`
        representing the variables to be grouped according to the edge
        set :math:`E`
    weighing: str = 'sqrt_group_size`,
        Weiging strategy for each group within the grouping. Defaults to the
        square root of the group size.
    nodelist: list
        The list, mapping the variable indices to the graph nodes. Defaults to
        the ordering produced by `G.nodes()` consistent with `networkx`.
    edgelist: list
        The list, mapping the graph to the graph incidence matrix. Defaults to
        the ordering produced by `G.edges()` consistent with `networkx`.
    device: str | torch.device = 'cuda'.
        Torch device where the grouping tensors reside.
    dtype: torch.dtype = torch.double.
        Data type of the variables.
    
    Returns
    -------
    LatentGrouping
    """
    B = nx.incidence_matrix(G,  # pylint: disable=invalid-name
                            oriented=False,
                            nodelist=nodelist,
                            edgelist=edgelist)
    G_ind = torch.sparse_csr_tensor(B.indptr,   # pylint: disable=invalid-name
                                    B.indices,
                                    B.data,
                                    device=device, dtype=dtype)
    return LatentGrouping(
        G_ind,
        weighing=weighing,
        )

def init_graph_grouping(
    G: nx.Graph | nx.DiGraph,   # pylint: disable=invalid-name
    grouping: str = 'edge',
    weighing: str = 'size_normalized',
    r_hop: int | None = 1,
    nodelist: list | None =None,
    edgelist: list | None =None,
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.float32,
    ) -> LatentGrouping:
    """Wrapper for graph grouping initializers."""
    if grouping == 'edge':
        return init_edge_grouping(
            G,
            weighing,
            nodelist,
            edgelist,
            device,
            dtype,
        )
    if grouping == 'neighbor':
        return init_neighborhood_grouping(
            G,
            weighing,
            r_hop,
            nodelist,
            device,
            dtype
        )
