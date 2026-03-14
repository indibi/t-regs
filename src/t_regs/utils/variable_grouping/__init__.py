"""Module of classes and initializers of variable groupings for regularization.

Author: Mert Indibi
"""
__all__ = [
    'Grouping',
    'LatentGrouping',
    'init_neighborhood_grouping',
    'init_edge_grouping',
    'init_graph_grouping',
]

from .grouping import Grouping, LatentGrouping
from .graph_grouping import init_neighborhood_grouping, init_edge_grouping
from .graph_grouping import init_graph_grouping
