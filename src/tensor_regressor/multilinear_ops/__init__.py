import numpy as np
import torch

__all__ = ['matricization', 'cp', 'tucker', 'tensor_products']

from .matricization import matricize, tensorize, fold, unfold
from .tensor_products import mode_n_product, multi_mode_product
from .mode_svd import mode_svd
# from .matrix_product
from .tucker import TuckerTensor, TuckerOperator, SumTuckerOperator
