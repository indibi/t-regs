import numpy as np
import torch

__all__ = ['matricization',
           'tensor_products',
           'cp',
           'tucker',
           'tensor_products',
           'multi_mode_product',
           'mode_n_product',
           'matricize',
           'tensorize',
           'fold',
           'unfold',
           ]

from .matricization import matricize, tensorize, fold, unfold
from .tensor_products import mode_n_product, multi_mode_product
from .mode_svd import mode_svd
# from .matrix_product
from .tucker import TuckerOperator, SumTuckerOperator
