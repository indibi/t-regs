"""Tensor operations and representation related to CANDECOMP/PARAFAC (CP) decomposition"""

class CPTensor:
    """Class representing a tensor in CANDECOMP/PARAFAC (CP) format.

    Attributes:
        factors (list of np.ndarray or torch.Tensor): Factor matrices for each mode.
        weights (np.ndarray or torch.Tensor): Weights for each component.
    """

    def __init__(self, factors, weights=None):
        pass

    def to_tensor(self):
        """Convert CP representation to full tensor.

        Returns:
            np.ndarray or torch.Tensor: Full tensor reconstructed from CP format.
        """
        pass

    