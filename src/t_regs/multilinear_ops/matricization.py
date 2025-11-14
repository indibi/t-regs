"""Matricization module of the multi-dimensional array representation of tensors."""

import numpy as np
import torch

def matricize(X, rows, cols=None):
    """Matricize the tensor X by mapping modes in `rows` to the rows of the matrix.

    
    Args:
        X (torch.Tensor, np.ndarray): Multi-dimensonal array to be matricized.
        rows (list[int]): list of integers indicating the modes mapped to the rows.
            Should be a subset of {1,...,N} where N is the order of the tensor X.
        cols (list[int], optional): List of integers indicating the modes mapped to
            the columns of the resulting matrix. Defaults to the complement of `rows`,
            ordered increasingly.

    Returns:
        torch.Tensor, np.ndarray: Matricized version of X with modes in `rows` mapped
            to the rows and modes in `cols` mapped to the columns.
    
    Examples:
    >>> import numpy as np
    >>> X = np.random.rand(3,4,2)
    >>> X_mat = matricize(X, rows=[1,3])
    >>> X_mat.shape
    (6, 4)
    """
    N = len(X.shape)
    dims = [i for i in range(1,N+1)]
    # Check to see if indicated rows are valid
    if isinstance(rows, tuple) or isinstance(rows, list):
        if not set(rows).issubset(set(dims)):
            raise ValueError("Indicated rows are not valid!")

    # If the columns are not specified, map the dimensions other than rows to columns
    if isinstance(cols, type(None)):
        cols = tuple(sorted(set(dims).difference(set(rows))))
    elif isinstance(cols, tuple) or isinstance(cols, list):
        # If the columns are specified check for their validity
        if not set(cols).isdisjoint(set(rows)):
            raise ValueError("Indicated rows and columns intersect!")
        elif set(cols).union(set(rows)) != set(dims):
            raise ValueError("Rows and columns do not cover all dimensions")
    else:
        raise ValueError("Indicated columns are not valid!")

    d_r = [X.shape[i-1] for i in rows]
    d_c = [X.shape[i-1] for i in cols]
    dims_ = [i-1 for i in dims]
    rows_ = [i-1 for i in rows]
    cols_ = [i-1 for i in cols]
    if isinstance(X, np.ndarray):
        X = np.moveaxis(X, rows_+cols_, dims_)
    elif isinstance(X, torch.Tensor):
        X = torch.moveaxis(X, rows_+cols_, dims_)
    X = X.ravel().reshape((np.prod(d_r, dtype=int),np.prod(d_c, dtype=int)))
    return X




def tensorize(X, og_shape, rows, cols=None):
    """Tensorizes a matrix back to its original tensor form, given the original shape.

    Complements the `matricize` function.
    Args:
        X (np.array, torch.Tensor): Matrix to be tensorized
        og_shape (tuple): Original shape of the tensor before matricized by matricize function
        rows (list[int]): The indices corresponding to rows used by matricize function 
        cols (list[int], optional): Optional. Defaults to None.


    Returns:
        (np.array, torch.Tensor) : Tensorized version of X with shape og_shape
    """
    N = len(og_shape)
    dims = [i for i in range(1,N+1)]
    # Check to see if indicated rows are valid
    if isinstance(rows, tuple) or isinstance(rows, list):
        if not set(rows).issubset(set(dims)):
            raise ValueError("Indicated rows are not valid!")

    # If the columns are not specified, map the dimensions other than rows to columns
    if isinstance(cols, type(None)):
        cols = tuple(set(dims).difference(set(rows)))
    elif isinstance(cols, tuple) or isinstance(cols, list):
        # If the columns are specified check for their validity
        if not set(cols).isdisjoint(set(rows)):
            raise ValueError("Indicated rows and columns intersect!")
        elif set(cols).union(set(rows)) != set(dims):
            raise ValueError("Rows and columns do not cover all dimensions")
    else:
        raise ValueError("Indicated columns are not valid!")
    d_r = [og_shape[i-1] for i in rows]
    d_c = [og_shape[i-1] for i in cols]
    dims_ = [i-1 for i in dims]
    rows_ = [i-1 for i in rows]
    cols_ = [i-1 for i in cols]
    X = X.ravel().reshape(d_r+d_c)
    if isinstance(X, np.ndarray):
        X = np.moveaxis(X, dims_, rows_+cols_)
    elif isinstance(X, torch.Tensor):
        X = torch.moveaxis(X, dims_, rows_+cols_)
    return X

def fold(Xm, dims, m=1):
    """Tensorizes the matrix obtained by t2m to its original state.

    Args:
        Xm (np.ndarray): Matrix
        dims (tuple): original dimensions of the tensor
        m (int): The mode for which the matricization was originally made with t2m. Defaults to 1.
    Returns:
        X (np.ndarray): Tensor
    """
    N = len(dims)
    if m > N or m <1:
        raise ValueError(f"Invalid tensorization order m={m}, N={N}")

    old_dest = (np.arange(N) + (m-1))%N
    dims2 = tuple([dims[i] for i in old_dest])
    X = Xm.ravel().reshape(dims2)
    X = __unroll_from_dim(X, m)
    return X


def unfold(X, m=1):
    """Matricisez the tensor X in the m'th mode.
    
    It is done by stacking fibers of mode m as column vectors.
    Order of the other modes follow cyclic order.
    ie ( I_m x I_(m+1). ... .I_N x I_0. ... I_(m-1) ).
    Args:
        X (np.ndarray): Tensor to be matricized
        m (int, optional): The mode whose fibers are stacked as vectors. Defaults to 1.
    Returns:
        M (np.ndarray): Matricized tensor.
    """
    n = X.shape
    if m>len(n) or m<1:
        raise ValueError(f"Invalid unfolding mode provided. m={m}, X shape:{n}")
    Xm = __roll_2_dim(X,m).ravel().reshape((n[m-1], int(np.prod(n)/n[m-1])))
    return Xm

def __roll_2_dim(X, m):
    n = X.shape
    N = len(n)
    dest = np.arange(N)
    src = (np.arange(N) + (m-1))%N
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))

def unfold_convert_index(n,i,k):
    """Convert the index of an element of a tensor into it's k'th mode unfolding index

    Args:
        n (tuple): Tensor shape
        i (tuple): Tensor index
        k (int): Mode unfolding
    Returns:
        idx (tuple): Index of the element corresponding to the matricized tensor
    """
    if not isinstance(n, tuple):
        raise TypeError("Dimension of the tensor, n is not a tuple")
    if not isinstance(i, tuple):
        raise TypeError("index of the tensor element, i is not a tuple")
    if len(n) ==1:
        raise ValueError(f"The provided dimension n={n} is for the vector case")
    for j, i_ in enumerate(i):
        if i_>= n[j]:
            raise ValueError(f"Index i exceeds the dimension n in {j+1}'th mode")
    if k > len(n) or k<1:
        raise ValueError(f"Unfolding mode {k} is impossible for {len(n)}'th order tensor.")
    j=0
    idx = [i[k-1]]
    n_ = list(n)
    n_ = n_[k:] + n_[:k-1]
    i_ = list(i)
    i_ = i_[k:] + i_[:k-1]
    for p,i_k in enumerate(i_):
        j+= i_k*np.prod(n_[p+1:])
    idx.append(int(j))
    return tuple(idx)

def matricize_convert_index(indices, shape, rows, cols=None):
    """Maps the tensor indices to row and column index of its matricization

    Args:
        indices (tuple): Indices specifying the tensor element
        shape (tuple): Shape of the original tensor
        row (tuple[int]): Indicator list of the dimensions mapped to the rows.
        col (tuple[int]): Indicator list of the dimensions mapped to the columns.

    Returns:
        r,c (tuple): row and column indices of the entry specified by `indices`
    """
    N = len(shape)
    dims = [i for i in range(1,N+1)]
    # Check to see if indicated rows are valid
    if isinstance(rows, tuple) or isinstance(rows, list):
        if not set(rows).issubset(set(dims)):
            raise ValueError("Indicated rows are not valid!")

    # If the columns are not specified, map the dimensions other than rows to columns
    if isinstance(cols, type(None)):
        cols = tuple(sorted(set(dims).difference(set(rows))))
    elif isinstance(cols, tuple) or isinstance(cols, list):
        # If the columns are specified check for their validity
        if not set(cols).isdisjoint(set(rows)):
            raise ValueError("Indicated rows and columns intersect!")
        elif set(cols).union(set(rows)) != set(dims):
            raise ValueError("Rows and columns do not cover all dimensions")
    else:
        raise ValueError("Indicated columns are not valid!")
    M = len(rows)
    dims_r = [dims[i-1] for i in rows]
    dims_c = [dims[i-1] for i in cols]
    i_r = [indices[i-1] for i in rows]
    i_c = [indices[i-1] for i in cols]

    r = 0
    for k in range(M):
        r += i_r[k]*np.prod(dims_r[k+1:], dtype=int)
    c = 0
    for k in range(N-M):
        c += i_c[k]*np.prod(dims_c[k+1:], dtype=int)
    return (r,c)


def __unroll_from_dim(X, m):
    n = X.shape
    N = len(n)
    dest = (np.arange(N) + (m-1))%N
    src = np.arange(N)
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))
