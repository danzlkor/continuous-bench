import numba
import numpy as np
from numpy import linalg


@numba.jit(nopython=True)
def find_t(l1, l2, l3):
    """
    Helper function for hyp_Sapprox

    Args:
        l1: float
            negative first eigenvalue
        l2: float
            negative second eigenvalue
        l3: float
            negative third eigenvalue

    Returns: float
        I guess the return value is t

    """
    a3 = l1 * l2 + l2 * l3 + l1 * l3
    a2 = 1.5 - l1 - l2 - l3
    a1 = a3 - l1 - l2 - l3
    a0 = 0.5 * (a3 - 2 * l1 * l2 * l3)

    inv3 = 1. / 3.
    p = (a1 - a2 * a2 * inv3) * inv3
    q = (-9 * a2 * a1 + 27 * a0 + 2 * a2 * a2 * a2) / 54;
    D = q * q + p * p * p
    offset = a2 * inv3

    if D > 0:
        ee = np.sqrt(D)
        z1 = (-q + ee) ** inv3 + (-q - ee) ** inv3 - offset
        z2 = z1
        z3 = z1
    elif D < 0:
        ee = np.sqrt(-D)
        angle = 2 * inv3 * np.arctan(ee / (np.sqrt(q * q + ee * ee) - q))
        sqrt3 = np.sqrt(3.)
        c = np.cos(angle)
        s = np.sin(angle)
        ee = np.sqrt(-p)
        z1 = 2 * ee * c - offset
        z2 = -ee * (c + sqrt3 * s) - offset
        z3 = -ee * (c - sqrt3 * s) - offset
    else:
        tmp = (-q) ** (inv3)
        z1 = 2 * tmp - offset
        if p != 0 or q != 0:
            z2 = tmp - offset
        else:
            z2 = z1
        z3 = z2
    if z1 < z2 and z1 < z3:
        return z1
    elif z2 < z3:
        return z2
    else:
        return z3


@numba.jit(nopython=True)
def hyp_Sapprox(x):
    """
    Computes 1F1(1/2; 3/2; M) where ``x`` are the eigenvalues from M

    see ``der_hyp_Sapprox`` to only numerically estimate the derivative

    Args:
        x: (3, ) float np.ndarray
            eigenvalues in descending order

    Returns: float
        Result of the hypergeometric function

    """
    if x[0] == 0 and x[1] == 0 and x[2] == 0:
        return 1
    else:
        t = find_t(-x[0], -x[1], -x[2])
        R = 1.
        K2 = 0.
        K3 = 0.
        K4 = 0.

        for idx in range(3):
            R /= np.sqrt(-x[idx] - t)
            K2 += 0.5 * (x[idx] + t) ** -2
            K3 -= (x[idx] + t) ** -3
            K4 += 3 * (x[idx] + t) ** -4

        T = K4 / (8 * K2 * K2) - 5 * K3 * K3 / (24 * K2 ** 3)
        c1 = (np.sqrt(2 / K2) * np.pi * R * np.exp(-t)) * np.exp(T) / (4 * np.pi)
        return c1


@numba.jit(nopython=True)
def der_hyp_Sapprox(x, der_x, dx=1e-6):
    """
    Computes 1F1(1/2; 3/2; M) where ``x`` are the eigenvalues from M and its derivatives

    see ``hyp_Sapprox`` to only calculate the main value

    Args:
        x: (3, ) float np.ndarray
            eigenvalues in descending order
        out: (3, ) float
            Will contain the derivatives
        dx: float
            step size to numerically estimate the derivative

    Returns: float
        Result of the hypergeometric function
        Derivatives will be written to ``out``
    """
    val = hyp_Sapprox(x)
    for i in range(3):
        x[i] += dx
        der_x[i] = (hyp_Sapprox(x) - val) / dx
        x[i] -= dx
    return val


@numba.jit(nopython=True)
def vector_hyp_Sapprox(res, eigenvalues):
    """
    Computes 1F1(1/2; 3/2; M) where ``eigenvalues`` are the eigenvalues from M

    vectorized version of hyp_Sapprox

    Args:
        res: (N, ) float np.ndarray
            Output array with the results from the hypergeometric functions
        eigenvalues: (N, 3) float np.ndarray
            eigenvalues for N matrices in descending order

    """
    for idx in range(len(res)):
        res[idx] = hyp_Sapprox(eigenvalues[idx])


@numba.jit(nopython=True)
def vector_der_hyp_Sapprox(res, der_res, eigenvalues):
    """
    Computes 1F1(1/2; 3/2; M) where ``eigenvalues`` are the eigenvalues from M

    vectorized version of hyp_Sapprox

    Args:
        res: (N, ) float np.ndarray
            Output array with the results from the hypergeometric functions
        der_res: (N, 3) float np.ndarray
            Output array with the results from the derivative of the hypergeometric functions
        eigenvalues: (N, 3) float np.ndarray
            eigenvalues for N matrices in descending order

    """
    for idx in range(len(res)):
        res[idx] = der_hyp_Sapprox(eigenvalues[idx], der_res[idx])


def bingham_normalization(matrices):
    """
    Computes the 1F1(1/2; 3/2; M) for a sequence of matrices

    Args:
        matrices: (N, 3, 3) np.ndarray
            N 3x3 matrices

    Returns: (N, ) np.ndarray
        1F1(1/2; 3/2; M) for the matrices

    """
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        if matrices.shape != (3, 3):
            raise ValueError("Input matrix should be 3x3")
        return bingham_normalization([matrices])[0]
    elif matrices.ndim != 3:
        raise ValueError("Input matrix array should be 2 or 3-dimensional")
    if matrices.shape[1:] != (3, 3):
        raise ValueError("Input matrices should be 3x3")
    eigvals = linalg.eigvalsh(matrices)
    res = np.zeros(matrices.shape[0])
    vector_hyp_Sapprox(res, eigvals)
    return res


def der_bingham_normalization(matrices):
    """
    Computes the 1F1(1/2; 3/2; M) for a sequence of matrices

    Args:
        matrices: (N, 3, 3) np.ndarray
            N 3x3 matrices

    Returns: (N, ) np.ndarray & (N, 3, 3) np.ndarray
        - 1F1(1/2; 3/2; M) for the matrices
        - derivative of 1F1(1/2; 3/2; M) to the matrix elements
    """
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        if matrices.shape != (3, 3):
            raise ValueError("Input matrix should be 3x3")
        return tuple(v[0] for v in der_bingham_normalization([matrices]))
    if matrices.ndim != 3:
        raise ValueError("Input matrix array should be 2 or 3-dimensional")
    if matrices.shape[1:] != (3, 3):
        raise ValueError("Input matrices should be 3x3")
    eigvals, eigvecs = linalg.eigh(matrices)
    res = np.zeros(matrices.shape[0])
    der_eigvals = np.zeros((matrices.shape[0], 3))
    vector_der_hyp_Sapprox(res, der_eigvals, eigvals)
    der_mat = np.sum(der_eigvals[:, None, None, :] * (eigvecs[:, :, None, :] * eigvecs[:, None, :, :]), -1)
    return res, der_mat

