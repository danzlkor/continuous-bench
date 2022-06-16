#!/usr/bin/env python3
"""
Summary metrics based on diffusion tensor fit
"""
import numpy as np
import torch
from scipy.linalg import block_diag
from bench import acquisition


def summary_names(acq):
    names = list()
    for sh in acq.shells:
        names.append(f'b{sh.bval}_MD')
        if sh.lmax > 0:
            names.append(f'b{sh.bval}_FA')
            names.append(f'b{sh.bval}_Vol')
    return names


def fit_dtm(signal, bval, bvec):
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    sum_meas = list()
    acq = acquisition.Acquisition.from_bval_bvec(bval, bvec)
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        bvecs = acq.bvecs[dir_idx]
        shell_signal = signal[..., dir_idx]
        sm = summary_np(shell_signal, bvecs, this_shell.bval, 1)
        if this_shell.lmax == 0:
            sm = {'MD': sm['MD']}

        sum_meas.extend(list(sm.values()))

    sum_meas = np.array(sum_meas).T
    return sum_meas


def dtm_cov(signal, acq, noise_level):
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    variances = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        bvecs = acq.bvecs[dir_idx]
        shell_signal = signal[..., dir_idx]
        v = noise_variance(shell_signal, bvecs, this_shell.bval, noise_level, 1)
        variances.append(v)

    cov = block_diag(*variances)
    return cov


def summary_np(signal, gradients, bval, s0):
    """
    Creates summary data based on diffusion MRI data

    :param gradients: (M, 3) array with gradient orientations
    :param signal: (..., M) array of diffusion MRI data for the `M` gradients
    :param bval: bvalue
    :param s0:
    :return: Dict mapping 'mean' and 'anisotropy' to (..., )-shaped array
    """
    signal = signal / s0
    if bval == 0:
        return {'MD': np.mean(signal, axis=-1), 'FA': None}

    g = g_mat(gradients)
    d = - np.log(signal) @ np.linalg.pinv(g).T / bval
    dt = np.array([[d[..., 0], d[..., 3], d[..., 4]],
                   [d[..., 3], d[..., 1], d[..., 5]],
                   [d[..., 4], d[..., 5], d[..., 2]]])
    dt = dt.transpose([*np.arange(2, dt.ndim), 0, 1])
    eigs = np.linalg.eigvalsh(dt)

    smm = {'MD': np.mean(eigs, axis=-1),
           'FA': 3 * np.sqrt(1 / 2) * np.std(eigs, axis=-1) / np.linalg.norm(eigs, axis=-1)}
           # 'Vol': volume_summary(eigs)}

    return smm


def g_mat(gradients):
    x, y, z = gradients.T
    return np.array([x ** 2, y ** 2, z ** 2, 2 * x * y, 2 * x * z, 2 * y * z]).T


def summary(signal, gradients, bval, s0):
    """
    Computes  FA and MD from a single shell signal
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients
    :param bval: bvalue
    :return: jacobian vectors.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        return {'MD': np.mean(signal, axis=-1), 'FA': None}

    signal = torch.from_numpy(signal).to(dtype).requires_grad_(False)
    x, y, z = gradients.T
    g = torch.tensor([x ** 2, 2 * x * y, y ** 2, 2 * x * z,  2 * y * z, z ** 2], dtype=dtype).requires_grad_(False)
    d = - torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

    dt = torch.zeros((* signal.shape[:-1], 3, 3), dtype=dtype)
    idx = torch.tril_indices(3, 3)
    dt[..., idx[0], idx[1]] = d

    md = dt.diagonal(dim1=-1, dim2=-2).mean(axis=-1)  # torch doesnt have trace for nd arrays
    r = dt / (md.view((*md.shape, 1, 1)) * 3)  # normalized tensor matrix
    fa = torch.sqrt(0.5 * (3 - 1 / (r @ r).diagonal(dim1=-1, dim2=-2).sum(axis=-1)))
    smm = {'MD': md.detach().numpy(), 'FA': fa.detach().numpy()}

    return smm


def summary_jacobian(signal, gradients, bval, s0):
    """
    Compute jacobian matrix for FA and MD w.r.t signal
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients tuple (..., 3)
    :param bval: bvalue float
    :param s0: signal at b0
    :return: jacobian vectors for md anf fa.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        j_md = None
        j_fa = None
    else:
        signal = torch.from_numpy(signal).to(dtype).requires_grad_(True)
        x, y, z = gradients.T
        g = torch.tensor([x ** 2, 2 * x * y, y ** 2, 2 * x * z, 2 * y * z, z ** 2], dtype=dtype).requires_grad_(False)
        d = - torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

        dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
        idx = torch.tril_indices(3, 3)
        dt[..., idx[0], idx[1]] = d

        md = dt.diagonal(dim1=-1, dim2=-2).mean(axis=-1)  # torch doesnt have trace for nd arrays

        md.backward(retain_graph=True)
        j_md = signal.grad.detach().numpy()
        signal.grad.data.zero_()

        r = dt / (md.view((*md.shape, 1, 1)) * 3)  # normalized tensor matrix
        fa = torch.sqrt(0.5 * (3 - 1 / (r @ r).diagonal(dim1=-1, dim2=-2).sum(axis=-1)))

        fa.backward()
        j_fa = signal.grad.detach().numpy()
    return j_md, j_fa


def noise_variance(signal, gradients, bval, sigma_n, s0):

    j_md = md_jacobian(signal, gradients, bval, s0)
    j_fa = fa_jacobian(signal, gradients, bval, s0)
    if bval > 0:
        j = np.stack([j_md, j_fa])
    else:
        j = j_md
    j = np.squeeze(j)
    m = j @ j.T * (sigma_n ** 2)
    return m


def md_jacobian(signal, gradients, bval, s0):
    """
    Compute jacobian matrix for FA and MD w.r.t signal
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients tuple (..., 3)
    :param bval: bvalue float
    :param s0: signal at b0
    :return: jacobian vectors for md anf fa.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        j_md = np.ones_like(signal) / gradients.shape[0]
    else:
        signal = torch.from_numpy(signal).to(dtype).requires_grad_(True)
        x, y, z = gradients.T
        g = torch.tensor([x ** 2, 2 * x * y, y ** 2, 2 * x * z, 2 * y * z, z ** 2], dtype=dtype).requires_grad_(False)
        d = - torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

        dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
        idx = torch.tril_indices(3, 3)
        dt[..., idx[0], idx[1]] = d

        md = dt.diagonal(dim1=-1, dim2=-2).mean(axis=-1)  # torch doesnt have trace for nd arrays
        md.backward(retain_graph=True)
        j_md = signal.grad.detach().numpy()
    return j_md


def fa_jacobian(signal, gradients, bval, s0):
    """
    Compute jacobian matrix for FA and MD w.r.t signal
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients tuple (..., 3)
    :param bval: bvalue float
    :param s0: signal at b0
    :return: jacobian vectors for md anf fa.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        j_fa = None
    else:
        signal = torch.from_numpy(signal).to(dtype).requires_grad_(True)
        x, y, z = gradients.T
        g = torch.tensor([x ** 2, 2 * x * y, y ** 2, 2 * x * z, 2 * y * z, z ** 2], dtype=dtype).requires_grad_(False)
        d = - torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

        dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
        idx = torch.tril_indices(3, 3)
        dt[..., idx[0], idx[1]] = d

        md = dt.diagonal(dim1=-1, dim2=-2).mean(axis=-1)  # torch doesnt have trace for nd arrays
        r = dt / (md.view((*md.shape, 1, 1)) * 3)  # normalized tensor matrix
        fa = torch.sqrt(0.5 * (3 - 1 / (r @ r).diagonal(dim1=-1, dim2=-2).sum(axis=-1)))

        fa.backward()
        j_fa = signal.grad.detach().numpy()
    return j_fa


def volume_summary(eigs):
    return (32 * np.pi / 945) * (2 * (eigs ** 3).sum(axis=-1) -
                                  3 * (eigs[:, 0] ** 2 * (eigs[:, 1] + eigs[:, 2]) +
                                       eigs[:, 1] ** 2 * (eigs[:, 0] + eigs[:, 2]) +
                                       eigs[:, 2] ** 2 * (eigs[:, 0] + eigs[:, 1])) +
                                  12 * eigs[:, 0] * eigs[:, 1] * eigs[:, 2])
