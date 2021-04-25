#!/usr/bin/env python3

"""
This module contains functions for fitting spherical harmonics to diffusion data.
"""

import nibabel as nib
import numpy as np
from dipy.reconst.shm import real_sym_sh_basis
from bench import acquisition, image_io
import os


def summary_names(acq, shm_degree, cg=False):
    names = []
    for sh in acq.shells:
        names.append(f"b{sh.bval:1.0f}_mean")
        if sh.lmax > 0:
            for degree in np.arange(2, shm_degree + 1, 2):
                names.append(f"b{sh.bval:1.0f}_l{degree}")
            if cg:
                names.append(f"b{sh.bval:1.0f}_l2_cg")
    return names


def normalized_shms(bvecs, lmax):
    _, phi, theta = cart2spherical(*bvecs.T)
    y, m, l = real_sym_sh_basis(lmax, theta, phi)
    y = y  # /y[0, 0]  # normalization is required to make the first summary measure represent mean signal
    return y, l


def fit_shm(signal, acq, shm_degree):
    """
    Cumputes summary measurements from spherical harmonics fit.
    :param signal: diffusion signal
    :param acq: acquistion protocol
    :param shm_degree: maximum degree for summary measurements
    :return: summary measurements
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    sum_meas = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        bvecs = acq.bvecs[dir_idx]
        shell_signal = signal[..., dir_idx]

        sum_meas.append(shell_signal.mean(axis=-1))

        if this_shell.lmax > 0:
            y, l = normalized_shms(bvecs, this_shell.lmax)
            y_inv = np.linalg.pinv(y.T)
            coeffs = shell_signal @ y_inv
            for degree in np.arange(2, shm_degree + 1, 2):
                sum_meas.append(np.power(coeffs[..., l == degree], 2).mean(axis=-1))

    sum_meas = np.stack(sum_meas, axis=-1)
    return sum_meas


def shm_cov(sum_meas, acq, sph_degree, noise_level):
    if sum_meas.ndim == 1:
        sum_meas = sum_meas[np.newaxis, :]

    variances = np.zeros_like(sum_meas)
    s_idx = 0
    for shell_idx, this_shell in enumerate(acq.shells):
        ng = np.sum(acq.idx_shells == shell_idx)
        variances[:, s_idx] = 1 / ng * (noise_level ** 2)
        s_idx += 1
        if this_shell.lmax > 0:
            y, l = normalized_shms(acq.bvecs[acq.idx_shells == shell_idx], this_shell.lmax)
            c = y[0].T @ y[0]
            for degree in np.arange(2, sph_degree + 1, 2):
                f = (noise_level ** 2) / c
                variances[:, s_idx] = f * (2 * sum_meas[:, s_idx] + f) / (2 * degree + 1)
                s_idx += 1

    sigma_n = np.array([np.diag(v) for v in variances])
    return sigma_n


def shm_jacobian(signal, bvecs, lmax=6, max_degree=4):
    assert lmax == 0 or lmax >= max_degree

    y, l = normalized_shms(bvecs, lmax)
    y_inv = np.linalg.pinv(y.T)

    def derivatives(degree):
        if lmax == 0 and degree > 0:
            return None
        else:
            if degree == 0:
                return y_inv[..., l == 0]
            else:
                return 2 * (signal.dot(y_inv[..., l == degree])).dot((y_inv[..., l == degree]).T) / np.sum(l == degree)

    der = np.array([derivatives(deg) for deg in np.arange(0, max_degree + 1, 2)])
    return der


def cart2spherical(x, y, z):
    """
    Converts to spherical coordinates

    :param x: x-component of the vector
    :param y: y-component of the vector
    :param z: z-component of the vector
    :return: tuple with (r, phi, theta)-coordinates
    """
    vectors = np.array([x, y, z])
    r = np.sqrt(np.sum(vectors ** 2, 0))
    theta = np.arccos(vectors[2] / r)
    phi = np.arctan2(vectors[1], vectors[0])
    if vectors.ndim == 1:
        if r == 0:
            phi = 0
            theta = 0
    else:
        phi[r == 0] = 0
        theta[r == 0] = 0
    return r, phi, theta


def nan_mat(shape):
    a = np.empty(shape)
    if len(shape) > 0:
        a[:] = np.NAN
    else:
        a = np.NAN
    return a


#  image functions:
def fit_summary_single_subject(subj_idx: str, diff_add: str, xfm_add: str, bvec_add: str,
                               bval_add: str, mask_add: str, shm_degree: int, output_add: str):
    """
        the main function that fits summary measurements for a single subject
        :param subj_idx: used to name the summary files.
        :param diff_add: path to diffusion data
        :param xfm_add: path to transformation from native space to standard space.
        :param bvec_add: path to bvec file
        :param bval_add: path to bvec file
        :param mask_add: path to mask
        :param shm_degree: path to the trained model of change file
        :param output_add: path to output directory
        :return:
        """
    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    if os.path.exists(fname):
        print(f'Summary measurements for subject {subj_idx} alredy exists.')
        return 2
    
    print('diffusion data address:' + diff_add)
    print('xfm address:' + xfm_add)
    print('bvec address: ' + bvec_add)
    print('mask address: ' + mask_add)
    print('bval address: ' + bval_add)
    print('shm_degree: ' + str(shm_degree))
    print('output path: ' + output_add)

    def_field = f"{output_add}/def_field_{subj_idx}.nii.gz"
    data, valid_vox = image_io.sample_from_native_space(diff_add, xfm_add, mask_add, def_field)

    acq = acquisition.Acquisition.from_bval_bvec(bval_add, bvec_add)
    summaries = fit_shm(data, acq, shm_degree=shm_degree)

    # write to nifti:
    image_io.write_nifti(summaries, mask_add, fname, np.logical_not(valid_vox))
    if not os.path.exists(f'{output_add}/summary_names.txt'):  # write summary names to a text file in the folder.
        names = summary_names(acq, shm_degree)
        with open(f'{output_add}/summary_names.txt', 'w') as f:
            for t in names:
                f.write("%s\n" % t)
            f.close()
           
    print(f'Summary measurements for subject {subj_idx} computed')
    return 1


def normalize_summaries(summaries, names):
    """
    Normalises summary measures for all subjects. (divide by average attenuation)
    :param names: name of summaries, needed for knowing how to normalize
    :param summaries: array of summaries for all subjects
    :return: normalised summaries
    """
    if not len(names) == summaries.shape[2]:
        raise ValueError(f'Number of summary measurements doesnt match with the trained model.'
                         f'\n Expected {len(names)} measures but got {summaries.shape[2]}.')

    b0_idx = names.index('b0_mean')
    summary_type = [l.split('_')[1] for l in names]
    mean_b0 = np.nanmean(summaries[:, :, b0_idx], axis=0)
    summaries_norm = np.zeros_like(summaries)
    for smm_idx, l in enumerate(summary_type):
        data = summaries[:, :, smm_idx]
        if l == 'mean':
            data = data / mean_b0.T
        else:
            data = data / (mean_b0.T ** 2)
        summaries_norm[:, :, smm_idx] = data

    return summaries_norm


