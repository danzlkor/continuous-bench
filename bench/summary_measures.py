"""
This module contains functions for fitting spherical harmonics to diffusion data.
"""

import glob
import os
from warnings import warn

import nibabel as nib
import numpy as np
import sys
from dipy.reconst.shm import real_sym_sh_basis
from fsl.data.image import Image
from fsl.transform import fnirt
from fsl.wrappers import convertwarp
from scipy.special import lpmv, factorial
from sympy.physics.quantum.cg import CG

from bench.acquisition import Acquisition, ShellParameters


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


def fit_shm(signal, acq, shm_degree, cg=False):
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

            if cg:
                sum_meas.append(cleb_gord_summary_complex(shell_signal, bvecs, this_shell.lmax))

    sum_meas = np.array(sum_meas).T
    return sum_meas


def shm_cov(sum_meas, signal, acq, sph_degree, noise_level, cg=False):
    if sum_meas.ndim == 1:
        sum_meas = sum_meas[np.newaxis, :]
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

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
            if cg:
                y_inv = np.linalg.pinv(y).T
                coeffs = signal[:, acq.idx_shells == shell_idx] @ y_inv
                grad = cleb_gord_grad(coeffs[..., l == 2], y_inv[:, l == 2])
                variances[:, s_idx] = (grad * grad).sum(axis=-1) * (noise_level ** 2)
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
                               bval_add: str, mask_add: str, sph_degree: int, output_add: str):
    """
        the main function that fits summary measurements for a single subject
        :param subj_idx: used to name the summary files.
        :param diff_add: path to diffusion data
        :param xfm_add: path to transformation from native space to standard space.
        :param bvec_add: path to bvec file
        :param bval_add: path to bvec file
        :param mask_add: path to mask
        :param sph_degree: path to the trained model of change file
        :param output_add: path to output directory
        :return:
        """
    print('diffusion data address:' + diff_add)
    print('xfm address:' + xfm_add)
    print('bvec address: ' + bvec_add)
    print('mask address: ' + mask_add)
    print('bval address: ' + bval_add)
    print('shm_degree: ' + str(sph_degree))
    print('output path: ' + output_add)

    def_field = f"{output_add}/def_field_{subj_idx}.nii.gz"
    data, valid_vox = sample_from_native_space(diff_add, xfm_add, mask_add, def_field)
    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    bvals = np.genfromtxt(bval_add)
    idx_shells, shells = ShellParameters.create_shells(bval=bvals)
    acq = Acquisition(shells, idx_shells, bvecs)

    summaries, _ = fit_shm(data, acq, shm_degree=sph_degree)

    # write to nifti:
    mask_img = Image(mask_add)
    mat = np.zeros((*mask_img.shape, len(summaries))) + np.nan
    std_indices = np.array(np.where(mask_img.data > 0)).T
    std_indices_valid = std_indices[valid_vox]

    mat[tuple(std_indices_valid.T)] = np.array(summaries).T

    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    nib.Nifti1Image(mat, mask_img.nibImage.affine).to_filename(fname)
    if subj_idx == '0':
        np.save(f'{output_add}/summary_names', summary_names(acq, sph_degree))

    print(f'Summary measurements for subject {subj_idx} computed')


def read_summary_images(summary_dir: str, mask: str, normalize=True):
    """
    Reads summary measure images
    :param summary_dir: path to the summary measurements
    :param mask: roi mask file name
    :param normalize: normalize the data by group average
    :return: 3d numpy array containing summary measurements, inclusion mask
    """
    mask_img = Image(mask)
    n_subj = len(glob.glob(summary_dir + '/subj_*.nii.gz'))
    if n_subj == 0:
        raise Exception('No summary measures found in the specified directory.')

    summary_list = list()
    for subj_idx in range(n_subj):
        f = f'{summary_dir}/subj_{subj_idx}.nii.gz'
        summary_list.append(Image(f).data[mask_img.data > 0, :])

    print(f'loaded summaries from {n_subj} subjects')
    all_summaries = np.array(summary_list)  # n_subj, n_vox, n_dim
    subj_n_invalids = np.isnan(all_summaries).sum(axis=(1, 2))
    faulty_subjs = np.argwhere(subj_n_invalids > 0.2 * all_summaries.shape[1])
    for s in faulty_subjs:
        warn(f'Subject No. {s + 1} has more than 20% out of scope voxels. You may need to exclude it in the GLM.')

    invalid_voxs = np.isnan(all_summaries).any(axis=(0, 2))

    if invalid_voxs.sum() > 0:
        warn(f'{invalid_voxs.sum()} voxels are excluded since they were '
             f'outside of the image in at least one subject.')
    names = np.load(f'{summary_dir}/summary_names.npy')
    if normalize:
        all_summaries = normalize_summaries(all_summaries, names)

    return all_summaries, invalid_voxs


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

    b0_idx = [i for i, (b, _) in enumerate(names) if b == 0]
    mean_b0 = np.nanmean(summaries[:, :, b0_idx], axis=0)
    summaries_norm = np.zeros_like(summaries)
    for smm_idx, (_, l) in enumerate(names):
        data = summaries[:, :, smm_idx]
        if l == 'l0':
            data = data / mean_b0.T
        else:
            data = data / (mean_b0.T ** 2)
        summaries_norm[:, :, smm_idx] = data

    return summaries_norm


def convert_warp_to_deformation_field(warp_field, std_image, def_field, overwrite=False):
    """
    Converts a warp wield to a deformation field

    This is required because fnirt.Readfnirt only accepts deformation field. If the def_field
     exists it does nothing.
    :param warp_field:
    :param std_image:
    :param def_field:
    :param overwrite:
    :return:
    """
    if not os.path.exists(def_field) or overwrite is True:
        convertwarp(def_field, std_image, warp1=warp_field)
        img = nib.load(def_field)
        img.header['intent_code'] = 2006  # for displacement field style warps
        img.to_filename(def_field)


def transform_indices(native_image, std_mask, def_field):
    """
    Findes nearest neighbour of each voxel of a standard space mask in native space image.
    :param native_image:
    :param std_mask:
    :param def_field:
    :return:
    """
    std_indices = np.array(np.where(std_mask.data > 0)).T
    transform = fnirt.readFnirt(def_field, native_image, std_mask)
    native_indices = np.around(transform.transform(std_indices, 'voxel', 'voxel')).astype(int)

    valid_vox = [np.all([0 < native_indices[j, i] < native_image.shape[i] for i in range(3)])
                 for j in range(native_indices.shape[0])]

    if not np.all(valid_vox):
        warn('Some voxels in mask lie out of native space box.')

    return native_indices, valid_vox


def sample_from_native_space(image, xfm, mask, def_field):
    """
    Sample data from native space using a mask in standard space
    :param image:
    :param xfm:
    :param mask:
    :param def_field:
    :return:
    """
    convert_warp_to_deformation_field(xfm, mask, def_field)
    data_img = Image(image)
    mask_img = Image(mask)
    subj_indices, valid_vox = transform_indices(data_img, mask_img, def_field)
    data_vox = data_img.data[tuple(subj_indices[valid_vox, :].T)].astype(float)
    return data_vox, valid_vox


def from_cmd(args):
    """
        Wrapper function that parses the input from commandline
        :param args: list of strings containing all required parameters for fit_summary_single_subj()
        """
    args.pop(0)  # this tends to be python file name!
    subj_idx_ = args[0]
    diff_add_ = args[1]
    xfm_add_ = args[2]
    bvec_add_ = args[3]
    bval_add_ = args[4]
    mask_add_ = args[5]
    sph_degree = int(args[6])
    output_add_ = args[7]
    fit_summary_single_subject(subj_idx_, diff_add_, xfm_add_, bvec_add_, bval_add_, mask_add_, sph_degree, output_add_)


mv = np.arange(-2, 3)
cleb_gord_idx = np.array([[m1, m2, m3] for m1 in mv for m2 in mv for m3 in mv]).astype(int)
cleb_gord_coef = np.array([CG(2, m[0], 2, m[1], 2, m[2]).doit().evalf() for m in cleb_gord_idx]).astype(float)
cleb_gord_idx = cleb_gord_idx + 2


def cleb_gord_summary_real(shm):
    """
    computes celebch-gordon summary measures for 2nd degree spherical harmonics coefficients
    :param shm: coefficients of spherical harmonics for degree 2
    :return:
    """
    r = np.zeros(shm.shape[:-1])
    for m, c in zip(cleb_gord_idx, cleb_gord_coef):
        r += shm[:, m[0]] * shm[:, m[1]] * shm[:, m[2]] * c
    return r


def cleb_gord_grad(shm, y_inv):
    r = np.zeros((*shm.shape[:-1], y_inv.shape[0]))
    for idx, c in zip(cleb_gord_idx, cleb_gord_coef):
        d1 = (shm[..., idx[1]] * shm[..., idx[2]] * c)[..., np.newaxis] @ y_inv[:, idx[0]].T[np.newaxis, :]
        d2 = (shm[..., idx[0]] * shm[..., idx[2]] * c)[..., np.newaxis] @ y_inv[:, idx[1]].T[np.newaxis, :]
        d3 = (shm[..., idx[0]] * shm[..., idx[1]] * c)[..., np.newaxis] @ y_inv[:, idx[2]].T[np.newaxis, :]
        r += d1 + d2 + d3
    return r


def _legendrep(m, l, x):
    return (-1) ** m * lpmv(m, l, x)


def gaunt_sph_harm(m, l, theta, phi):
    x = np.cos(theta)
    am = np.abs(m)
    val = 1j ** (m + am)
    if l + am == 0:
        f = 1.0
    else:
        f = factorial(l - am) / factorial(l + am)

    val *= np.sqrt((2 * l + 1) / (4 * np.pi) * f).astype(complex)
    val *= _legendrep(am, l, x).astype(complex)
    val *= np.exp(1j * m * phi)
    return val


def complex_sh_basis(lmax, theta, phi):
    y, m, l = [], [], []
    for l_ in np.arange(0, lmax + 1, 2):
        for m_ in range(-l_, l_ + 1):
            y.append(gaunt_sph_harm(m_, l_, theta, phi))
            m.append(m_)
            l.append(l_)
    return np.array(y).T, np.array(m), np.array(l)


def cleb_gord_summary_complex(signal, bvecs, lmax):
    """
    computes celebch-gordon summary measures for 2nd degree spherical harmonics coefficients
    :param shm: coefficients of spherical harmonics for degree 2
    :return: cg-summary measurements
    """
    _, phi, theta = cart2spherical(*bvecs.T)
    y, m, l = complex_sh_basis(lmax, theta, phi)

    y_inv = np.linalg.pinv(y.T)
    coeffs = signal @ y_inv
    shm = coeffs[..., l == 2]
    r = np.zeros(shm.shape[:-1], dtype=shm.dtype)
    for m, c in zip(cleb_gord_idx, cleb_gord_coef):
        r += np.conj(shm[:, m[0]]) * np.conj(shm[:, m[1]]) * shm[:, m[2]] * c
    return np.abs(r)


if __name__ == '__main__':
    from_cmd(sys.argv)
