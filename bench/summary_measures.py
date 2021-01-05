"""
This module contains functions to fit spherical harmonics to diffusion data.
"""

from fsl.transform import fnirt
from fsl.data.image import Image
import nibabel as nib
import os
import sys
import glob
from fsl.wrappers import convertwarp
import fsl.utils.fslsub as fslsub
from fsl.utils.run import run
from warnings import warn
import numpy as np
from dipy.reconst.shm import real_sym_sh_basis
from bench.acquisition import Acquisition, ShellParameters
from scipy.linalg import block_diag


def summary_names(acq, sph_degree):
    return [f"b{sh.bval:1.1f}_l{degree}"
            for sh in acq.shells
            for degree in np.arange(0, sph_degree+1 , 2)
            if degree <= sh.lmax]


def fit_shm(bvecs, signal, lmax=6):
    """
    Fit spherical harmonics up to given degree to the data

    :param bvecs: (M, 3) array with gradient orientations
    :param signal: (n_voxels, M) array of diffusion MRI data for the `M` gradients
    :param lmax: maximum degree of the spherical harmonics in the fit
    :return: Tuple with:

        - (..., K) array with spherical harmonic components
        - (K, ) array with the degree of the harmonics
        - (K, ) array with the order of the harmonics
    """
    _, phi, theta = cart2spherical(*bvecs.T)
    y, m, l = real_sym_sh_basis(lmax, theta, phi)
    return signal.dot(np.linalg.pinv(y.T)), m, l


def fit_summary_shell(signal, bvecs, sph_degree=4, lmax=6):
    """
    Creates summary data based on diffusion MRI data

    :param bvecs: (M, 3) array with gradient orientations
    :param signal: (..., M) array of diffusion MRI data for the `M` gradients
    :param lmax: maximum degree of the spherical harmonics in the fit
    :param sph_degree: maximum degree for calculation of measurements
    :return: Dict mapping 'mean' and 'anisotropy' to (..., )-shaped arrays (anisotropy set to None if lmax is 0)
    """
    assert lmax == 0 or lmax >= sph_degree
    coeffs, m, l = fit_shm(signal, bvecs, lmax)

    if lmax == 0:
        sph_degree = 0

    def sm(degree):
        if lmax == 0 and degree > 0:
            return nan_mat(signal.shape[:-1])
        else:
            if degree == 0:
                return coeffs[..., l == 0].mean(axis=-1)
            else:
                return np.power(coeffs[..., l == degree], 2).mean(axis=-1)

    smm = [sm(degree) for degree in np.arange(0, sph_degree + 1, 2)]

    return smm


def compute_summary(signal, acq: Acquisition, sph_degree):
    """
    Computes summary measurements from simulated diffusion MRI signal
    :param signal: array of simulated diffusion signal (..., n_bvecs)
    :param acq: Acquisition object containing acquisition parameters
    :param sph_degree: degree of sh coefficients
    :return: summary_measures (..., n_shells x n_smm - invalids)
    """
    assert signal.shape[-1] == len(acq.idx_shells)

    summaries = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        shell_summaries = fit_summary_shell(acq.bvecs[dir_idx], signal[..., dir_idx], sph_degree, this_shell.lmax)
        summaries += shell_summaries

    return summaries


def noise_propagation(summaries, acq, sph_degree, noise_level):
    """
    Computes noise covariance matrix of measurements for a given signal
        :param summaries: (n_samples, dim) array of data
        :param acq: acquisition parameters
        :param noise_level: noise std
        :param sph_degree: degree for spherical harmonics
        :return: mu, cov(n_sample, dim, dim)
    """
    variances = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dirs = acq.idx_shells == shell_idx
        bval = this_shell.bval
        if bval == 0:
            variances.append(
                noise_variance(acq.bvecs[dirs], summaries[..., 0], 0, noise_level, this_shell.lmax))

        else:
            for l in np.arange(0, sph_degree + 1, 2):
                idx = (shell_idx - 1) * (sph_degree // 2 + 1) + l // 2 + 1
                variances.append(noise_variance(acq.bvecs[dirs], summaries[idx], l, noise_level, this_shell.lmax))

    cov = block_diag(*variances)
    return cov


def noise_variance(gradients, smm, l, sigma_n, l_max):
    """
    :param gradients:
    :param smm:
    :param l:
    :param sigma_n:
    :param l_max:
    :return:
    """
    if l == 0:
        _, phi, theta = cart2spherical(*gradients.T)
        sh_mat, m, l = real_sym_sh_basis(l_max, theta, phi)
        c = np.linalg.pinv(sh_mat).T
        j = c[..., l == 0].mean(axis=-1)
        return j.T.dot(j) * (sigma_n ** 2)
    else:
        ng = gradients.shape[0]
        f = 4 * np.pi * (sigma_n ** 2) / ng
        return smm * 4 * f / (2 * l + 1) + 2 * (f ** 2) / (2 * l + 1)


def compute_summary_jacobian(signal, gradients, lmax=6, max_degree=4):
    assert lmax == 0 or lmax >= max_degree
    _, phi, theta = cart2spherical(*gradients.T)
    y, m, l = real_sym_sh_basis(lmax, theta, phi)
    yp = np.linalg.pinv(y.T)

    def compute_smm_derivative(degree):
        if lmax == 0 and degree > 0:
            return None
        else:
            if degree == 0:
                return yp[..., l == 0]
            else:
                return 2 * (signal.dot(yp[..., l == degree])).dot((yp[..., l == degree]).T) / np.sum(l == degree)

    der = {f"l{deg}": compute_smm_derivative(deg) for deg in np.arange(0, max_degree + 1, 2)}
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

def fit_summary_to_dataset(data: list, bvecs: list, bvals: str, xfms: list, roi_mask: str, sph_degree: int,
                           output: str):
    """
    Resamples diffusion data to std space, fits spherical harmonics to the input images and stores the outputs per
    seubject in a separate image.

    :param data: list of filenames for diffusion images.
    :param bvecs: list of filenames for bvecs
    :param bvals: bval filename, this must be the same for all subjects.
    :param xfms: transformation from diffusion space to standard space
    :param roi_mask: region of interest mask filename, must be in standard space
    :param output: path of the output directory
    :param sph_degree: degree for spherical harmonics.
    :return: saves images to the specified path and returns true flag if the process done completely.
    """
    os.makedirs(output, exist_ok=True)
    if len(glob.glob(output + '/subj_*.nii.gz')) < len(data):
        py_file_path = os.path.realpath(__file__)
        task_list = list()
        for subj_idx, (x, d, bv) in enumerate(zip(xfms, data, bvecs)):
            cmd = f'python3 {py_file_path} {subj_idx} {d} {x} {bv} {bvals} {roi_mask} {sph_degree} {output}'
            task_list.append(cmd)
            # from_cmd(cmd.split()[1:])

        if 'SGE_ROOT' in os.environ.keys():
            with open(f'{output}/tasklist.txt', 'w') as f:
                for t in task_list:
                    f.write("%s\n" % t)
                f.close()

                job_id = run(f'fsl_sub -t {output}/tasklist.txt '
                             f'-q short.q -N bench_summary -l {output}/log -s openmp,2')
                print(f'Jobs were submitted to SGE. waiting ...')
                fslsub.hold(job_id)
        else:
            os.system('; '.join(task_list))

        fails = len(data) - len(glob.glob(output + '/subj_*.nii.gz'))
        if fails > 0:
            raise RuntimeError(f'Summary measures were not computed for {fails} subjects.')
        else:
            print(f'Summary measures computed for {len(data)} subjects.')
    else:
        print('Summary measurements already exist in the specified path.'
              'If you want to re-compute them, delete the current files.')
        fails = 0

    return fails == 0


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
    summary_names = np.load(f'{summary_dir}/summary_names.npy')
    if normalize:
        all_summaries = normalize_summaries(all_summaries, summary_names)

    return all_summaries, invalid_voxs


def normalize_summaries(summaries, summary_names):
    """
    Normalises summary measures for all subjects. (divide by average attenuation)
    :param summary_names: name of summaries, needed for knowing how to normalize
    :param summaries: array of summaries for all subjects
    :return: normalised summaries
    """
    if not len(summary_names) == summaries.shape[2]:
        raise ValueError(f'Number of summary measurements doesnt match with the trained model.'
                         f'\n Expected {len(summary_names)} measures but got {summaries.shape[2]}.')

    b0_idx = [i for i, (b, _) in enumerate(summary_names) if b == 0]
    mean_b0 = np.nanmean(summaries[:, :, b0_idx], axis=0)
    summaries_norm = np.zeros_like(summaries)
    for smm_idx, (_, l) in enumerate(summary_names):
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
    print('sph_degree: ' + str(sph_degree))
    print('output path: ' + output_add)

    def_field = f"{output_add}/def_field_{subj_idx}.nii.gz"
    data, valid_vox = sample_from_native_space(diff_add, xfm_add, mask_add, def_field)
    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    bvals = np.genfromtxt(bval_add)
    idx_shells, shells = ShellParameters.create_shells(bval=bvals)
    acq = Acquisition(shells, idx_shells, bvecs)

    summaries = compute_summary(data, acq, sph_degree=sph_degree)

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


if __name__ == '__main__':
    from_cmd(sys.argv)
