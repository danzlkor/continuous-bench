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
from acquisition import Acquisition
from scipy.linalg import block_diag


def summary_name(b, l):
    return f"b{b:1.1f}_l{l}"


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


def compute_summary_shell(signal, bvecs, sph_degree=4, lmax=6):
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

    smm = {f'l{degree}': sm(degree) for degree in np.arange(0, sph_degree+1, 2)}

    return smm


def compute_summary(signal, acq: Acquisition, sph_degree=4):
    """
    Computes summary measurements from simulated diffusion MRI signal
    :param signal: array of simulated diffusion signal (..., n_bvecs)
    :param acq: Acquisition object containing acquisition parameters
    :param sph_degree: degree of sh coefficients
    :return: summary_measures (..., n_shells x n_smm - invalids)
    """
    assert signal.shape[-1] == len(acq.idx_shells)

    sum_meas = dict()
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        if this_shell.bval == 0:
            smm = compute_summary_shell(acq.bvecs[dir_idx], signal[..., dir_idx], sph_degree, this_shell.lmax)
        else:
            smm = compute_summary_shell(acq.bvecs[dir_idx], signal[..., dir_idx], sph_degree, this_shell.lmax)
        sum_meas.update({(this_shell.bval, k): v for k, v in smm.items()})
    return sum_meas


def noise_propagation(signal, data, acq, sigma_n=1, sph_degree=4):
    """
    Computes noise covariance matrix of measurements for a given signal
        :param signal: single shell signal
        :param data: (n_samples, dim) array of data
        :param acq: acquisition parameters
        :param sigma_n: noise std
        :return: mu, cov(n_sample, dim, dim)
    """
    variances = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dirs = acq.idx_shells == shell_idx
        bval = this_shell.bval
        if bval == 0:
            variances.append(
                noise_variance(acq.bvecs[dirs], data[(0, 'l0')], 0, sigma_n, this_shell.lmax))

        else:
            for l in np.arange(0, sph_degree + 1, 2):
                name = (bval, f'l{l}')
                variances.append(
                    noise_variance(acq.bvecs[dirs], data[name], l, sigma_n, this_shell.lmax))

    cov = block_diag(*variances)
    return cov


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

    der = {f"l{deg}": compute_smm_derivative(deg) for deg in np.arange(0, max_degree+1, 2)}
    return der


def noise_variance(gradients, smm, l, sigma_n, l_max):
    if l == 0:
        _, phi, theta = cart2spherical(*gradients.T)
        sh_mat, m, l = real_sym_sh_basis(l_max, theta, phi)
        c = np.linalg.pinv(sh_mat).T
        j = c[..., l == 0].mean(axis=-1)
        return j.T.dot(j) * (sigma_n ** 2)
    else:
        ng = gradients.shape[0]
        f = 4 * np.pi * (sigma_n ** 2) / ng
        return smm * 4 * f / (2*l+1) + 2 * (f ** 2) / (2*l + 1)


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
def fit_summary_to_dataset(diffs: list, bvecs: list, bvals: str, xfms: list, roi_mask: str, sph_degree: int, output: str):
    """
    Resamples diffusion data to std space, and fits spherical harmonics to the input images and stores the outputs per
    seubject in a separate image.

    :param diffs: list of filenames for diffusion images.
    :param bvecs: list of filenames for bvecs
    :param bvals: bval filename, this must be the same for all subjects.
    :param xfms: transformation from diffusion space to standard space
    :param roi_mask: region of interest mask filename, must be in standard space
    :param output: path of the output directory
    :param sph_degree: degree for spherical harmonics.
    :return: saves images to the specified path and returns true flag if the process done completely.
    """
    os.makedirs(output, exist_ok=True)
    if len(glob.glob(output + '/subj_*.nii.gz')) < len(diffs):
        py_file_path = os.path.realpath(__file__)
        task_list = list()
        for subj_idx, (x, d, bv) in enumerate(zip(xfms, diffs, bvecs)):
            cmd = f'python3 {py_file_path} {subj_idx} {d} {x} {bv} {bvals} {roi_mask} {sph_degree} {output}'
            task_list.append(cmd)
            # from_cmd(cmd.split()[1:])

        if 'SGE_ROOT' in os.environ.keys():
            print('Submitting jobs to SGE ...')
            with open(f'{output}/tasklist.txt', 'w') as f:
                for t in task_list:
                    f.write("%s\n" % t)
                f.close()

                job_id = run(f'fsl_sub -t {output}/tasklist.txt '
                             f'-q short.q -N bench_summary -l {output}/log -s openmp 2')
                fslsub.hold(job_id)
        else:
            os.system('; '.join(task_list))

        fails = len(diffs) - len(glob.glob(output + '/subj_*.nii.gz'))
        if fails > 0:
            raise RuntimeError(f'Summary measures were not computed for {fails} subjects.')
        else:
            print(f'Summary measures computed for {len(diffs)} subjects.')
    else:
        print('Summary measurements already exist in the specified path')
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
    summaries = list()
    for subj_idx in range(n_subj):
        f = f'{summary_dir}/subj_{subj_idx}.nii.gz'
        summaries.append(Image(f).data[mask_img.data > 0, :])

    print(f'loaded summaries from {n_subj} subjects')
    all_data = np.array(summaries)
    invalid_voxs = np.isnan(summaries).any(axis=(0, 2))
    summaries = all_data[:, invalid_voxs == 0, :]
    if invalid_voxs.sum() > 0:
        warn(f'{invalid_voxs.sum()} voxels are dropped since they were '
             f'outside of the brain mask in some subjects.')

    if normalize:
        summaries = normalize_summaries(summaries)

    return summaries, invalid_voxs


def normalize_summaries(summaries):
    """
    Normalises summary measures for all subjects. (divide by average attenuation)
    :param summaries: array of summaries for all subjects
    :return: normalised summaries
    """
    mdl = 0
    summary_names = mdl.summary_names
    if not len(summary_names) == summaries.shape[2]:
        raise ValueError(f'Number of summary measurements doesnt match with the trained model.'
                         f'\n Expected {len(summary_names)} measures but got {summaries.shape[2]}.')

    b0_idx = [i for i, (b, _) in enumerate(summary_names) if b == 0]
    mean_b0 = summaries[:, :, b0_idx].mean(axis=0)
    summaries_norm = np.zeros_like(summaries)
    for smm_idx, sm in enumerate(summary_names):
        data = summaries[:, :, smm_idx]
        if sm[1] == 'l0':
            data = data / mean_b0.T
        else:
            data = data / (mean_b0.T ** 2)
        summaries_norm[:, :, smm_idx] = data

    return summaries_norm


def transform_indices(xfm, mask, diffusion, def_field_name='def_field_tmp.nii.gz'):
    """
        transforms coordinates in standard space to coordinates in native diffusion space.
        :param xfm: address to transformation from native to standard space
        :param mask: mask image.
        :param diffusion: diffusion image
        :param def_field_name: name of the temporary deformation field.
        :return: transformed indices and list of voxels that lie within diffusion space.
    """
    std_indices = np.array(np.where(mask.data > 0)).T
    convertwarp(def_field_name, mask, warp1=xfm)
    img = nib.load(def_field_name)
    img.header['intent_code'] = 2006  # for displacement field style warps
    img.to_filename(def_field_name)

    transform = fnirt.readFnirt(def_field_name, diffusion, mask)
    os.remove(def_field_name)
    subj_indices = np.around(transform.transform(std_indices, 'voxel', 'voxel')).astype(int)

    valid_vox = [np.all([0 < subj_indices[j, i] < diffusion.shape[i] for i in range(3)])
                 for j in range(subj_indices.shape[0])]

    if not np.all(valid_vox):
        warn('Transformation on mask gave invalid coordinates')

    return subj_indices, valid_vox


def fit_summary_to_image(subj_idx, diff_add, xfm_add, bvec_add, mask_add, sph_degree, output_add):
    """
        the main function that fits summary measurements for a single subject
        :param subj_idx: used to name the summary files.
        :param diff_add: path to diffusion data
        :param xfm_add: path to transformation from native space to standard space.
        :param bvec_add: path to bvec file
        :param mask_add: path to mask
        :param sph_degree: path to the trained model of change file
        :param output_add: path to output directory
        :return:
        """
    print('diffusion data address:' + diff_add)
    print('xfm address:' + xfm_add)
    print('bvec address: ' + bvec_add)
    print('mask address: ' + mask_add)
    print('sph_degree: ' + sph_degree)
    print('output path: ' + output_add)

    diffusion_img = Image(diff_add)
    mask_img = Image(mask_add)
    subj_indices, valid_vox = transform_indices(xfm_add, mask_img, diffusion_img,
                                                f"{output_add}/def_field_{subj_idx}.nii.gz")

    diffusion_vox = diffusion_img.data[tuple(subj_indices[valid_vox, :].T)].astype(float)

    mdl = 0
    acq = mdl.acq

    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    acq.bvecs = bvecs
    summaries = compute_summary(diffusion_vox, acq, sph_degree=sph_degree)

    # write to nifti:
    mat = np.zeros((*mask_img.shape, len(summaries))) + np.nan
    std_indices = np.array(np.where(mask_img.data > 0)).T
    std_indices_valid = std_indices[valid_vox]

    mat[tuple(std_indices_valid.T)] = np.array(list(summaries.values())).T

    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    nib.Nifti1Image(mat, mask_img.nibImage.affine).to_filename(fname)
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
    mask_add_ = args[4]
    mdl_add_ = args[5]
    output_add_ = args[6]
    fit_summary_to_image(subj_idx_, diff_add_, xfm_add_, bvec_add_, mask_add_, mdl_add_, output_add_)


if __name__ == '__main__':
    from_cmd(sys.argv)
