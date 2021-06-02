#!/usr/bin/env python3

"""
This module contains functions for reading and writing to image files
"""

import glob
import os
from warnings import warn

import nibabel as nib
import numpy as np
from fsl.data.image import Image
from fsl.transform import fnirt
from fsl.wrappers import convertwarp
from typing import List
from joblib import Parallel, delayed


def read_summary_images(summary_dir: str, mask: str):
    """
    Reads summary measure images
    :param summary_dir: path to the summary measurements
    :param mask: roi mask file name
    :param normalize: normalize the data by group average
    :return: 3d numpy array containing summary measurements, inclusion mask
    """
    mask_img = Image(mask)
    n_subj = len(glob.glob(summary_dir + '/subj_*.nii.gz'))
    print(summary_dir)
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
    with open(f'{summary_dir}/summary_names.txt', 'r') as reader:
        names = [line.rstrip() for line in reader]

    return all_summaries, invalid_voxs, names


def convert_warp_to_deformation_field(warp_field, std_image, def_field, overwrite=True):
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
    Findes nearest neighbour of each voxel of a standard space mask in a native space image.
    :param native_image: images object of native space
    :param std_mask: image object of a standard space mask
    :param def_field: string
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


def sample_from_native_space(image, xfm, mask, def_field=None):
    """
    Sample data from native space using a mask in standard space
    :param image: address to the image file in native space
    :param xfm: address to the transformation from standard to native
    :param mask: adress to the mask in standard space
    :param def_field: adress to the deformation field file, if not exsits will creat it, otherwise will use it.
    :return: sampled data, and valid voxels
    """
    convert_warp_to_deformation_field(xfm, mask, def_field)
    data_img = Image(image)
    mask_img = Image(mask)
    subj_indices, valid_vox = transform_indices(data_img, mask_img, def_field)
    data_vox = data_img.data[tuple(subj_indices[valid_vox, :].T)].astype(float)
    return data_vox, valid_vox


def write_glm_results(data, delta_data, sigma_n, mask, invalid_vox, glm_dir):
    """
    Writes the results of GLM into files,
    :param data: baseline measurement matrix (n_vox, n_sm)
    :param delta_data:  change matrix (n_vox, n_sm)
    :param sigma_n noise covariance of change (n_vox, n_sm, n_sm)
    :param glm_dir: path to the glm dir, it write data.nii.gz, delta_data.nii.gz, variance.nii.gz,
    and valid_mask.nii.gz in the address.
    :param mask: path to the mask file.
    :param invalid_vox: should be of size of mask voxels, contains 0 for valid voxels and 1 for invalid ones.
    This is used for droping voxels that are not valid.

    """
    os.makedirs(glm_dir, exist_ok=True)
    tril_idx = np.tril_indices(sigma_n.shape[-1])
    covariances = np.stack([s[tril_idx] for s in sigma_n], axis=0)

    os.makedirs(glm_dir, exist_ok=True)
    write_nifti(data, mask, glm_dir + '/data', invalid_vox)
    write_nifti(delta_data, mask, glm_dir + '/delta_data', invalid_vox)
    write_nifti(covariances, mask, glm_dir + '/variances', invalid_vox)

    valid_mask = np.ones((data.shape[0], 1))
    write_nifti(valid_mask, mask, glm_dir + '/valid_mask', invalid_vox)


def read_glm_results(glm_dir, mask_add=None):
    """
    :param glm_dir: path to the glm dir, it must contain data.nii.gz, delta_data.nii.gz, variance.nii.gz,
    and valid_mask.nii.gz
    :param mask_add: address of mask file, by default it uses the mask in glm dir.
    :return:
    """
    if mask_add is None:
        mask_add = glm_dir + '/valid_mask.nii'

    mask_img = Image(mask_add)
    data = Image(f'{glm_dir}/data.nii').data[mask_img.data > 0, :]
    delta_data = Image(f'{glm_dir}/delta_data.nii').data[mask_img.data > 0, :]
    variances = Image(f'{glm_dir}/variances.nii').data[mask_img.data > 0, :]

    n_vox, n_dim = data.shape
    tril_idx = np.tril_indices(n_dim)
    diag_idx = np.diag_indices(n_dim)
    sigma_n = np.zeros((n_vox, n_dim, n_dim))
    for i in range(n_vox):
        sigma_n[i][tril_idx] = variances[i]
        sigma_n[i] = sigma_n[i] + sigma_n[i].T
        sigma_n[i][diag_idx] /= 2

    return data, delta_data, sigma_n


def read_glm_weights(data: List[str], xfm: List[str],  mask: str, save_xfm_path:str, parallel=True):
    """
    reads voxelwise glm weights for each subject in an arbitrary space and a transformation from that space to standard,
    then takes voxels that lie within the mask (that is in standard space).

    :param output: output directory to save intermediate transformation files
    :param data: list of nifti files one per subject
    :param xfm: list of transformations from the native space to standad space
    :param mask: address of roi mask in standard space
    :returns: weights matrix (n_subj , n_vox). For the voxels that lie outside of image boundaries it places nans.

    """

    os.makedirs(save_xfm_path, exist_ok=True)
    mask_img = Image(mask)
    std_indices = np.array(np.where(mask_img.data > 0)).T

    n_vox = std_indices.shape[0]
    n_subj = len(data)
    weights = np.zeros((n_subj, n_vox)) + np.nan
    print('Reading GLM weights:')

    def func(subj_idx):
        subjdata, valid_vox = sample_from_native_space(
            data[subj_idx], xfm[subj_idx], mask, f"{save_xfm_path}/def_field_{subj_idx}.nii.gz")
        print('weights for', subj_idx,'loaded.')
        return subjdata, valid_vox

    if parallel:
        res = Parallel(n_jobs=-1, verbose=True)(delayed(func)(i) for i in range(n_subj))
    else:
        res = []
        for i in range(n_subj):
            res.append(func(i))

    for subj_idx in range(n_subj):
        weights[subj_idx, res[subj_idx][1]] = res[subj_idx][0]

    return weights


def write_nifti(data: np.ndarray, mask_add: str, fname: str, invalids=None):
    """
    writes data to a nifti file.

    :param data: data matrix to be written to the file (M, d)
    :param mask_add: mask address, the mask should have exactly N ones.
    :param fname: full path to the output nifiti file
    :param invalids: invalid voxels (filled with zeros) (N, 1) with M false entries
    :return:
    """

    mask = Image(mask_add)
    std_indices = np.array(np.where(mask.data > 0)).T

    if invalids is None:
        invalids = np.zeros((std_indices.shape[0],), dtype=bool)

    std_indices_valid = std_indices[np.logical_not(invalids)]
    std_indices_invalid = std_indices[invalids]

    img = np.zeros((*mask.shape, data.shape[1]))
    img[tuple(std_indices_valid.T)] = data
    img[tuple(std_indices_invalid.T)] = np.nan

    nib.Nifti1Image(img, mask.nibImage.affine).to_filename(fname)
