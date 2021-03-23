#!/usr/bin/env python3

"""
    This module reads diffusion data and returns data in proper format for inference
"""

import os

import numpy as np
from fsl.data.featdesign import loadDesignMat
from fsl.data.image import Image
from typing import List

from bench.summary_measures import sample_from_native_space


def group_glm(data, design_mat, design_con):
    """
    Performs group glm on the given data


    :param data: 3d numpy array (n_subj, n_vox, n_dim) 
    :param design_mat: path to design.mat file
    :param design_con: path to design.con file, the first contrast must be first group mean, the second contrast is the 
    change across groups contrast
    :return: data1, delta_data and noise covariance matrices.
    """
    x = loadDesignMat(design_mat)
    c_names, c = loadcontrast(design_con)

    if data.shape[0] == x.shape[0]:
        print(f'running glm for {data.shape[0]} subjects')
    else:
        raise ValueError(f'number of subjects in design matrix is {x.shape[0]} but'
                         f' {data.shape[0]} summary measures were loaded.')

    y = np.transpose(data, [1, 2, 0])
    beta = y @ np.linalg.pinv(x).T
    copes = beta @ c.T

    r = y - beta @ x.T
    sigma_sq = np.array([np.cov(i) for i in r])
    varcopes = sigma_sq[..., np.newaxis] * np.diagonal(c @ np.linalg.inv(x.T @ x) @ c.T)

    data1 = copes[:, :, 0]
    delta_data = copes[:, :, 1]
    variances = varcopes[..., 1]

    return data1, delta_data, variances


def read_glm(glm_dir, mask_add=None):
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
    return data, delta_data, variances


def loadcontrast(design_con):
    """
    Reads design.con file. This function adopted from fslpy.data.loadContrasts with some minor changes


    :param design_con: path to a design.con file generated with fsl glm_gui
    :return: name of contrasts and the contrast vectors.
    """
    names = {}
    with open(design_con, 'rt') as f:
        while True:
            line = f.readline().strip()
            if line.startswith('/ContrastName'):
                tkns = line.split(None, 1)
                num = [c for c in tkns[0] if c.isdigit()]
                num = int(''.join(num))
                if len(tkns) > 1:
                    name = tkns[1].strip()
                    names[num] = name

            elif line.startswith('/NumContrasts'):
                n_contrasts = int(line.split()[1])

            elif line == '/Matrix':
                break

        contrasts = np.loadtxt(f, ndmin=2)

    names = [names[c + 1] for c in range(n_contrasts)]

    return names, contrasts


def voxelwise_group_glm(data, weights, design_con):
    """
    Performs voxel-wise group glm on the given data with weights


    :param data: 3d numpy array (n_subj, n_vox, n_dim)
    :param weights: 2d numpy array (n_subj, n_vox)
    :param design_con: path to design.con file, the first contrast must be first group mean, the second contrast is the
    change across groups contrast
    :return: data1, delta_data and noise covariance matrices.
    """
    c_names, c = loadcontrast(design_con)

    if data.shape[:2] == weights.shape:
        print(f'running glm for {data.shape[0]} subjects and {data.shape[1]}')
    else:
        raise ValueError(f' glm weights and data are not matched')
    if data.ndim == 2:
        data = data[..., np.newaxis]

    n_subj, n_vox, n_dim = data.shape
    copes = np.zeros((n_vox, n_dim, 2))
    varcopes = np.zeros((n_vox, n_dim, n_dim, 2))
    for vox in range(n_vox):
        y = data[:, vox, :].T
        x = np.zeros((n_subj, 2))
        x[:, 0] = weights[:, vox] == 0
        x[:, 1] = weights[:, vox] == 1
        beta = y @ np.linalg.pinv(x).T
        copes[vox] = beta @ c.T

        r = y - beta @ x.T
        sigma_sq = np.cov(r)
        varcopes[vox] = sigma_sq[..., np.newaxis] * np.diagonal(c @ np.linalg.inv(x.T @ x) @ c.T)

    data1 = copes[:, :, 0]
    delta_data = copes[:, :, 1]
    sigma_n = varcopes[..., 1]

    return data1, delta_data, sigma_n


def read_glm_weights(data: List[str], xfm: List[str],  mask: str, save_xfm_path:str):
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

    for subj_idx, (d, x) in enumerate(zip(data, xfm)):
        data, valid_vox = sample_from_native_space(d, x, mask, f"{save_xfm_path}/def_field_{subj_idx}.nii.gz")
        weights[subj_idx, valid_vox] = data
        print(subj_idx, end=' ', flush=True)
    return weights
