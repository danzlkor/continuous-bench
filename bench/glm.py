#!/usr/bin/env python3

"""
    This module reads diffusion data and returns data in proper format for inference
"""

import numpy as np
from fsl.data.featdesign import loadDesignMat
from bench import change_model


def group_glm(data, design_mat, design_con):
    """
    Performs group glm on the given data


    :param data: 3d numpy array (n_subj, n_vox, n_dim) 
    :param design_mat: path to design.mat file
    :param design_con: path to design.con file, the first contrast must be first group mean, the second the group difference
    3rd contrast is the difference between two groups.
    change across groups contrast
    :return: data1, delta_data and noise covariance matrices.
    """
    x = loadDesignMat(design_mat)
    c_names, c = loadcontrast(design_con)
    n_subj, n_vox, n_dim = data.shape

    if data.shape[0] == x.shape[0]:
        print(f'running glm for {data.shape[0]} subjects')
    else:
        raise ValueError(f'number of subjects in design matrix is {x.shape[0]} but'
                         f' {data.shape[0]} summary measures were loaded.')

    y = np.transpose(data, [1, 2, 0])  # make it (n_vox, n_dim, n_subj)
    beta = y @ np.linalg.pinv(x).T
    copes = beta @ c.T

    r = y - beta @ x.T
    sigma_sq = np.array([np.cov(i) for i in r])
    varcopes = sigma_sq[..., np.newaxis] * np.diagonal(c @ np.linalg.inv(x.T @ x) @ c.T)

    data1 = copes[:, :, 0]
    delta_data = copes[:, :, 1]
    sigma_n = varcopes[..., 1]

    if n_subj <= n_dim:
        Warning.warn('fewer samples than features, regularising sigma_n with 0.1 on diagonals')
        sigma_n += 0.1 * np.eye(sigma_n.shape[-1])

    return data1, delta_data, sigma_n


def group_glm_paired(data):
    """
    Performs group glm on the given pared data, assumes data is sorter as:
        subj1_scan1, subj2_scan1, ..., subjectn_scan1, subj1_scan2, subject2_scan2, ...
    :param data: 3d numpy array (2 * n_subj, n_vox, n_dim)
    :return: data1, delta_data and noise covariance matrices.
    """
    n_subj = data.shape[0] // 2
    data1 = data[:n_subj].mean(axis=0)
    diffs = data[n_subj:] - data[:n_subj]
    delta_data = diffs.mean(axis=0)

    # sigma_n = np.array([np.cov(diffs[:, i, :].T) for i in range(diffs.shape[1])])
    offset = diffs - delta_data
    sigma_n = np.einsum('kij,kil->ijl', offset, offset) / (n_subj - 1)
    sigma_n = sigma_n / n_subj

    return data1, delta_data, sigma_n


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


def voxelwise_group_glm(data, weights, design_con, equal_samples=False, baseline_sigman=False):
    """
    Performs voxel-wise group glm on the given data with weights


    :param data: 3d numpy array (n_subj, n_vox, n_dim)
    :param weights: 2d numpy array (n_subj, n_vox)
    :param design_con: path to design.con file, the first contrast must be first group mean, the second contrast is the
    change across groups contrast
    :param equal_samples: take equal number of samples per class
    :param baseline_sigman: if true returns covariance matrix for the first class rather than the difference
    :return: data1, delta_data and noise covariance matrices.
    """
    c_names, c = loadcontrast(design_con)

    if data.shape[:2] == weights.shape:
        print(f'running glm for {data.shape[0]} subjects and {data.shape[1]} voxels.')
    else:
        raise ValueError(f' glm weights and data are not matched')
    if data.ndim == 2:
        data = data[..., np.newaxis]

    n_subj, n_vox, n_dim = data.shape
    copes = np.zeros((n_vox, n_dim, 2))
    varcopes = np.zeros((n_vox, n_dim, n_dim, 2))
    sigma_n_base = np.zeros((n_vox, n_dim, n_dim))

    for vox in range(n_vox):
        y = data[:, vox, :].T
        if equal_samples:
            n_wmh = weights[:, vox].sum().astype(int)
            all_0_idx = np.argwhere(weights[:, vox] == 0)
            c_idx = all_0_idx[np.random.randint(0, len(all_0_idx), n_wmh)]
            y = np.hstack([np.squeeze(y[:, c_idx]), y[:, weights[:, vox] == 1]])
            x = np.array([[1, 0]] * n_wmh + [[0, 1]] * n_wmh)
        else:
            x = np.zeros((n_subj, 2))
            x[:, 0] = weights[:, vox] == 0
            x[:, 1] = weights[:, vox] == 1
        beta = y @ np.linalg.pinv(x.T)
        copes[vox] = beta @ c.T
        sigma_n_base[vox] = np.cov(y[:, x[:, 0] == 1]) / x[:, 1].sum()
        # shape of the covariance matrix estimated from healthy subjects, but divided by the number of patients

        r = y - beta @ x.T
        sigma_sq = np.cov(r)
        varcopes[vox] = sigma_sq[..., np.newaxis] * np.diagonal(c @ np.linalg.inv(x.T @ x) @ c.T)

    data1 = copes[:, :, 0]
    delta_data = copes[:, :, 1]
    if baseline_sigman:
        sigma_n = sigma_n_base
    else:
        sigma_n = varcopes[..., 1]

    return data1, delta_data, sigma_n


## ==== Continuous portion ==== ##

def continuous_glm(data, design_mat, faulty_subjs=None):
    """
    Performs continuous glm on the given data


    :param data: 3d numpy array (n_subj, n_vox, n_dim)
    :param design_mat: path to design matrix .npz file (n_subj, n_phenotypes)
    :param faulty_subjs: index of faulty subjects to remove
    :return: data1, delta_data and noise covariance matrices.
    """
    x = np.load(design_mat)["a"]

    if faulty_subjs is not None:
        x = np.delete(x, faulty_subjs, axis=0)
        subject_deleting = np.load(design_mat)["b"][faulty_subjs]
        print("=== Removing subjects from design matrix ===")
        print(f"{subject_deleting}")

        print("=== Summary measures should already have removed these subjects in image_io function ===")

    # b is the number of subjects
    c_names = np.load(design_mat)["c"]  #c is the wanted variables
    effect_size = np.load(design_mat)["d"]  # d is effect sizes of the wanted variables
    c = np.eye(x.shape[-1])  # select each variable

    n_subj, n_vox, n_dim = data.shape

    if data.shape[0] == x.shape[0]:
        print(f'running glm for {data.shape[0]} subjects')
    else:
        raise ValueError(f'number of subjects in design matrix is {x.shape[0]} but'
                         f' {data.shape[0]} summary measures were loaded.')

    y = np.transpose(data, [1, 2, 0])  # make it (n_vox, n_dim, n_subj)
    beta = y @ np.linalg.pinv(x).T  # (n_vox, n_dim, n_phenotypes)
    copes = beta @ c.T
    r = y - beta @ x.T
    sigma_sq = np.array([np.cov(i) for i in r])
    varcopes = sigma_sq[..., np.newaxis] * np.diagonal(c @ np.linalg.pinv(x.T @ x) @ c.T)

    baseline = copes[:, :, 0]

    final_copes = copes[:, :, :c_names.shape[0]+1] #to remove confounds out of the saving process
    final_varcopes = varcopes[:, :, :c_names.shape[0] + 1]

    dictionary_of_deltas = {}
    dictionary_of_covars = {}

    for idx, var_name in enumerate(c_names):
        variable_effect = effect_size[idx]
        dictionary_of_deltas[var_name] = final_copes[:, :, idx + 1] * variable_effect
        dictionary_of_covars[var_name] = final_varcopes[..., idx + 1] * (variable_effect ** 2)

    if n_subj <= n_dim:
        Warning.warn('fewer samples than features, regularising sigma_n with 0.1 on diagonals')

        for var_name in c_names:
            dictionary_of_covars[var_name] += 0.1 * np.eye(dictionary_of_covars[var_name].shape[-1])

    return baseline, dictionary_of_deltas, dictionary_of_covars


def calc_r_squared(X, y):
    """
    Calculates the R^2 (explained variance) for each predictor variable
    in the design matrix X given the response matrix y.

    Parameters:
    -----------
    X : numpy array
        Design matrix of shape (n_samples, n_features)

    y : numpy array
        Response matrix of shape (n_samples, 1)

    Returns:
    --------
    r_squared : numpy array
        Array of length n_features containing the R^2 for each predictor variable
    """
    n_samples, n_features = X.shape

    # calculate mean of y
    y_mean = np.mean(y)

    # calculate total sum of squares
    total_sum_squares = np.sum((y - y_mean) ** 2)

    # initialize array to store R^2 for each predictor variable
    r_squared = np.zeros(n_features)

    # loop over each predictor variable
    for i in range(n_features):
        # calculate regression coefficients for ith predictor variable
        beta = np.linalg.lstsq(X[:, i].reshape(n_samples, 1), y, rcond=None)[0]

        # calculate predicted values for ith predictor variable
        y_pred = np.dot(X[:, i].reshape(n_samples, 1), beta)

        # calculate residual sum of squares for ith predictor variable
        residual_sum_squares = np.sum((y - y_pred) ** 2)

        # calculate R^2 for ith predictor variable
        r_squared[i] = 1 - (residual_sum_squares / total_sum_squares)

    return r_squared


def deconfounding_glm(data, confound_design_mat,faulty_subjs=None):
    """
    Regresses the summary measures with confounds to remove its effects
    Note that the summary measures do not need to be demeaned -- we account for it with a column of 1s + demeaned predictors
    Why? Because keeping the mean of the response variable is important for us ==> it is the baseline measure.

    :param data: 3d numpy array (n_subj, n_vox, n_dim)
    :param design_mat: path to design matrix .npz file (n_subj, n_phenotypes)
    :return: deconfounded data (n_subj, n_vox, n_dim) Assumes that confounds have already been demeaned, with an added column of 1s.
    """

    x = np.load(confound_design_mat)["a"]  # (n_subj, n_pheno+1)

    if faulty_subjs is not None:

        print("=== Removing subjects from design matrix ===")
        x = np.delete(x, faulty_subjs, axis=0)

        subject_deleting = np.load(confound_design_mat)["b"][faulty_subjs]
        print(f"{subject_deleting}")

        print("=== Summary measures should already have removed these subjects (i.e., when loading original SM) ===")



    x_without_intercept = np.copy(x[:, 1:])  # (n_subj, n_phenotypes)

    if data.shape[0] == x.shape[0]:
        print(f'running glm for {data.shape[0]} subjects')
    else:
        raise ValueError(f'number of subjects in design matrix is {x.shape[0]} but'
                         f' {data.shape[0]} summary measures were loaded.')

    y = np.transpose(data, [1, 2, 0])  # make it (n_vox, n_dim, n_subj)
    beta = y @ np.linalg.pinv(
        x).T  # beta is derived with intercept included (i.e., it is observed change FROM the intercept/y_mean) # (n_vox, n_dim, n_phenotypes)
    beta_without_beta_0 = np.copy(beta[:, :, 1:])
    deconfounded_y = y - beta_without_beta_0 @ x_without_intercept.T  # we don't include the intercept because we still want the mean here (i.e., including the intercept removes the mean)

    deconfounded_y_transposed = np.transpose(deconfounded_y, [2, 0, 1])  # (n_subj, n_vox, n_dim); back to original form

    return deconfounded_y_transposed


def squared(y):
    return y @ np.transpose(y, [0, 2, 1])
