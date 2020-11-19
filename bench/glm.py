
"""
This module reads diffusion data and returns data in proper format for inference
"""

import numpy as np
from fsl.data.featdesign import loadDesignMat


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
    sigma_n = varcopes[..., 1]

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
