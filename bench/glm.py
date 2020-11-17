"""
This module reads diffusion data and returns data in proper format for inference
"""


def group_glm(data, design_mat, design_con):
    """
    Performs group glm on the given data 
    :param data: 3d numpy array (n_subj, n_vox, n_dim) 
    :param design_mat: path to design.mat file
    :param design_con: path to design.con file, the first contrast must be first group mean, the second contrast is the 
    change across groups contrast
    :return: data1, delta_data and noise covariance matrices.
    """
    pass

