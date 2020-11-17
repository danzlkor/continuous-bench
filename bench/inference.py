"""
This module performs inference using change models
"""


def compute_posteriors(change_model, sigma_v, data, delta_data, sigma_n):
    """

    :param change_model: class containing change models.
    :param sigma_v: a float or dict containing the std of prior for delta_v (delta_v ~ N(0, sigma_v)
    :param data: numpy array (n_vox, n_dim) containing the first group average
    :param delta_data: numpy array (n_vox, n_dim) containing the change between groups.
    :param sigma_n: numpy array or list (n_vox, n_dim, n_dim)
    :return: posterior probabilities for each voxel (n_vox, n_params)
    """
    pass
