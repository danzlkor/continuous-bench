"""
This module contains functions to fit spherical harmonics to diffusion data.
"""


def fit_summary(diff: list, bvecs: list, bvals: str, xfms: list, roi_mask: str, sph_degree: int, output: str):
    """
    Resamples diffusion data to std space, and fits spherical harmonics to the input images and stores the outputs per
    seubject in a separate image.

    :param diff: list of filenames for diffusion images.
    :param bvecs: list of filenames for bvecs
    :param bvals: a string containg path to bval file, this must be the same for all subjects.
    :param xfms: transformation from diffusion space to standard space
    :param roi_mask: mask in standard space that defines regions of interest
    :param output: path to the output directory
    :param sph_degree: degree for spherical harmonics.
    :return: saves images to the specified path and returns true flag if the process done completely.
    """
    pass


def read_summaries(path: str):
    """
    Reads summary measure images
    :param path: path to the summary measurements
    :return: 3d numpy array containing summary measurements, inclusion mask
    """
    pass
