#!/usr/bin/env python3


import numpy as np
from fsl.data.image import Image

from bench import change_model, image_io


def continuous_inference_from_cli(args):
    c_names = np.load(args.designmat)["c"] #c_names are variables (i.e., different phenotypes names)

    baseline, dictionary_of_deltas, dictionary_of_covars, summary_names = read_continuous_glm_results(
        args.glmdir,
        c_names)

    ch_mdl = change_model.ChangeModel.load(args.model)
    sm_names = ch_mdl.summary_names

    reg = np.eye(len(sm_names)) * 1e-5  # regularisation for low covariances

    # perform inference:

    for variable in c_names:
        posteriors, predictions, peaks, bad_samples = ch_mdl.infer(data=baseline,
                                                                   delta_data=dictionary_of_deltas[variable],
                                                                   sigma_n=dictionary_of_covars[variable] + reg,
                                                                   integral_bound=float(args.integral_bound),
                                                                   parallel=True)

        # save the results:
        image_io.write_continuous_inference_results(path=f'{args.output}/{variable}',
                                                    model_names=ch_mdl.model_names,
                                                    predictions=predictions,
                                                    posteriors=posteriors,
                                                    peaks=peaks,
                                                    mask=f'{args.glmdir}/valid_mask',
                                                    variable=variable)

        dv, offset, deviation = ch_mdl.estimate_quality_of_fit(y1=baseline,
                                                               dy=dictionary_of_deltas[variable],
                                                               sigma_n=dictionary_of_covars[variable] + reg,
                                                               predictions=predictions,
                                                               amounts=peaks)

        image_io.write_nifti(deviation[:, np.newaxis], f'{args.glmdir}/valid_mask',
                             f'{args.output}/{variable}/sigma_deviations.nii.gz')

    print(f'Analysis completed successfully.')


def read_continuous_glm_results(glm_dir, c_names, mask_add=None):
    """
    :param glm_dir: path to the glm dir, it must contain data.nii.gz, delta_data.nii.gz, variance.nii.gz,
    and valid_mask.nii.gz
    :param mask_add: address of mask file, by default it uses the mask in glm dir.
    :return: tuple (data (n, d), delta_data (n, d), sigma_n(n, d, d) )
    """

    if mask_add is None:
        mask_add = glm_dir + '/valid_mask.nii.gz'

    mask_img = np.nan_to_num(Image(mask_add).data)
    ## === load baseline === ##
    baseline = Image(f'{glm_dir}/baseline.nii.gz').data[mask_img > 0, :]
    n_vox, n_dim = baseline.shape

    dictionary_of_deltas = {}
    dictionary_of_covars = {}

    ## === load the different contrasts change + variances === ##

    for c in c_names:

        delta_data = Image(f'{glm_dir}/delta_{c}.nii.gz').data[mask_img > 0, :]
        variances = Image(f'{glm_dir}/covariances_{c}.nii.gz').data[mask_img > 0, :]

        # === get the sigma === ##

        tril_idx = np.tril_indices(n_dim)
        diag_idx = np.diag_indices(n_dim)
        sigma_n = np.zeros((n_vox, n_dim, n_dim))
        for i in range(n_vox):
            sigma_n[i][tril_idx] = variances[i]
            sigma_n[i] = sigma_n[i] + sigma_n[i].T
            sigma_n[i][diag_idx] /= 2

        dictionary_of_deltas[c] = delta_data
        dictionary_of_covars[c] = sigma_n

    with open(f'{glm_dir}/summary_names.txt', 'r') as reader:
        summary_names = [line.rstrip() for line in reader]

    return baseline, dictionary_of_deltas, dictionary_of_covars, summary_names