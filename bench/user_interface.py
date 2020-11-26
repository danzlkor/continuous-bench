"""
This module is to parse inputs from commandline and call the proper functions from other modules.
"""
import argparse
import os
from warnings import warn
import numpy as np
from bench import change_model, glm, summary_measures, diffusion_models, acquisition
from fsl.data.image import Image
import nibabel as nib


def inference_from_cmd(argv=None):
    """
    Wrapper function to parse the input from commandline and run the requested pipeline.


    :param argv: string from command line containing all required inputs 
    :return: saves the output images to the specified path
    """

    args = inference_parse_args(argv)
    """
    Main function that calls all the required steps.
    :param args: output namespace from argparse from commandline input  
    :return: runs the process and save images to the specified path
    :raises: when the input files are not available
    """

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # if summaries are not provided fit summaries:
    if args.summary_dir is None:
        print('Fitting summary measurements to the data.')
        args.summary_dir = f'{args.output}/SummaryMeasures'
        job_id = summary_measures.fit_summary_to_dataset(data=args.data, bvecs=args.bvecs, roi_mask=args.mask,
                                                         bvals=args.bval, xfms=args.xfm, output=args.summary_dir,
                                                         sph_degree=args.sph_degree)
    else:
        print(f'Loading summary_measures from {args.summary_dir}')

    if args.model is None:
        print(f'Summary measures are stored in {args.summary_dir}.\n done.')
    else:
        summaries, invalid_vox = summary_measures.read_summary_images(summary_dir=args.summary_dir, mask=args.mask)

        summaries = summaries[:, invalid_vox == 0, :]
        # perform glm:
        data, delta_data, sigma_n = glm.group_glm(summaries, args.design_mat, args.design_con)

        # perform inference:
        ch_mdl = change_model.ChangeModel.load(args.model)
        posteriors, predictions = ch_mdl.predict(data, delta_data, sigma_n)

        # save the results:
        vec_names = ['No-change'] + [m.name for m in ch_mdl.models]
        maps_dir = f'{args.output}/PosteriorMaps/{ch_mdl.name}'
        write_nifti(vec_names, posteriors, args.mask, maps_dir, invalid_vox)
        print(f'Analysis completed successfully, the posterior probability maps are stored in {maps_dir}')


def inference_parse_args(argv):
    """
    Parses the commandline input anc checks for the consistency of inputs
    :param argv: input string from commandline
    :return: arg namespce from argparse
    :raises: if the number of provided files do not match with other arguments
    """

    parser = argparse.ArgumentParser("BENCH: Bayesian EstimatioN of CHange")
    parser.add_argument("--mask", help="Mask in standard space indicating which voxels to analyse", required=True)
    parser.add_argument("--output", help="Path to the output directory", required=True)

    inference = parser.add_argument_group("Inference arguments")
    inference.add_argument("--design-mat", help="Design matrix for the group glm", required=False)
    inference.add_argument("--design-con", help="Design contrast for the group glm", required=False)
    inference.add_argument("--model", help="Forward model, either name of a standard model or full path to"
                                           "a trained model json file", default=None, required=False)
    inference.add_argument("--sigma-v", default=0.1,
                           help="Standard deviation for prior change in parameters (default = 0.1)", required=False)

    # pre-processing arguments:
    preproc = parser.add_argument_group("Summary fit arguments")
    preproc.add_argument('--summary-dir', default=None,
                         help='Path to the pre-computed summary measurements', required=False)
    preproc.add_argument("--data", nargs='+', help="List of dMRI data in subject native space", required=False)
    preproc.add_argument("--xfm", help="Non-linear warp fields mapping subject diffusion space to the mask space",
                         nargs='+', metavar='xfm.nii', required=False)
    preproc.add_argument("--bvecs", nargs='+', metavar='bvec', required=False,
                         help="Gradient orientations for each subject")
    preproc.add_argument("--bval", metavar='bval', required=False,
                         help="b_values (should be the same for all subjects")
    preproc.add_argument("--sph_degree", default=4, help=" Degree for spherical harmonics summary measurements",
                         required=False, type=int)

    args = parser.parse_args(argv)

    if not os.path.exists(args.mask):
        raise FileNotFoundError('Mask file was not found.')

    if os.path.isdir(args.output):
        warn('Output directory already exists, contents might be overwritten.')
        if not os.access(args.output, os.W_OK):
            raise PermissionError('user does not have permission to write in the output location.')
    else:
        os.makedirs(args.output, exist_ok=True)

    if args.summary_dir is None:
        n_subjects = min(len(args.xfm), len(args.data), len(args.bvecs))
        if len(args.data) > n_subjects:
            raise ValueError(f"Got more diffusion MRI dataset than transformations/bvecs: {args.data[n_subjects:]}")
        if len(args.xfm) > n_subjects:
            raise ValueError(f"Got more transformations than diffusion MRI data/bvecs: {args.xfm[n_subjects:]}")
        if len(args.bvecs) > n_subjects:
            raise ValueError(f"Got more bvecs than diffusion MRI data/transformations: {args.bvecs[n_subjects:]}")

        for subj_idx, (nl, d, bv) in enumerate(zip(args.xfm, args.data, args.bvecs), 1):
            print(f'Scan {subj_idx}: dMRI ({d} with {bv}); transform ({nl})')
            for f in [nl, d, bv]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f'{f} not found. Please check the input files.')

        if not os.path.exists(args.bval):
            raise FileNotFoundError(f'{args.bval} not found. Please check the paths for input files.')

    if args.model is not None:
        if args.design_mat is None:
            raise RuntimeError('For inference you have to provide a design matrix file.')
        elif not os.path.exists(args.design_mat):
            raise FileNotFoundError(f'{args.design_mat} file not found.')

        if args.design_con is None:
            raise RuntimeError('For inference you need to provide a design contrast file.')
        elif not os.path.exists(args.design_con):
            raise FileNotFoundError(f'{args.design_con} file not found.')

    return args


def train_from_cmd(argv=None):
    args = train_parse_args(argv)
    available_models = list(diffusion_models.prior_distributions.keys())
    funcdict = {name: f for (name, f) in diffusion_models.__dict__.items() if name in available_models}
    forward_model = funcdict[args.model]
    param_dist = diffusion_models.prior_distributions[args.model]

    bvals = np.genfromtxt(args.bval)
    idx_shells, shells = acquisition.ShellParameters.create_shells(bval=bvals)

    bvecs = np.array(diffusion_models.spherical2cart(
        *diffusion_models.uniform_sampling_sphere(len(idx_shells)))).T
    acq = acquisition.Acquisition(shells, idx_shells, bvecs)
    trainer = change_model.Trainer(forward_model=diffusion_models.decorator(forward_model),
                                   x=(acq, int(args.d)),
                                   vecs=args.change_vecs,
                                   param_prior_dists=param_dist)

    ch_model = trainer.train(n_samples=int(args.n), k=int(args.k), model_name=forward_model.__name__,
                             poly_degree=int(args.p), regularization=float(args.alpha))
    ch_model.save(path='', file_name=args.output)
    print('Change model trained successfully')


def train_parse_args(argv):
    parser = argparse.ArgumentParser("BENCH Train: Training models of change")

    required = parser.add_argument_group("required arguments")
    required.add_argument("--model", help="Forward model name", required=True)
    required.add_argument("--output", help="name of the trained model", required=True)
    required.add_argument("--bval", required=True)

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-k", default=100, type=int, help="number of nearest neighbours", required=False)
    optional.add_argument("-n", default=1000, type=int, help="number of training samples", required=False)
    optional.add_argument("-p", default=2, type=int, help="polynomial degree for design matrix", required=False)
    optional.add_argument("-d", default=4, type=int, help="spherical harmonics degree", required=False)
    optional.add_argument("--alpha", default=0.5, type=float, help="regularization weight", required=False)
    optional.add_argument("--change-vecs", help="vectors of change", default=None, required=False)

    args = parser.parse_args(argv)

    # handle the problem of getting single arg for list arguments:
    if args.model in diffusion_models.prior_distributions.keys():
        print('Parameters of the forward model are:')
        print(list(diffusion_models.prior_distributions[args.model].keys()))
    else:
        model_names = ', '.join(list(diffusion_models.prior_distributions.keys()))
        raise ValueError(f'The forward model is not defined in the library. '
                         f'Defined models are:\n {model_names}')

    return args


def write_nifti(model_names: list, posteriors: np.ndarray, mask_add: str, output='.', invalids=None):
    """
    Writes the results to nifti files per change model.

    """
    if os.path.isdir(output):
        warn('Output directory already exists, contents might be overwritten.')
    else:
        os.makedirs(output, exist_ok=True)

    winner = np.argmax(posteriors, axis=1)
    mask = Image(mask_add)

    std_indices = np.array(np.where(mask.data > 0)).T
    std_indices_valid = std_indices[[not v for v in invalids]]
    std_indices_invalid = std_indices[invalids == 1]

    for s_i, s_name in enumerate(model_names):
        data = posteriors[:, s_i]
        tmp1 = np.zeros(mask.shape)
        tmp1[tuple(std_indices_valid.T)] = data
        tmp1[tuple(std_indices_invalid.T)] = np.nan
        tmp2 = np.zeros(mask.shape)
        tmp2[tuple(std_indices_valid.T)] = winner == s_i
        tmp2[tuple(std_indices_invalid.T)] = np.nan
        mat = np.stack([tmp1, tmp2], axis=-1)

        fname = f"{output}/{s_name}.nii.gz"
        nib.Nifti1Image(mat, mask.nibImage.affine).to_filename(fname)
