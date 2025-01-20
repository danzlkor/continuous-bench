#!/usr/bin/env python3
"""
This module is to parse inputs from commandline and call the proper functions from other modules.
"""

import argparse
import os

import numpy as np
from file_tree import FileTree
from fsl.utils.fslsub import submit
from scipy import stats as st

from bench import change_model, glm, summary_measures, diffusion_models, acquisition, image_io, model_inversion


def main(argv=None):
    """
    Wrapper function to parse the input from commandline and run the requested pipeline.

    :param argv: string from command line containing all required inputs 
    :return: saves the output images to the specified path
    """
    args = parse_args(argv)
    args.func(args)


def parse_args(argv):
    """
    Parses the commandline input anc checks for the consistency of inputs
    :param argv: input string from commandline
    :return: arg namespce from argparse
    :raises: if the number of provided files do not match with other arguments
    """

    parser = argparse.ArgumentParser("BENCH: Bayesian EstimatioN of CHange")
    parser.set_defaults(func=print_avail_commands)

    subparsers = parser.add_subparsers(dest='commandname')
    diff_train_parser = subparsers.add_parser('diff-train')
    diff_summary_parser = subparsers.add_parser('diff-summary')
    submit_summary_parser = subparsers.add_parser('submit-summary')
    diff_normalise_parser = subparsers.add_parser('diff-normalise')
    glm_parser = subparsers.add_parser('glm')
    inference_parser = subparsers.add_parser('inference')

    # === Adding my own GLM and inference parser === ##
    deconfounding_summary_parser = subparsers.add_parser('deconfound-summary')
    continuous_glm_parser = subparsers.add_parser('continuous-glm')
    continuous_inference_parser = subparsers.add_parser('continuous-inference')
    deconfound_glm_parser = subparsers.add_parser('deconfound-inference')

    # train arguments:
    train_required = diff_train_parser.add_argument_group("required arguments")
    available_models = list(diffusion_models.prior_distributions.keys())
    train_required.add_argument("-m", "--model",
                                help=f"name of the forward model. Available models:\n{available_models}", required=True)
    train_required.add_argument("-o", "--output", help="name of the trained model", required=True)
    train_required.add_argument("-b", "--bvals", help="b-values for training", required=True)

    train_optional = diff_train_parser.add_argument_group("optional arguments")
    train_optional.add_argument("-n", "--samples", default=10000, type=int,
                                help="number of training samples (default=10000)",
                                required=False)
    train_optional.add_argument("--bvecs", default=None,
                                help="Gradient directions. (default: 64 uniform samples over sphere)",
                                required=False)
    train_optional.add_argument("--b0-thresh", default=1, type=float,
                                help="threshold for b0 (default=1)")
    train_optional.add_argument("-p", "--poly-degree", default=2, type=int,
                                help="polynomial degree for mean (default=2)", required=False)
    train_optional.add_argument("-ps", default=1, type=int, help="polynomial degree for variance (default=1)",
                                required=False)
    train_optional.add_argument("--summarytype", default='sh', type=str,
                                help='type of summary measurements. Either shm (spherical harmonic model)'
                                     ' or dtm (diffusion tensor model) (default shm)', required=False)

    train_optional.add_argument("-d", default=2, type=int,
                                help=" maximum degree for summary measures (must be even numbers, default=2)",
                                required=False)
    train_optional.add_argument("--alpha", default=0.0, type=float,
                                help="regularisation weight for training regression models(default=0)", required=False)
    train_optional.add_argument("--change-vecs",
                                help="text file for defining vectors of change (refer to documentations)", default=None,
                                required=False)

    train_optional.add_argument('--verbose', help='flag for printing optimisation outputs', dest='verbose',
                                action='store_true', default=False)
    diff_train_parser.set_defaults(func=train_from_cli)

    # fit summary arguments:
    submit_summary_parser.add_argument("--file-tree", help="file-tree text file", required=True)
    submit_summary_parser.add_argument("--mask", help="mask in standard space.", required=True)
    submit_summary_parser.add_argument("--summarytype", default='sh', type=str,
                                       help='type of summary measurements. either shm (spherical harmonics)'
                                            ' or dtm (diffusion tensor) (default shm)', required=False)

    submit_summary_parser.add_argument("--shm-degree", default=2,
                                       help=" degree for spherical harmonics (even number)",
                                       required=False, type=int)
    submit_summary_parser.add_argument("--b0-thresh", default=1, type=float,
                                       help="b0-threshhold (default=10)")
    submit_summary_parser.add_argument("--logdir",
                                       help=" log directory")
    submit_summary_parser.set_defaults(func=submit_summary)

    # single subject summary:
    diff_summary_parser.add_argument('--data', required=True)
    diff_summary_parser.add_argument('--bvecs', required=True)
    diff_summary_parser.add_argument('--bvals', required=True)
    diff_summary_parser.add_argument('--mask', required=True)
    diff_summary_parser.add_argument('--xfm',
                                     help='Transformation from diffusion to mask', required=False)
    diff_summary_parser.add_argument('--output', required=True)
    diff_summary_parser.add_argument('--shm-degree', default=2, type=int, required=False)
    diff_summary_parser.add_argument('--summarytype', default='sh', required=False)
    diff_summary_parser.add_argument('--b0-thresh', default=1, type=float, required=False)
    diff_summary_parser.add_argument('--normalise', dest='normalise', action='store_true')

    diff_summary_parser.set_defaults(func=summary_from_cli)

    # normalization args
    diff_normalise_parser.add_argument('--study-dir', default=None,
                                       help='Path to the un-normalised summary measurements', required=True)
    # deconfound summary arguments:
    deconfounding_summary_parser.add_argument("--summarydir", help='Path to summary measurements folder', required=True)
    deconfounding_summary_parser.add_argument("--mask", help='Path to the mask', required=True)
    deconfounding_summary_parser.add_argument("--confoundmat", help="Confound design matrix for the glm (.npz file)",
                                              required=True)
    deconfounding_summary_parser.add_argument("--output", help='Output directory (i.e., deconfound summary measure)',
                                              required=True)
    deconfounding_summary_parser.set_defaults(func=deconfounding_summary_from_cli)

    # glm arguments:
    glm_parser.add_argument("--summarydir", help='Path to summary measurements folder', required=True)
    glm_parser.add_argument("--mask", help='Path to the mask', required=True)
    glm_parser.add_argument("--designmat", help="Design matrix for the group glm", required=False)
    glm_parser.add_argument("--designcon", help="Design contrast for the group glm", required=False)
    glm_parser.add_argument('--paired', dest='paired', action='store_true')
    glm_parser.add_argument("--output", help='Output directory', required=True)
    glm_parser.set_defaults(func=glm_from_cli)

    # continuous glm arguments:
    continuous_glm_parser.add_argument("--summarydir", help='Path to summary measurements folder', required=True)
    continuous_glm_parser.add_argument("--mask", help='Path to the mask', required=True)
    continuous_glm_parser.add_argument("--designmat", help="Design matrix for the glm (.npz file)",
                                       required=False)
    continuous_glm_parser.add_argument("--output", help='Output directory', required=True)
    continuous_glm_parser.add_argument("--faulty", default=None, help='Path to faulty subjects', required=False)
    continuous_glm_parser.set_defaults(func=continuous_glm_from_cli)

    # continuous glm with deconfounding arguments:
    deconfound_glm_parser.add_argument("--summarydir", help='Path to summary measurements folder', required=True)
    deconfound_glm_parser.add_argument("--mask", help='Path to the mask', required=True)
    deconfound_glm_parser.add_argument("--designmat", help="Design matrix for the glm (.npz file)",
                                       required=False)
    deconfound_glm_parser.add_argument("--confoundmat", help="Confound design matrix for the glm (.npz file)",
                                       required=False)
    deconfound_glm_parser.add_argument("--output", help='Output directory', required=True)
    deconfound_glm_parser.set_defaults(func=continuous_glm_with_deconfounding_from_cli)

    # continuous inference arguments:
    continuous_inference_parser.add_argument("--model", help="Path to a trained model of change",
                                             default=None, required=True)
    continuous_inference_parser.add_argument('--glmdir', help="Path to the glm dir", required=True)
    continuous_inference_parser.add_argument('--output', help="Path to the output dir", required=True)
    continuous_inference_parser.add_argument("--mask",
                                             help='Path to the mask (if none passed uses the valid_mask in glm folder)',
                                             default=None, required=False)
    continuous_inference_parser.add_argument("--integral-bound",
                                             help='The maximum value for integrating over the amount of change. (default=1)',
                                             default=1.0, required=False)
    continuous_inference_parser.add_argument("--designmat",
                                             help="Design matrix for the group glm; needed to get the variables of interest (.npz file)",
                                             required=True)
    continuous_inference_parser.set_defaults(func=continuous_inference_from_cli)

    # inference arguments:
    inference_parser.add_argument("--model", help="Path to a trained model of change (output of bench diff-train)",
                                  default=None, required=True)
    inference_parser.add_argument('--glmdir', help="Path to the glm dir", required=True)
    inference_parser.add_argument('--output', help="Path to the output dir", required=True)
    inference_parser.add_argument("--mask", help='Path to the mask (if none passed uses the valid_mask in glm folder)',
                                  default=None, required=False)
    inference_parser.add_argument("--integral-bound",
                                  help='The maximum value for integrating over the amount of change. (default=1)',
                                  default=1.0, required=False)
    inference_parser.set_defaults(func=inference_from_cli)

    args = parser.parse_args(argv)

    return args


def print_avail_commands(argv=None):
    print('BENCH: Bayesian EstimatioN of CHange')
    print('usage: bench <command> [options]')
    print('')
    print('available commands: diff-train, diff-summary,diff-single-summary,diff-normalise,glm,inference')


def train_from_cli(args):
    available_models = list(diffusion_models.prior_distributions.keys())
    if args.model in available_models:
        param_prior_dists = diffusion_models.prior_distributions[args.model]
        p_names = [i for p in param_prior_dists.keys() for i in ([p] if isinstance(p, str) else p)]
        print(f'Parameters of {args.model} are:{p_names}')
    else:
        model_names = ', '.join(list(diffusion_models.prior_distributions.keys()))
        raise ValueError(f'model {args.model} is not defined in the library. '
                         f'Current defined models are:\n {model_names}')

    funcdict = {name: f for (name, f) in diffusion_models.__dict__.items() if name in available_models}
    forward_model = funcdict[args.model]
    param_dist = diffusion_models.prior_distributions[args.model]

    bvals = acquisition.read_bvals(args.bvals)
    if args.bvecs is None:
        bvecs = np.array(diffusion_models.spherical2cart(*diffusion_models.uniform_sampling_sphere(len(bvals)))).T
    else:
        bvecs = acquisition.read_bvecs(args.bvecs)

    if args.change_vecs is not None:
        with open(args.change_vecs, 'r') as reader:
            args.change_vecs = [line.rstrip() for line in reader]

    func, summary_names = summary_measures.summary_decorator(
        model=forward_model, bval=bvals, bvec=bvecs, summary_type=args.summarytype, shm_degree=int(args.d))
    print('The model is trained using summary measurements:', summary_names)
    trainer = change_model.Trainer(
        forward_model=func, kwargs={'noise_std': 0.}, change_vecs=args.change_vecs,
        summary_names=summary_names, priors=param_dist)

    ch_model = trainer.train_ml(n_samples=int(args.n),
                                mu_poly_degree=int(args.p),
                                sigma_poly_degree=int(args.ps),
                                alpha=float(args.alpha),
                                parallel=True,
                                verbose=args.verbose)

    ch_model.forward_model_name = forward_model.__name__
    ch_model.measurement_names = summary_names
    ch_model.save(path='', file_name=args.output)
    print('All change models were trained successfully.')


def summary_from_cli(args):
    """
        Wrapper function that parses the input from commandline
        :param args: list of strings containing all required parameters for fit_summary_single_subj()
        """
    print('args', args)

    if args.xfm is None:
        print('no transformation is provided, the results will be in the same space as the input image.')
        data = image_io.read_image(args.data, args.mask)
        valid_vox = np.ones(data.shape[0])
    else:

        rootpath = os.path.dirname(args.output)
        subj = os.path.basename(args.output)
        def_field = f"{rootpath}/{subj}_tmp.nii.gz"
        data, valid_vox = image_io.sample_from_native_space(args.data, args.xfm, args.mask, def_field)

    acq = acquisition.Acquisition.from_bval_bvec(args.bvals, args.bvecs, args.b0_thresh)
    summaries = summary_measures.fit_shm(data, acq.bvals, acq.bvecs, shm_degree=args.shm_degree)
    names = summary_measures.summary_names(bvals=acq.bvals,
                                           b0_threshold=args.b0_thresh,
                                           summarytype=args.summarytype,
                                           shm_degree=args.shm_degree)

    if args.normalise:
        print('Summary measures are normalised by b0_mean.')
        # summaries = summary_measures.normalise_summaries(summaries, names)
        # names = [f'{n}/b0' for n in names[1:]]

        summaries = summary_measures.normalise_summaries(summaries, names, delete=False)  # don't delete for your case
        names = [f'{n}/b0' for n in names]

    # write to nifti:
    image_io.write_nifti(summaries, args.mask, args.output, np.logical_not(valid_vox))
    with open(f'./summary_names.txt', 'w') as f:
        for t in names:
            f.write("%s\n" % t)
        f.close()
    print(f'Summary measurements are computed.')


def submit_summary(args):
    """
    submits jobs to cluster (or runs serially) to compute summary measurements for the given subjects
    :param args: namespace form parseargs output that contains all addresses to the required files
    (masks, transformations, diffusion data, bvalues, and bvecs) and output folder
    :return: job ids (the results are saved to files once the jobs are done)
    """
    print(args)
    tree = FileTree.read(args.file_tree).update_glob("data")
    subject_jobs = []

    for subject_tree in tree.iter("data"):
        cmd = (
            "bench", "diff-summary",
            "--data", subject_tree.get("data"),
            "--bvals", subject_tree.get("bvals"),
            "--bvecs", subject_tree.get("bvecs"),
            "--xfm", subject_tree.get("diff2std"),
            "--mask", args.mask,
            "--summarytype", args.summarytype,
            "--shm-degree", str(args.shm_degree),
            "--b0-thresh", str(args.b0_thresh),
            "--output", subject_tree.get("subject_summary_dir", make_dir=True)
        )
        # main(cmd[1:])
        subject_jobs.append(submit(cmd, logdir=args.logdir, job_name=f'bench.summary'))

    acq = acquisition.Acquisition.from_bval_bvec(subject_tree.get("bvals"), subject_tree.get("bvecs"))
    summary_names = summary_measures.summary_names(acq, args.summarytype, args.shm_degree)
    with open(f'{subject_tree.get("summary_dir")}/summary_names.txt', 'w') as f:
        for t in summary_names:
            f.write("%s\n" % t)
        f.close()
    print(f'Summary measurements are computed.')


from tqdm import tqdm


def deconfounding_summary_from_cli(args):
    exclude = True  # exclude any faulty subjects

    assert (args.confoundmat is not None)

    os.makedirs(args.output, exist_ok=True)

    ## Get all proposed summary files

    subject_ids_from_dm = np.array(np.load(args.confoundmat)["b"])
    proposed_summary_files = np.array([args.summarydir + f'/{subj}.nii.gz' for subj in subject_ids_from_dm])

    not_in_sm = []
    for subj in subject_ids_from_dm:
        summary_file = args.summarydir + f'/{subj}.nii.gz'
        if not os.path.exists(summary_file):
            not_in_sm.append(subj)

    not_in_sm = np.array(not_in_sm)

    if len(not_in_sm) > 0:
        print("Some subjects not present. Remove from design matrix first")
        print(not_in_sm)

        np.savez_compressed(
            file=f"{args.output}/missing_subjects",
            a=not_in_sm)
        return

    ## Read summary measures

    print(f" === Loading {proposed_summary_files.shape[0]} summary measure niftis === ")

    ## Remove the faulty subject you have discovered in the past

    """

    print(f" == Removing subject {os.path.basename(proposed_summary_files[340])}...")
    proposed_summary_files =np.delete(proposed_summary_files, 340) #subject 341 = index + 1 needs to be excluded.
    print(f"...now with {proposed_summary_files.shape[0]} summary measure niftis == ")
    
    """

    summaries, invalid_vox, summary_names, subj_names, faulty_subjs = image_io.read_summary_images_from_predefined_list(
        summary_files=proposed_summary_files,
        mask=args.mask,
        exclude=exclude)  # print the faulty_subjs list

    summaries = summaries[:, invalid_vox == 0, :]

    ## Perform deconfounding

    summaries = glm.deconfounding_glm(summaries, args.confoundmat, faulty_subjs)

    if exclude is True:
        subjects_excluded = np.copy(subject_ids_from_dm)
        subjects_excluded = subjects_excluded[faulty_subjs]  # get the subject ID of those actually excluded

        subject_ids_from_dm = np.delete(subject_ids_from_dm, faulty_subjs, axis=0)  # final subject ID list

        ## save faulty subject index + ID

        np.savez_compressed(file=f"{args.output}/subjects_excluded",
                            a=subjects_excluded,
                            b=faulty_subjs)

    ## Save deconfounded summaries as nifti, one for each subject
    # write to nifti:

    for idx, subj in tqdm(enumerate(subject_ids_from_dm), desc="Saving subject's deconfounded niftis"):
        image_io.write_nifti(summaries[idx, :, :],
                             args.mask,
                             f'{args.output}/{subj}.nii.gz',
                             invalid_vox)

    valid_mask = np.ones((summaries.shape[1], 1))
    image_io.write_nifti(valid_mask, args.mask, args.output + '/valid_mask.nii.gz', invalid_vox)

    ## save used ID

    np.savez_compressed(file=f"{args.output}/subjects_used",
                        a=subject_ids_from_dm)


def glm_from_cli(args):
    """
    Runs glm on the summary meausres.
    :param args: output from argparse, should contain desing matrix anc contrast addresss, summary_dir and masks
    :return:
    """
    assert args.paired or (args.designmat is not None and args.designcon is not None)

    os.makedirs(args.output, exist_ok=True)
    summaries, invalid_vox, summary_names = image_io.read_summary_images(summary_dir=args.summarydir, mask=args.mask)
    summaries = summaries[:, invalid_vox == 0, :]

    if args.paired:
        data, delta_data, sigma_n = glm.group_glm_paired(summaries)
    else:
        data, delta_data, sigma_n = glm.group_glm(summaries, args.designmat, args.designcon)

    image_io.write_glm_results(data, delta_data, sigma_n, summary_names,
                               args.mask, invalid_vox, args.output)
    y_norm, dy_norm, sigma_n_norm = \
        summary_measures.normalise_summaries(data, summary_names, delta_data, sigma_n)

    image_io.write_glm_results(y_norm, dy_norm, sigma_n_norm, summary_names,
                               args.mask, invalid_vox, args.output + '_normalised')
    print(f'GLM is done. Results are stored in {args.output}')


def continuous_glm_from_cli(args):
    """
    Runs glm on the summary meausres.
    :param args: output from argparse, should contain desing matrix anc contrast addresss, summary_dir and masks
    :return:
    """

    assert (args.designmat is not None)

    subject_ids_from_dm = np.array(np.load(args.designmat)["b"])

    ## Check to see if there are faulty subjects to remove

    if args.faulty is None:
        faulty_subjs = None
    else:
        subjects_excluded = np.load(args.faulty)["a"]

        # get the index of these faulty subjects in subject_ids_from_dm
        matching_elements = np.intersect1d(subjects_excluded, subject_ids_from_dm)
        faulty_subjs = np.where(np.isin(subject_ids_from_dm, matching_elements))[0]

        # Remove the matching elements from subject_ids_from_dm
        subject_ids_from_dm = np.delete(subject_ids_from_dm, faulty_subjs)

    os.makedirs(args.output, exist_ok=True)

    ## Check to make sure the subject IDs for the design matrix match ARE in the summary measure directory

    proposed_summary_files = np.array(
        [args.summarydir + f'/{subj}.nii.gz' for subj in subject_ids_from_dm])  # ALREADY removed the faulty subjects

    not_in_sm = []
    for subj in subject_ids_from_dm:
        summary_file = args.summarydir + f'/{subj}.nii.gz'
        if not os.path.exists(summary_file):
            not_in_sm.append(subj)

    not_in_sm = np.array(not_in_sm)

    if len(not_in_sm) > 0:
        print("Some subjects not present. Remove from design matrix first")
        print(not_in_sm)

        np.savez_compressed(
            file=f"{args.output}/missing_subjects",
            a=not_in_sm)
        return

    ## Exclude with pre-defined list
    ## Read summary measures and perform GLM

    summaries, invalid_vox, summary_names, subj_names, faulty_subjs = image_io.read_summary_images_from_predefined_list(
        summary_files=proposed_summary_files,
        mask=args.mask,
        exclude=True)  # note that proposed_summary_files already has excluded subject removed, if you include an
    # explicit faulty_subjs (so you set this to false if you do; otherwise, set this to true if no faulty subjs input)

    # summaries, invalid_vox, summary_names, subj_names, _ = image_io.read_summary_images_from_predefined_list(
    #    summary_files=proposed_summary_files,
    #    mask=args.mask,
    #    exclude=False) # note that proposed_summary_files already has excluded subject removed, if you include an
    # explicit faulty_subjs (so you set this to false if you do; otherwise, set this to true if not faulty subjs input)

    summaries = summaries[:, invalid_vox == 0, :]

    baseline, dictionary_of_deltas, dictionary_of_covars = glm.continuous_glm(summaries, args.designmat, faulty_subjs)

    image_io.write_continuous_glm_results(baseline=baseline,
                                          dictionary_of_deltas=dictionary_of_deltas,
                                          dictionary_of_covars=dictionary_of_covars,
                                          summary_names=summary_names,
                                          mask=args.mask,
                                          invalid_vox=invalid_vox,
                                          glm_dir=args.output)

    # save subject names in order of design matrix to check

    np.savez_compressed(
        file=f"{args.output}/present_subjects",
        a=np.array(subj_names))

    """
    y_norm, dy_norm, sigma_n_norm = \
        summary_measures.normalise_summaries(data, summary_names, delta_data, sigma_n)

    image_io.write_glm_results(y_norm, dy_norm, sigma_n_norm, summary_names,
                               args.mask, invalid_vox, args.output + '_renormalised')"""

    print(f'Continuous GLM is done. Results are stored in {args.output}')


def continuous_glm_with_deconfounding_from_cli(args):
    """
    Runs glm on the summary meausres.
    :param args: output from argparse, should contain desing matrix anc contrast addresss, summary_dir and masks
    :return:
    """
    assert (args.designmat is not None)

    os.makedirs(args.output, exist_ok=True)

    ## Check to make sure the subject IDs for the design matrix match the subject IDs for summary measures

    subject_ids_from_dm = np.array(np.load(args.designmat)["b"])
    proposed_summary_files = np.array([args.summarydir + f'/{subj}.nii.gz' for subj in subject_ids_from_dm])

    not_in_sm = []
    for subj in subject_ids_from_dm:
        summary_file = args.summarydir + f'/{subj}.nii.gz'
        if not os.path.exists(summary_file):
            not_in_sm.append(subj)

    not_in_sm = np.array(not_in_sm)

    if len(not_in_sm) > 0:
        print("Some subjects not present. Remove from design matrix first")
        print(not_in_sm)

        np.savez_compressed(
            file=f"{args.output}/missing_subjects",
            a=not_in_sm)
        return

    ## Read summary measures and perform GLM

    summaries, invalid_vox, summary_names, subj_names = image_io.read_summary_images_from_predefined_list(
        summary_files=proposed_summary_files,
        mask=args.mask)

    summaries = glm.deconfounding_glm(summaries, args.confoundmat)

    summaries = summaries[:, invalid_vox == 0, :]

    baseline, dictionary_of_deltas, dictionary_of_covars = glm.continuous_glm(summaries, args.designmat)

    image_io.write_continuous_glm_results(baseline=baseline,
                                          dictionary_of_deltas=dictionary_of_deltas,
                                          dictionary_of_covars=dictionary_of_covars,
                                          summary_names=summary_names,
                                          mask=args.mask,
                                          invalid_vox=invalid_vox,
                                          glm_dir=args.output)

    # save subject names in order of design matrix to check

    np.savez_compressed(
        file=f"{args.output}/present_subjects",
        a=np.array(subj_names))


def inference_from_cli(args):
    data, delta_data, sigma_n, summary_names = image_io.read_glm_results(args.glmdir)
    # perform inference:
    ch_mdl = change_model.ChangeModel.load(args.model)

    posteriors, predictions, peaks, bad_samples = ch_mdl.infer(
        data, delta_data, sigma_n, integral_bound=float(args.integral_bound), parallel=False)

    # save the results:
    image_io.write_inference_results(args.output, ch_mdl.model_names, predictions, posteriors, peaks,
                                     f'{args.glmdir}/valid_mask')

    dv, offset, deviation = ch_mdl.estimate_quality_of_fit(
        data, delta_data, sigma_n, predictions, peaks)

    image_io.write_nifti(deviation[:, np.newaxis], f'{args.glmdir}/valid_mask',
                         f'{args.output}/sigma_deviations.nii.gz')
    print(f'Analysis completed successfully.')


def continuous_inference_from_cli(args):
    if args.mask is None:
        args.mask = f'{args.glmdir}/valid_mask'

    c_names = np.load(args.designmat)["c"]

    baseline, dictionary_of_deltas, dictionary_of_covars, summary_names = image_io.read_continuous_glm_results(
        glm_dir=args.glmdir,
        c_names=c_names,
        mask_add=args.mask)

    ch_mdl = change_model.ChangeModel.load(args.model)
    sm_names = ch_mdl.summary_names

    reg = np.eye(len(sm_names)) * 1e-5  # regularisation for low covariances

    # perform inference:

    for variable in c_names:

        savepath = f'{args.output}/{variable}'

        print(f"Inferring for {variable}")

        if not os.path.exists(savepath):
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
                                                        mask=args.mask,
                                                        variable=variable)
        else:

            print(f"...{variable} already inferred.")

            """
            dv, offset, deviation = ch_mdl.estimate_quality_of_fit(y1=baseline,
                                                                   dy=dictionary_of_deltas[variable],
                                                                   sigma_n=dictionary_of_covars[variable] + reg,
                                                                   predictions=predictions,
                                                                   amounts=peaks)
    
            image_io.write_nifti(deviation[:, np.newaxis], f'{args.glmdir}/valid_mask',
                                 f'{args.output}/{variable}/sigma_deviations.nii.gz')
            """

    print(f'Analysis completed successfully.')


def submit_invert(args):
    os.makedirs(args.output, exist_ok=True)
    pe_dir = f'{args.output}/pes/{args.model}'
    os.makedirs(pe_dir, exist_ok=True)

    py_file_path = os.path.realpath(__file__)
    task_list = list()
    for subj_idx, (x, d, bval, bvec) in enumerate(zip(args.xfm, args.data, args.bval, args.bvecs)):
        task_list.append(
            f'python {py_file_path} {subj_idx} {d} {x} {bvec} {bval} {args.mask} {args.model} {pe_dir}')

    # if 'SGE_ROOT' in os.environ.keys():
    #     print('Submitting jobs to SGE ...')
    #     with open(f'{args.output}/tasklist.txt', 'w') as f:
    #         for t in task_list:
    #             f.write("%s\n" % t)
    #         f.close()
    #
    #         job_id = run(f'fsl_sub -t {args.output}/tasklist.txt '
    #                      f'-T 240 -N bench_inversion -l {pe_dir}/log')
    #         print('jobs submitted to SGE ...')
    #         fslsub.hold(job_id)
    # else:
    #     os.system('; '.join(task_list))

    else:
        print('parameter estimates already exist in the specified path')

    # apply glm:
    if args.design_mat is not None:
        param_names = diffusion_models.prior_distributions[args.model].keys()
        fit_results, invalids = image_io.read_pes(pe_dir, args.mask)
        x = glm.loadDesignMat(args.design_mat)
        if not fit_results.shape[0] == x.shape[0]:
            raise ValueError(f'Design matrix with {x.shape[0]} subjects does not match with '
                             f'loaded parameters for {fit_results.shape[0]} subjects.')

        pe1 = fit_results[x[:, 0] == 1, :, :len(param_names)].mean(axis=0)
        pe2 = fit_results[x[:, 1] == 1, :, :len(param_names)].mean(axis=0)

        varpe1 = fit_results[x[:, 0] == 1, :, :len(param_names)].var(axis=0)
        varpe2 = fit_results[x[:, 1] == 1, :, :len(param_names)].var(axis=0)

        z_values = (pe2 - pe1) / np.sqrt(varpe1 / np.sqrt(x[:, 0].sum()) + varpe2 / np.sqrt(x[:, 1].sum()))
        p_values = st.norm.sf(abs(z_values)) * 2  # two-sided

        for d, p in zip(p_values, param_names):
            fname = f'{args.output}/zmaps/{p}'
            image_io.write_nifti(d, args.mask, fname=fname, invalids=invalids)
        print(f'Analysis completed sucessfully, the z-maps are stored at {args.output}')


def invert_from_cli(subj_idx, diff_add, xfm_add, bvec_add, bval_add, mask_add, mdl_name, output_add):
    print('diffusion data address:' + diff_add)
    print('xfm address:' + xfm_add)
    print('bvec address: ' + bvec_add)
    print('bval address: ' + bval_add)
    print('mask address: ' + mask_add)
    print('model name: ' + mdl_name)
    print('output path: ' + output_add)

    bvals = np.genfromtxt(bval_add)
    bvals = np.round(bvals / 1000, 1)
    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T

    def_field_dir = f"{output_add}/def_fields/"
    os.makedirs(def_field_dir, exist_ok=True)
    def_field = f"{def_field_dir}/{subj_idx}.nii.gz"
    data, valid_vox = summary_measures.sample_from_native_space(diff_add, xfm_add, mask_add, def_field)
    data = data / 1000
    params, stds = model_inversion.map_fit(data, 0.01, mdl_name, bvals, bvecs)
    print(f'subject {subj_idx} parameters estimated.')

    # write down [pes, vpes] to 4d files
    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    image_io.write_nifti(params, mask_add, fname, np.logical_not(valid_vox))
    print(f'Model fitted to subject {subj_idx} data.')


if __name__ == '__main__':
    print('This is bench user interface.')
