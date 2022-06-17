#!/usr/bin/env python3
"""
This module is to parse inputs from commandline and call the proper functions from other modules.
"""

import argparse
import os
from file_tree import FileTree
import numpy as np
from bench import change_model, glm, summary_measures, diffusion_models, acquisition, image_io
from fsl.utils.fslsub import submit


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

    # train arguments:
    train_required = diff_train_parser.add_argument_group("required arguments")
    available_models = list(diffusion_models.prior_distributions.keys())
    train_required.add_argument(
        "--model", help=f"name of the forward model. Available models:\n{available_models}", required=True)
    train_required.add_argument("--output", help="name of the trained model", required=True)
    train_required.add_argument("--bvals", help="b-values for training", required=True)

    train_optional = diff_train_parser.add_argument_group("optional arguments")
    train_optional.add_argument("-n", default=10000, type=int, help="number of training samples (default=10000)",
                                required=False)
    train_optional.add_argument("-b0-thresh", default=1, type=float,
                                help="threshold for b0 (default=1)")
    train_optional.add_argument("-p", default=2, type=int, help="polynomial degree for mean (default=2)", required=False)
    train_optional.add_argument("-ps", default=1, type=int, help="polynomial degree for variance (default=1)",
                                required=False)
    train_optional.add_argument("-d", default=2, type=int,
                                help=" maximum degree for summary measures (must be even numbers, default=2)",
                                required=False)
    train_optional.add_argument("--alpha", default=0.0, type=float,
                                help="regularisation weight for training regression models(default=0)", required=False)
    train_optional.add_argument("--change-vecs", help="text file for defining vectors of change (refer to documentations)", default=None, required=False)
    train_optional.add_argument("--summarytype", default='shm', type=str,
                                help='type of summary measurements. Either shm (spherical harmonic model)'
                                     ' or dtm (diffusion tensor model) (default shm)', required=False)
    train_optional.add_argument("--bvecs", help="gradient directions", required=False)

    train_optional.add_argument('--verbose', help='flag for printing optimisation outputs', dest='verbose',
                                action='store_true', default=False)
    diff_train_parser.set_defaults(func=train_from_cli)

    # fit summary arguments:
    submit_summary_parser.add_argument("--file-tree", help="file-tree text file", required=True)
    submit_summary_parser.add_argument("--mask", help="mask in standard space.", required=True)
    submit_summary_parser.add_argument("--summarytype", default='shm', type=str,
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
    diff_summary_parser.add_argument('--summarytype', default='shm', required=False)
    diff_summary_parser.add_argument('--b0-thresh', default=1, type=float, required=False)
    diff_summary_parser.add_argument('--normalise', dest='normalise', action='store_true')

    diff_summary_parser.set_defaults(func=summary_from_cli)

    # normalization args
    diff_normalise_parser.add_argument('--study-dir', default=None,
                                      help='Path to the un-normalised summary measurements', required=True)

    # glm arguments:
    glm_parser.add_argument("--summarydir", help='Path to summary measurements folder',required=True)
    glm_parser.add_argument("--mask", help='Path to the mask', required=True)
    glm_parser.add_argument("--designmat", help="Design matrix for the group glm", required=False)
    glm_parser.add_argument("--designcon", help="Design contrast for the group glm", required=False)
    glm_parser.add_argument('--paired', dest='paired', action='store_true')
    glm_parser.add_argument("--output", help='Output directory', required=True)
    glm_parser.set_defaults(func=glm_from_cli)

    # inference arguments:
    inference_parser.add_argument("--model", help="Path to a trained model of change (output of bench diff-train)",
                                  default=None, required=True)
    inference_parser.add_argument('--glmdir', help="Path to the glm dir", required=True)
    inference_parser.add_argument('--output', help="Path to the output dir", required=True)
    inference_parser.add_argument("--mask", help='Path to the mask (if none passed uses the valid_mask in glm folder)', default=None, required=False)
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

    func, summary_names = diffusion_models.bench_decorator(
        model=forward_model, bval=bvals, bvec=bvecs, summary_type=args.summarytype, shm_degree=int(args.d))
    print('The model is trained using summary measurements:', summary_names)
    trainer = change_model.Trainer(
        forward_model=func, kwargs={'noise_level': 0.}, change_vecs=args.change_vecs,
        summary_names=summary_names, param_prior_dists=param_dist)

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
        def_field = f"tmp.nii.gz"
        data, valid_vox = image_io.sample_from_native_space(args.data, args.xfm, args.mask, def_field)

    acq = acquisition.Acquisition.from_bval_bvec(args.bvals, args.bvecs, args.b0_thresh)
    summaries = summary_measures.fit_shm(data, acq, shm_degree=args.shm_degree)
    names = summary_measures.summary_names(acq, args.summarytype, args.shm_degree)

    if args.normalise:
        print('Summary measures are normalised by b0_mean.')
        summaries = summary_measures.normalise_summaries(summaries, names)
        names = [f'{n}/b0' for n in names[1:]]

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


if __name__ == '__main__':
    print('This is bench user interface.')
