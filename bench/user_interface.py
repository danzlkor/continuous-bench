"""
This module is to parse inputs from commandline and call the proper functions from other modules.
"""
import argparse
import os
from warnings import warn
from bench import inference, train_change_model, glm, summary_measurements


def from_command_line(argv=None):
    """
    Wrapper function to parse input and run main.
    :param argv: string from command line containing all required inputs 
    :return: saves the output images to the specified path
    """
    
    args = parse_args(argv)
    main(args)


def main(args):
    """
    Main function that calls all the required steps.
    :param args: output namespace from argparse from commandline input  
    :return: runs the process and save images to the specified path
    :raises: when the input files are not available
    """
    
    change_model = train_change_model.ChangeModel.load(args.model)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # if summaries are not provided fit summaries:
    if args.sm_dir is None:
        args.sm_dir = f'{args.output}/SummaryMeasures'
        summary_measurements.fit_summary(diff=args.data, bvecs=args.bvecs,
                                         bvals=args.bval, xfms=args.xfm, output=args.sm_dir)

    summaries, invalid_vox = summary_measurements.read_summaries(path=args.summary_dir)

    # perform glm:
    data, delta_data, sigma_n = glm.group_glm(summaries, args.design_mat, args.design_con)

    # perform inference:
    sets, posteriors, _ = inference.compute_posteriors(change_model, args.prior_change, data, delta_data, sigma_n)

    # save the results:
    maps_dir = f'{args.output}/PosteriorMaps/{change_model.model_name}'
    inference.write_nifti(sets, posteriors, args.mask, maps_dir, invalid_vox)
    print(f'Analysis completed successfully, the posterior probability maps are stored in {maps_dir}')


def parse_args(argv):
    """
    Parses the commandline input anc checks for the consistency of inputs
    :param argv: input string from commandline
    :return: arg namespce from argparse
    :raises: if the number of provided files do not match with other arguments
    """

    parser = argparse.ArgumentParser("BENCH: Bayesian EstimatioN of CHange")

    required = parser.add_argument_group("required arguments")
    required.add_argument("--mask", help="Mask in standard space indicating which voxels to analyse", required=True)
    required.add_argument("--design-mat", help="Design matrix for the group glm", required=True)
    required.add_argument("--design-con", help="Design contrast for the group glm", required=True)
    required.add_argument("--model", help="Forward model, either name of a standard model or full path to "
                                          "a trained model json file", required=True)
    required.add_argument("--output", help="Path to the output directory", required=True)

    required.add_argument('--summary-dir', help='Path to the pre-computed summary measurements', required=False)
    required.add_argument("--data", nargs='+', help="List of dMRI data in subject native space", required=True)
    required.add_argument("--xfm", help="Non-linear warp fields mapping subject diffusion space to the mask space",
                          nargs='+', metavar='xfm.nii', required=True)
    required.add_argument("--bvecs", nargs='+', metavar='bvec', required=True,
                          help="Gradient orientations for each subject")

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("--sigma-v", default=0.1,
                          help="Standard deviation for prior change in parameters (default = 0.1)", required=False)

    summary_measurements.ShellParameters.add_to_parser(parser)

    args = parser.parse_args(argv)

    # fix the problem of getting a single arg for list arguments:
    if len(args.xfm) == 1:
        args.xfm = args.xfm[0].split()
    if len(args.data) == 1:
        args.data = args.data[0].split()
    if len(args.bvecs) == 1:
        args.bvecs = args.bvecs[0].split()

    for subj_idx, (nl, d, bv) in enumerate(zip(args.xfm, args.data, args.bvecs), 1):
        print(f'Scan {subj_idx}: dMRI ({d} with {bv}); transform ({nl})')
    print('')

    n_subjects = min(len(args.xfm), len(args.data), len(args.bvecs))
    if len(args.data) > n_subjects:
        raise ValueError(f"Got more diffusion MRI dataset than transformations/bvecs: {args.data[n_subjects:]}")
    if len(args.xfm) > n_subjects:
        raise ValueError(f"Got more transformations than diffusion MRI data/bvecs: {args.xfm[n_subjects:]}")
    if len(args.bvecs) > n_subjects:
        raise ValueError(f"Got more bvecs than diffusion MRI data/transformations: {args.bvecs[n_subjects:]}")
    if os.path.isdir(args.output):
        warn('Output directory already exists, contents might be overwritten.')
    else:
        os.makedirs(args.output, exist_ok=True)

    return args
