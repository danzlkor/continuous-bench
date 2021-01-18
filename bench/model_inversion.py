from bench import diffusion_models as dm, summary_measures, user_interface, acquisition, change_model
import numpy as np
import os
from scipy import optimize
from fsl.data.image import Image
import glob
import fsl.utils.fslsub as fslsub
from fsl.utils.run import run
import sys
from fsl.data.featdesign import loadDesignMat
from warnings import warn
import scipy.stats as st
from typing import Union, Callable, List
from progressbar import ProgressBar


def log_likelihood(params, model, acq, sph_degree, y, sigma_n):
    expected = np.squeeze(model(acq, sph_degree, 0, **params))
    return change_model.log_mvnpdf(mean=expected, cov=sigma_n, x=np.squeeze(y))


def log_prior(params, priors):
    return np.sum([np.log(priors[k].pdf(params[k]))
                   for k in params.keys()])


def log_posterior(params, priors, model, acq, sph_degree, y, sigma_n):
    param_dict = {k: v for k, v in zip(priors.keys(), params)}
    return -(log_likelihood(param_dict, model, acq, sph_degree, y, sigma_n)
             + log_prior(param_dict, priors))


def map_fit(model: Callable, acq: acquisition.Acquisition, sph_degree: int,
            priors: dict, y: np.ndarray, sigma_n: Union[np.ndarray, List]):
    x0 = np.array([v.mean() for v in priors.values()])
    bounds = [v.interval(1 - 1e-6) for v in priors.values()]
    p = optimize.minimize(log_posterior,
                          args=(priors, model, acq, sph_degree, y, sigma_n),
                          x0=x0,  bounds=bounds, options={'disp': False})

    return p.x


def fit_model(diffusion_sig, forward_model_name, bvals, bvecs, sph_degree):
    forward_model_name = forward_model_name.lower()
    available_models = list(dm.prior_distributions.keys())
    funcdict = {name: f for (name, f) in dm.__dict__.items() if name in available_models}

    if forward_model_name in available_models:
        priors = dm.prior_distributions[forward_model_name]
        func = dm.bench_decorator(funcdict[forward_model_name])
    else:
        raise ValueError('Forward model is not available in the library.')

    idx_shells, shells = acquisition.ShellParameters.create_shells(bval=bvals)
    acq = acquisition.Acquisition(shells, idx_shells, bvecs)
    y, sigma_n = summary_measures.fit_shm(diffusion_sig, acq, sph_degree)
    pbar = ProgressBar()
    for i in pbar(range(y.shape[0])):
        pe = map_fit(func, acq, sph_degree, priors, y[i], sigma_n[i])

    return pe


def read_pes(pe_dir, mask_add):
    mask_img = Image(mask_add)
    n_subj = len(glob.glob(pe_dir + '/subj_*.nii.gz'))
    pes = list()
    for subj_idx in range(n_subj):
        f = f'{pe_dir}/subj_{subj_idx}.nii.gz'
        pes.append(Image(f).data[mask_img.data > 0, :])

    print(f'loaded summaries from {n_subj} subjects')
    pes = np.array(pes)
    invalids = np.any(np.isnan(pes), axis=(0, 2))
    pes = pes[:, invalids == 0, :]
    if invalids.sum() > 0:
        warn(f'{invalids.sum()} voxels are dropped because of lying outside of brain mask in some subjects.')
    return pes, invalids


def pipeline(argv=None):
    args = user_interface.inference_parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    pe_dir = f'{args.output}/pes/{args.model}'
    os.makedirs(pe_dir, exist_ok=True)

    if len(glob.glob(pe_dir + '/subj_*.nii.gz')) < len(args.data):
        py_file_path = os.path.realpath(__file__)
        task_list = list()
        for subj_idx, (x, d, bv) in enumerate(zip(args.xfm, args.data, args.bvecs)):
            task_list.append(
                f'python3 {py_file_path} {subj_idx} {d} {x} {bv} {args.bval} {args.mask} {args.model} {pe_dir}')
        # uncomment to debug:
        # parse_args_and_fit(f'{py_file_path} {subj_idx} {d} {x} {bv} {args.bval} {args.mask} {args.model}
        # {pe_dir}'.split())

        if 'SGE_ROOT' in os.environ.keys():
            print('Submitting jobs to SGE ...')
            with open(f'{args.output}/tasklist.txt', 'w') as f:
                for t in task_list:
                    f.write("%s\n" % t)
                f.close()

                job_id = run(f'fsl_sub -t {args.output}/tasklist.txt '
                             f'-q short.q -N bench_inversion -l {pe_dir}/log -s openmp,2')
                print('jobs submitted to SGE ...')
                fslsub.hold(job_id)
        else:
            os.system('; '.join(task_list))

        if len(glob.glob(pe_dir + '/subj_*.nii.gz')) == len(args.data):
            print(f'model fitted to {len(args.data)} subjects.')
        else:
            n_fails = len(args.data) - len(glob.glob(pe_dir + '/subj_*.nii.gz'))
            raise ValueError(f'model fitting failed for {n_fails} subjects.')
    else:
        print('parameter estimates already exist in the specified path')

    # apply glm:
    if args.design_mat is not None:
        param_names = dm.prior_distributions[args.model].keys()
        fit_results, invalids = read_pes(pe_dir, args.mask)
        x = loadDesignMat(args.design_mat)
        if not fit_results.shape[0] == x.shape[0]:
            raise ValueError(f'Design matrix with {x.shape[0]} subjects does not match with '
                             f'loaded parameters for {fit_results.shape[0]} subjects.')

        pe1 = fit_results[x[:, 0] == 1, :, :len(param_names)].mean(axis=0)
        pe2 = fit_results[x[:, 1] == 1, :, :len(param_names)].mean(axis=0)

        varpe1 = fit_results[x[:, 0] == 1, :, :len(param_names)].var(axis=0)
        varpe2 = fit_results[x[:, 1] == 1, :, :len(param_names)].var(axis=0)

        z_values = (pe2 - pe1) / np.sqrt(varpe1/np.sqrt(x[:, 0].sum()) + varpe2 / np.sqrt(x[:, 1].sum()))
        p_values = st.norm.sf(abs(z_values)) * 2  # two-sided

        # _, delta, sigma = group_glm(pes, args.design_mat, args.design_con)
        # zvals = delta / np.sqrt(np.diagonal(sigma, axis1=1, axis2=2))
        for d, p in zip(p_values, param_names):
            fname = f'{args.output}/zmaps/{p}'
            user_interface.write_nifti(d, args.mask, fname=fname , invalids=invalids)
        print(f'Analysis completed sucessfully, the z-maps are stored at {args.output}')



def single_sub_fit(subj_idx, diff_add, xfm_add, bvec_add, bval_add, mask_add, mdl_name, output_add):
    print('diffusion data address:' + diff_add)
    print('xfm address:' + xfm_add)
    print('bvec address: ' + bvec_add)
    print('bval address: ' + bval_add)
    print('mask address: ' + mask_add)
    print('model name: ' + mdl_name)
    print('output path: ' + output_add)

    bvals = np.genfromtxt(bval_add)
    bvals = np.round(bvals/1000, 1)
    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T

    def_field_dir = f"{output_add}/def_fields/"
    os.makedirs(def_field_dir, exist_ok=True)
    def_field = f"{def_field_dir}/{subj_idx}.nii.gz"
    data, valid_vox = summary_measures.sample_from_native_space(diff_add, xfm_add, mask_add, def_field)

    params = fit_model(data, mdl_name, bvals, bvecs, sph_degree=4)
    print(f'subject {subj_idx} parameters estimated.')

    # write down [pes, vpes] to 4d files
    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    user_interface.write_nifti(params, mask_add, fname, np.logical_not(valid_vox))
    print(' Model fitted to subject')


def parse_args_and_fit(args):
    args.pop(0)  # python file name
    subj_idx_ = args[0]
    diff_add_ = args[1]
    xfm_add_ = args[2]
    bvec_add_ = args[3]
    bval_add_ = args[4]
    mask_add_ = args[5]
    mdl_add_ = args[6]
    output_add_ = args[7]
    single_sub_fit(subj_idx_, diff_add_, xfm_add_, bvec_add_, bval_add_, mask_add_, mdl_add_, output_add_)


if __name__ == '__main__':
    parse_args_and_fit(sys.argv)
