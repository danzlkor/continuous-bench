from bench import diffusion_models as dm, summary_measures, user_interface, acquisition, change_model
import numpy as np
import os
from scipy import optimize
from nibabel import Nifti1Image
from fsl.data.image import Image
import glob
import fsl.utils.fslsub as fslsub
from fsl.utils.run import run
import sys
from fsl.data.featdesign import loadDesignMat
from warnings import warn
import scipy.stats as st
from typing import Union, Callable, List


def ball_stick_func(grad, s_iso, s_a, d_iso, d_a):
    signal = np.zeros(grad.shape[0])
    for i in range(len(grad)):
        bval = grad[i, 0]
        bvec = grad[i, 1:]
        signal[i] = dm.ball_stick(bval=bval, bvec=bvec[np.newaxis, :], s_iso=s_iso, s_a=s_a,
                                  d_iso=d_iso, d_a=d_a)
    return signal


def watson_noddi_func(grad, s_iso, s_int, s_ext, odi):
    bval = grad[:, 0]
    bvec = grad[:, 1:]

    # fixed parameters:
    d_iso = 3
    dax_int = 1.7
    dax_ext = 1.7
    tortuosity = s_int / (s_int + s_ext)

    signal = dm.watson_noddi(bval=bval, bvec=bvec,
                             s_iso=s_iso, s_in=s_int, s_ex=s_ext,
                             d_iso=d_iso, d_a_in=dax_int, d_a_ex=dax_ext,
                             tortuosity=tortuosity, odi=odi)
    return signal


def bingham_noddi_func(grad, s_iso, s_int, s_ext, odi, odi_ratio):
    bval = grad[:, 0]
    bvec = grad[:, 1:]

    # fixed parameters:
    d_iso = 3
    dax_int = 1.7
    dax_ext = 1.7
    tortuosity = s_int / (s_int + s_ext)

    signal = dm.bingham_noddi(bval=bval, bvec=bvec,
                             s_iso=s_iso, s_in=s_int, s_ex=s_ext,
                             d_iso=d_iso, d_a_in=dax_int, d_a_ex=dax_ext,
                             tortuosity=tortuosity, odi=odi, odi_ratio=odi_ratio)
    return signal


ball_stick_param_bounds = np.array([[0, 0, 0, 0], [1.5, 1.5, 4, 3]])
watson_noddi_param_bounds = np.array([[0, 0, 0, 0, 0], [1.5, 1.5, 1.5, 1, 1]])
bingham_noddi_param_bounds = np.array([[0, 0, 0, 0],[1.5, 1.5, 1.5, 1]])
func_dict = {'ball_stick': (ball_stick_func, ball_stick_param_bounds),
             'watson_noddi': (watson_noddi_func, watson_noddi_param_bounds),
             'bingham_noddi': (bingham_noddi_func, bingham_noddi_param_bounds)
             }


def log_likelihood(params, x, y, sigma_n, func):
    return change_model.log_mvnpdf(mean=np.squeeze(func(x, *params)), cov=sigma_n, x=np.squeeze(y))


def log_prior(params, priors):
    return np.sum([np.log(f.pdf(v))
                   for v, f in zip(params, priors.values())])


def log_posterior(params, x, y, sigma_n, func, priors):
    return -(log_likelihood(params, x, y, sigma_n, func) + log_prior(params, priors))


def map_fit(forward_model: Callable, acq, priors: dict, y: np.ndarray, sigma_n: Union[np.ndarray, List]):
    p = optimize.minimize(log_posterior, args=(acq, y, sigma_n, forward_model, priors),
                          x0=np.array([0, 0, 0]), options={'disp': True})

    return p.x


def estimate_shms(signal, acq, sph_degree):

    sum_meas = list()
    residuals = list()
    noise_level = 1
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        lmax = this_shell.lmax
        bvecs = acq.bvecs[dir_idx]
        shell_signal = signal[..., dir_idx]

        _, phi, theta = summary_measures.cart2spherical(*bvecs.T)
        y, m, l = summary_measures.real_sym_sh_basis(lmax, theta, phi)
        coeffs = shell_signal.dot(np.linalg.pinv(y.T))
        shell_err = shell_signal - coeffs @ y.T
        residuals.append(shell_err)

        smm = [coeffs[..., l == 0].mean(axis=-1)]
        _, phi, theta = summary_measures.cart2spherical(*bvecs.T)
        sh_mat, m, l = summary_measures.real_sym_sh_basis(lmax, theta, phi)
        c = np.linalg.pinv(sh_mat).T
        j = c[..., l == 0].mean(axis=-1)
        snn = [j.T.dot(j) * (noise_level ** 2)]

        if lmax > 0:
            for degree in np.arange(2, sph_degree + 1, 2):
                smm.append(np.power(coeffs[..., l == degree], 2).mean(axis=-1))

                ng = bvecs.shape[0]
                f = 4 * np.pi * (noise_level ** 2) / ng
                snn.append(smm * 4 * f / (2 * l + 1) + 2 * (f ** 2) / (2 * l + 1))

        sum_meas += smm

    noise_level = np.concatenate(residuals, axis=-1).std(axis=-1)
    sigma_n = summary_measures.noise_propagation(sum_meas, acq, sph_degree, noise_level)
    return sum_meas, sigma_n


def invert(diffusion_sig, forward_model_name, bvals, bvecs, sph_degree):

    if forward_model_name in list(func_dict.keys()):
        func, priors = func_dict[forward_model_name]
    else:
        raise ValueError('Forward model is not available in the library.')
    idx_shells, shells = acquisition.ShellParameters.create_shells(bval=bvals)
    acq = acquisition.Acquisition(shells, idx_shells, bvecs)

    n_vox = diffusion_sig.shape[0]
    for i in range(n_vox):
        y, sigma_n = estimate_shms(diffusion_sig, acq, sph_degree)
        pe = map_fit(func, priors, y, sigma_n)

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
    pe_dir = f'{args.output}/pes'
    os.makedirs(pe_dir, exist_ok=True)

    if len(glob.glob(pe_dir + '/subj_*.nii.gz')) < len(args.data):
        # make a bval file
        py_file_path = os.path.realpath(__file__)
        task_list = list()
        for subj_idx, (x, d, bv) in enumerate(zip(args.xfm, args.data, args.bvecs)):
            task_list.append(
                f'python3 {py_file_path} {subj_idx} {d} {x} {bv} {args.bval} {args.mask} {args.model} {pe_dir}')
            parse_args_and_fit(
                f'{py_file_path} {subj_idx} {d} {x} {bv} {args.bval} {args.mask} {args.model} {pe_dir}'.split())
        if 'SGE_ROOT' in os.environ.keys():
            print('Submitting jobs to SGE ...')
            with open(f'{args.output}/tasklist.txt', 'w') as f:
                for t in task_list:
                    f.write("%s\n" % t)
                f.close()

                job_id = run(f'fsl_sub -t {args.output}/tasklist.txt '
                             f'-q short.q -N bench_inversion -l {pe_dir}/log -s openmp,2')
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

    def_field = f"{output_add}/def_field_{subj_idx}.nii.gz"
    data, valid_vox = summary_measures.sample_from_native_space(diff_add, xfm_add, mask_add, def_field)

    params, var_params = invert(data, mdl_name, bvals, bvecs, sph_degree=4)
    print(f'subject {subj_idx} parameters estimated.')

    # write down [pes, vpes] to 4d files
    mask_img = Image(mask_add)
    mat = np.zeros((*mask_img.shape, params.shape[1] * 2)) + np.nan
    std_indices = np.array(np.where(mask_img.data > 0)).T
    std_indices_valid = std_indices[valid_vox]

    mat[tuple(std_indices_valid.T)] = np.concatenate((params, var_params), axis=-1)
    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    Nifti1Image(mat, mask_img.nibImage.affine).to_filename(fname)
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
