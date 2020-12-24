from bench import diffusion_models as dm, summary_measures, user_interface, acquisition
from scipy.optimize import curve_fit
import numpy as np
from progressbar import ProgressBar
import os
from nibabel import Nifti1Image
from fsl.data.image import Image
import glob
import fsl.utils.fslsub as fslsub
from fsl.utils.run import run
import sys
from fsl.data.featdesign import loadDesignMat
from warnings import warn
import scipy.stats as st

sph_degree = 4


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

    acq = acquisition.ShellParameters.create_shells(bval=bval)
    sm = summary_measures.compute_summary(signal, acq, sph_degree=sph_degree)
    sm = np.stack(list(sm.values()))
    return sm


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
bingham_noddi_param_bounds = np.array([[0, 0, 0, 0], [1.5, 1.5, 1.5, 1]])
func_dict = {'ball_stick': (ball_stick_func, ball_stick_param_bounds),
             'watson_noddi': (watson_noddi_func, watson_noddi_param_bounds),
             'bingham_noddi': (bingham_noddi_func, bingham_noddi_param_bounds)
             }


def invert(diffusion_sig, forward_model_name, bvals, bvecs):
    grads = np.hstack([bvals[:, np.newaxis], bvecs])
    pbar = ProgressBar()

    if forward_model_name in list(func_dict.keys()):
        func, param_bounds = func_dict[forward_model_name]
    else:
        raise ValueError('Forward model is not available in the library.')

    fails = []
    pe = np.zeros((diffusion_sig.shape[0], len(param_bounds[0])))
    vpe = np.zeros((diffusion_sig.shape[0], len(param_bounds[0])))

    for i in pbar(range(diffusion_sig.shape[0])):
        try:
            pe[i], tmp = curve_fit(func, grads, diffusion_sig[i], bounds=param_bounds, p0=0.5 * param_bounds[1])
            vpe[i] = np.diagonal(tmp)
        except (RuntimeError, ValueError):
            print(f'Optimal paramaters could not be estimated for sample {i}')
            pe[i] = 0.5 * param_bounds[1]
            fails.append(i)
    if len(fails) > 0:
        print(f'{len(fails)} samples failed, middle of range returned instead.')
    return pe, vpe


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


def from_command_line(argv=None):
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

        z_values = (pe2 - pe1) / np.sqrt(varpe1 / np.sqrt(x[:, 0].sum()) + varpe2 / np.sqrt(x[:, 1].sum()))
        p_values = st.norm.sf(abs(z_values)) * 2  # two-sided

        # _, delta, sigma = group_glm(pes, args.design_mat, args.design_con)
        # zvals = delta / np.sqrt(np.diagonal(sigma, axis1=1, axis2=2))
        for d, p in zip(p_values, param_names):
            fname = f'{args.output}/zmaps/{p}'
            user_interface.write_nifti(d, args.mask, fname=fname, invalids=invalids)
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
    bvals = np.round(bvals / 1000, 1)
    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T

    def_field = f"{output_add}/def_field_{subj_idx}.nii.gz"
    data, valid_vox = summary_measures.sample_from_native_space(diff_add, xfm_add, mask_add, def_field)

    mean_b0 = data[:, bvals == 0].mean()
    diffusion_vox_normalized = data / mean_b0

    params, var_params = invert(diffusion_vox_normalized, mdl_name, bvals, bvecs)
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
