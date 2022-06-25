#!/usr/bin/env python3

import argparse
import glob
import os
from warnings import warn
import numpy as np
import scipy.stats as st
from fsl.data.featdesign import loadDesignMat
from scipy import optimize
from typing import Callable

import summary_measures
from bench import diffusion_models as dm, summary_measures, acquisition, change_model, image_io


def grad(f, p, bounds, dp=1e-6):
    dp = np.maximum(1e-10, abs(p * dp))
    g = []
    for i in range(len(p)):
        pi = np.zeros(len(p))
        pi[i] = dp[i]

        u = p + pi
        if u[i] > bounds[i][1]:
            u[i] = bounds[i][1]

        l = p - pi
        if l[i] < bounds[i][0]:
            l[i] = bounds[i][0]

        g.append((f(u) - f(l)) / (u[i] - l[i]))
    return np.stack(g, axis=0)


def hessian(f, p, bounds, dp=1e-6):
    g = lambda p_: grad(f, p_, bounds, dp)
    return grad(g, p, bounds, dp)


def log_prior(params, priors):
    return np.sum([priors[k].logpdf(params[k]) for k in params.keys() if hasattr(priors[k], 'logpdf')])


def log_likelihood_sig(params, model, signal, noise_std):
    expected = np.squeeze(model(params))
    d = expected.shape[0]
    expo = -0.5 * np.linalg.norm((signal - expected) / noise_std) ** 2
    nc = -0.5 * d * np.log((2 * np.pi * noise_std ** 2))
    return expo + nc


def log_posterior_sig(params, priors, model, signal, noise_std):
    param_dict = {k: v for k, v in zip(priors.keys(), params)}
    prior = log_prior(param_dict, priors)
    if np.isneginf(prior):
        ll = -np.inf
    else:
        ll = log_likelihood_sig(param_dict, model, signal, noise_std)

    return prior + ll


def map_fit_sig(model: Callable, priors: dict, signal: np.ndarray, noise_level: float):
    """
    inverts the model on the signal using maximum a posteriori estimation

    :param model: a callable that takes the params as input and produces signals as output
    :param priors: dictionary for prior distributions
    :param signal: measured signal (n_measurements, )
    :param noise_level: level of noise.
    :return:
      - estimated parameters.
      - std of the parameter estimations
    """
    x0 = np.array([getattr(v, 'mean', lambda: 0.5)() for v in priors.values()])
    bounds = [getattr(v, 'support', lambda: [0, 1])() for v in priors.values()]

    f = lambda x: -log_posterior_sig(x, priors, model, signal, noise_level)
    p = optimize.minimize(f, x0=x0, bounds=bounds, options={'disp': False}, method='Nelder-mead')
    h = hessian(f, p.x, bounds=bounds, dp=1e-2)
    std = np.sqrt(np.diag(abs(np.linalg.inv(h))))
    return p.x, std


def log_likelihood_smm(params, model, acq, sph_degree, y, noise_level):
    expected, sigma_n = model(acq, sph_degree, noise_level, **params)
    return change_model.log_mvnpdf(mean=np.squeeze(expected), cov=np.squeeze(sigma_n), x=np.squeeze(y))


def log_posterior_smm(params, priors, model, acq, sph_degree, y, sigma_n):
    param_dict = {k: v for k, v in zip(priors.keys(), params)}
    return -(log_likelihood_smm(param_dict, model, acq, sph_degree, y, sigma_n)
             + log_prior(param_dict, priors))


def map_fit_smm(model: Callable, acq: acquisition.Acquisition,
                shm_degree: int, priors: dict, y: np.ndarray, noise_level):
    x0 = np.array([v.mean() for v in priors.values()])
    bounds = [v.interval(1 - 1e-6) for v in priors.values()]
    p = optimize.minimize(log_posterior_smm,
                          args=(priors, model, acq, shm_degree, y, noise_level),
                          x0=x0, bounds=bounds, options={'disp': False})

    f = lambda p: log_posterior_smm(p, priors, model, acq, shm_degree, y, noise_level)
    h = hessian(f, p.x)
    std = 1 / np.sqrt(np.diag(h))

    return p.x, std


def map_fit(data, noise_cov, model, priors, x0=None):
    """
    Fits a model to a data using maximum a posteriori approach
    Important note: it requires each parameter to have a prior distribution chosen from
    scipy stats continuous rv class.
    :param data:
    :param model:
    :param priors:
    :param noise_cov:
    :param x0:
    :return:
    """
    if x0 is None:
        x0 = np.array([v.mean() for v in priors.values()])

    bounds = np.array([v.interval(1 - 1e-6) for v in priors.values()])

    def neg_log_posterior(params):
        params_dict = {k: v for k, v in zip(priors.keys(), params)}
        expected = model(**params_dict)
        llh = change_model.log_mvnpdf(mean=np.squeeze(expected),
                                      cov=np.squeeze(noise_cov),
                                      x=np.squeeze(data))
        lp = np.sum([priors[k].logpdf(params_dict[k])
                     for k in params_dict.keys() if hasattr(priors[k], 'logpdf')])
        return -np.asscalar(llh + lp)

    p = optimize.minimize(neg_log_posterior, x0=x0, bounds=bounds, method='Nelder-Mead')

    h = hessian(neg_log_posterior, p.x, bounds)
    std = 1 / np.sqrt(np.diag(h))

    return p.x, std


def infer_change(pe1, std_pe1, pe2, std_pe2, alpha=0.05):
    """
        infers the changed parameters given two parameter estimates

    :param pe1:
    :param std_pe1:
    :param pe2:
    :param std_pe2:
    :param alpha:
    :return:
    """
    zvals = (pe2 - pe1) / (std_pe1 + std_pe2)
    pvals = st.norm.sf(abs(zvals)) * 2  # twosided

    infered_change = np.argmax(zvals, axis=1)[:, np.newaxis]
    amount = np.take_along_axis(pe1 - pe2, infered_change, axis=1)
    idx = np.take_along_axis(pvals > alpha, infered_change, axis=1)
    infered_change += 1

    infered_change[idx] = 0
    amount[idx] = 0

    return infered_change, amount


def fit_model_smm(diffusion_sig, forward_model_name, bvals, bvecs, shm_degree):
    forward_model_name = forward_model_name.lower()
    available_models = list(dm.prior_distributions.keys())
    funcdict = {name: f for (name, f) in dm.__dict__.items() if name in available_models}

    if forward_model_name in available_models:
        priors = dm.prior_distributions[forward_model_name]
        func, names = summary_measures.bench_decorator(funcdict[forward_model_name])
    else:
        raise ValueError('Forward model is not available in the library.')

    idx_shells, shells = acquisition.ShellParameters.create_shells(bval=bvals)
    acq = acquisition.Acquisition(shells, idx_shells, bvecs)
    y, sigma_n = summary_measures.fit_shm(diffusion_sig, acq, shm_degree)
    pe = np.zeros((y.shape[0], len(priors)))
    for i in range(y.shape[0]):
        pe[i] = map_fit_smm(func, acq, shm_degree, priors, y[i], sigma_n[i])

    return pe


def pipeline(argv=None):
    args = inference_parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    pe_dir = f'{args.output}/pes/{args.model}'
    os.makedirs(pe_dir, exist_ok=True)

    if len(glob.glob(pe_dir + '/subj_*.nii.gz')) < len(args.data):
        py_file_path = os.path.realpath(__file__)
        task_list = list()
        for subj_idx, (x, d, bval, bvec) in enumerate(zip(args.xfm, args.data, args.bval, args.bvecs)):
            task_list.append(
                f'python {py_file_path} {subj_idx} {d} {x} {bvec} {bval} {args.mask} {args.model} {pe_dir}')
            # uncomment to debug:
            parse_args_and_fit(
                f'{py_file_path} {subj_idx} {d} {x} {bvec} {bval} {args.mask} {args.model} {pe_dir}'.split())

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
        fit_results, invalids = image_io.read_pes(pe_dir, args.mask)
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
            image_io.write_nifti(d, args.mask, fname=fname, invalids=invalids)
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

    def_field_dir = f"{output_add}/def_fields/"
    os.makedirs(def_field_dir, exist_ok=True)
    def_field = f"{def_field_dir}/{subj_idx}.nii.gz"
    data, valid_vox = summary_measures.sample_from_native_space(diff_add, xfm_add, mask_add, def_field)
    data = data / 1000
    params, stds = fit_model_sig(data, 0.01, mdl_name, bvals, bvecs)
    print(f'subject {subj_idx} parameters estimated.')

    # write down [pes, vpes] to 4d files
    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    image_io.write_nifti(params, mask_add, fname, np.logical_not(valid_vox))
    print(f'Model fitted to subject {subj_idx} data.')


