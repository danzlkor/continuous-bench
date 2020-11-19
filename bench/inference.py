"""
This module performs inference using change models
"""

import numpy as np
from typing import List
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from progressbar import ProgressBar
import warnings


def compute_posteriors(change_models: List, data, delta_data, sigma_n):
    """
    Computes the posterior probabilities for each model of change
    :param change_models: list containing change models.
    :param data: numpy array (n_vox, n_dim) containing the first group average
    :param delta_data: numpy array (n_vox, n_dim) containing the change between groups.
    :param sigma_n: numpy array or list (n_vox, n_dim, n_dim)
    :return: posterior probabilities for each voxel (n_vox, n_params)
    """

    print(f'running inference for {data.shape[0]} samples ...')
    lls = compute_log_likelihood(change_models, data, delta_data, sigma_n)
    priors = np.array([1] + [m.prior for m in change_models])  # the 1 is for empty set
    priors = priors / priors.sum()
    log_posteriors = lls + np.log(priors)
    posteriors = np.exp(log_posteriors)
    posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
    predictions = np.argmax(posteriors, axis=1)
    return posteriors, predictions


# warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_log_likelihood(models: List, y, delta_y, sigma_n):
    """
    Computes log_likelihood function for all models of change.
        :param y: (n_samples, n_x) array of summary measurements
        :param delta_y: (n_samples, n_x) array of delta data
        :param models: list of class ChangeVector containing parameters
        :param sigma_n: (n_samples, dim, dim) noise covariance per sample
    :return: np array containing log likelihood for each sample per class
    """

    n_samples, n_x = y.shape
    n_models = len(models) + 1
    log_prob = np.zeros((n_samples, n_models))
    pbar = ProgressBar()

    for sam_idx in pbar(range(n_samples)):
        y_s = y[sam_idx].T
        dy_s = delta_y[sam_idx]
        sigma_n_s = sigma_n[sam_idx]
        log_prob[sam_idx, 0] = log_mvnpdf(x=dy_s, mean=np.zeros(n_x), cov=sigma_n_s)

        for vec_idx, ch_mdl in enumerate(models, 1):
            try:
                mu, sigma_p = ch_mdl.predict(y_s)
                fun = lambda dv: np.exp(param_log_posterior(dv, dy_s, mu, sigma_p, sigma_n_s, ch_mdl.sigma_v))
                limits = find_range(dy_s, mu, sigma_p, sigma_n_s, ch_mdl.sigma_v)
                integral = quad(fun, *limits, epsrel=1e-3)[0]
                log_prob[sam_idx, vec_idx] = np.log(integral) if integral > 0 else -np.inf

            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    log_prob[sam_idx, vec_idx] = -1e3
                    warnings.warn(f'noise covariance was singular for sample {sam_idx}'
                                  f'with variances {np.diag(sigma_n_s)}')
                else:
                    raise

    return log_prob


def log_mvnpdf(x, mean, cov):
    """
    log of multivariate normal distribution
    :param x: input numpy array
    :param mean: mean of the distribution numpy array same size of x
    :param cov: covariance of distribution, numpy array
    :return scalar
    """
    if np.isscalar(cov):
        cov = np.array([[cov]])

    d = mean.shape[-1]
    e = -.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
    c = 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov.astype(float)))
    return np.log(c) + e


def param_log_posterior(dv, dd, mu, sigma_p, sigma_n, sigma_v):
    """
      Computes the log of function inside integral N(mu * dv , sigma_p * dv^2 + sigma_n) * N(0, sigma_v)
    """

    mean = np.squeeze(mu * dv)
    cov = np.squeeze(sigma_p) * dv ** 2 + sigma_n
    log_lh = log_mvnpdf(x=dd, mean=mean, cov=cov)
    log_prior = log_mvnpdf(x=dv, mean=np.zeros(1), cov=sigma_v ** 2)

    return log_lh + log_prior


def find_range(dd, mu, sigma_p, sigma_n, sigma_v, scale=1e2, search_range=100):
    """
      Computes the proper range for integration of the likelihood function
    """
    f = lambda dv: -param_log_posterior(dv, dd, mu, sigma_p, sigma_n, sigma_v)
    peak = minimize_scalar(f).x

    f2 = lambda dv: -f(dv) - (-f(peak) - np.log(scale))
    try:
        lower = root_scalar(f2, bracket=[-search_range + peak, peak], method='brentq').root
        upper = root_scalar(f2, bracket=[peak, search_range + peak], method='brentq').root

    except Exception as e:
        print(f"Error of type {e.__class__} occurred, while finding limits of integral."
              f"The peak +/-{search_range} is used instead")
        lower = -search_range + peak
        upper = search_range + peak

    return lower, upper


def performance_measures(posteriors, true_change, set_names):
    """
    Computes accuracy and avg posterior for true changes
    """
    posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
    predictions = np.argmax(posteriors, axis=1)

    indices = {str(s): i for i, s in enumerate(set_names)}
    true_change_idx = [indices[t] for t in true_change]

    accuracy = (predictions == true_change_idx).mean()
    true_posteriors = np.nanmean(posteriors[np.arange(len(true_change_idx)), true_change_idx])

    return accuracy, true_posteriors
