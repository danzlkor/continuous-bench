"""
This module contains classes and functions to train a change model for a given function.
"""

import numpy as np
import pickle
from scipy.spatial import KDTree
from scipy.stats import rv_continuous
from typing import Callable, List, Any, Mapping, Union, Sequence
from progressbar import ProgressBar
from dataclasses import dataclass
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline
import os
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
import warnings


@dataclass
class ChangeVector:
    """
    class for a single model of change
    """
    vec: np.ndarray
    mu_mdl: Pipeline
    l_mdl: Pipeline
    sigma_v: float = 0.1
    prior: float = 1
    name: str = 'unnamed'

    def predict(self, y: np.ndarray):
        """
            Computes predicted mu and sigma at the given point for this model of change
              :param y: sample data
              :return: Tuple with: mu and sigma as dictionaries
              """
        if y.ndim == 1:
            y = y[np.newaxis, :]

        mu = self.mu_mdl.predict(y)
        sigma = l_to_sigma(self.l_mdl.predict(y))

        return mu, sigma


@dataclass
class ChangeModel:
    models: List[ChangeVector]
    name: str = 'unnamed'

    def save(self, path='./', file_name=None):
        if file_name is None:
            file_name = getattr(self.name)

        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)

    def compute_posteriors(self, data, delta_data, sigma_n):
        """
        Computes the posterior probabilities for each model of change
        :param change_models: list containing change models.
        :param data: numpy array (n_vox, n_dim) containing the first group average
        :param delta_data: numpy array (n_vox, n_dim) containing the change between groups.
        :param sigma_n: numpy array or list (n_vox, n_dim, n_dim)
        :return: posterior probabilities for each voxel (n_vox, n_params)
        """

        print(f'running inference for {data.shape[0]} samples ...')
        lls = self.compute_log_likelihood(data, delta_data, sigma_n)
        priors = np.array([1] + [m.prior for m in self.models])  # the 1 is for empty set
        priors = priors / priors.sum()
        log_posteriors = lls + np.log(priors)
        posteriors = np.exp(log_posteriors)
        posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
        predictions = np.argmax(posteriors, axis=1)
        return posteriors, predictions

    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    def compute_log_likelihood(self, y, delta_y, sigma_n):
        """
        Computes log_likelihood function for all models of change.
            :param y: (n_samples, n_x) array of summary measurements
            :param delta_y: (n_samples, n_x) array of delta data
            :param models: list of class ChangeVector containing parameters
            :param sigma_n: (n_samples, dim, dim) noise covariance per sample
        :return: np array containing log likelihood for each sample per class
        """

        n_samples, n_x = y.shape
        n_models = len(self.models) + 1
        log_prob = np.zeros((n_samples, n_models))
        pbar = ProgressBar()

        for sam_idx in pbar(range(n_samples)):
            y_s = y[sam_idx].T
            dy_s = delta_y[sam_idx]
            sigma_n_s = sigma_n[sam_idx]
            log_prob[sam_idx, 0] = log_mvnpdf(x=dy_s, mean=np.zeros(n_x), cov=sigma_n_s)

            for vec_idx, ch_mdl in enumerate(self.models, 1):
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

    @classmethod
    def load(cls, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                mdl = pickle.load(f)
                return mdl
        else:
            raise SystemExit('model file not found')


class MyPipeline(Pipeline):
    degree: int
    alpha: float

    def __init__(self, degree, alpha):
        steps = [('features', PolynomialFeatures(degree=degree)),
                 ('reg', linear_model.Ridge(alpha=alpha, fit_intercept=False))]
        Pipeline.__init__(self, steps=steps)


@dataclass
class Trainer:
    """
    A class for training models of change
    """
    forward_model: Callable
    x: Any
    param_prior_dists: Mapping[str, rv_continuous]
    vecs: np.ndarray = None
    sigma_v: Union[float, List[float], np.ndarray] = 0.1
    priors: Union[float, Sequence] = 1

    def __post_init__(self):
        if self.vecs is None:
            self.vecs = np.eye(len(self.param_prior_dists))

        if self.vecs.shape[1] != len(self.param_prior_dists):
            raise (f'Length of change vectors must be equal to the number of parameters. '
                   f'{len(self.param_prior_dists)} expected but {self.vecs.shape[1]} is given.')
        if np.isscalar(self.sigma_v):
            self.sigma_v = [self.sigma_v] * len(self.vecs)

        if len(self.sigma_v) != self.vecs.shape[0]:
            raise ("sigma_v must be either a scalar (same for all change models) "
                   "or a sequence with size of number of change models.")

        if np.isscalar(self.priors):
            self.priors = [self.priors] * len(self.vecs)

        if len(self.priors) != self.vecs.shape[0]:
            raise ("priors must be either a scalar (same for all change models) "
                   "or a sequence with size of number of change models.")

        self.param_names = list(self.param_prior_dists.keys())

    def train(self, n_samples=10000, poly_degree=2, regularization=1, k=100, dv0=1e-6):
        """
          Train change models (estimates w_mu and w_l) using forward model simulation
            :param n_samples: number simulated samples to estimate forward model
            :param k: nearest neighbours parameter
            :param dv0: the amount perturbation in parameter to estimate derivatives
            :param poly_degree: polynomial degree for regression
            :param regularization: ridge regression alpha
          """
        models = []
        for vec, sigma_v, prior in zip(self.vecs, self.sigma_v, self.priors):
            name = ' + '.join([f'{v:1.1f}{p}' for p, v in zip(self.param_names, vec) if v != 0]).replace('1.0', '')
            models.append(ChangeVector(vec=vec,
                                       mu_mdl=MyPipeline(poly_degree, regularization),
                                       l_mdl=MyPipeline(poly_degree, regularization),
                                       sigma_v=sigma_v,
                                       prior=prior,
                                       name=name))

        params_test = {p: v.mean() for p, v in self.param_prior_dists.items()}
        n_y = len(self.forward_model(self.x, **params_test))
        n_vec = len(self.vecs)

        y_1 = np.zeros((n_samples, n_y))
        y_2 = np.zeros((n_vec, n_samples, n_y))

        for s_idx in range(n_samples):
            params_1 = {p: v.rvs() for p, v in self.param_prior_dists.items()}
            y_1[s_idx] = self.forward_model(self.x, **params_1)
            for v_idx, vec in enumerate(self.vecs):
                params_2 = {k: v + dv * dv0 for (k, v), dv in zip(params_1.items(), vec)}
                y_2[v_idx, s_idx] = self.forward_model(self.x, **params_2)

        dy = (y_2 - y_1) / dv0
        sample_mu, sample_l = knn_estimation(y_1, dy, k=k)

        for idx in range(n_vec):
            models[idx].mu_mdl.fit(y_1, sample_mu[idx])
            models[idx].l_mdl.fit(y_1, sample_l[idx])

        return ChangeModel(models=models)


# helper functions:
def knn_estimation(y, dy, k=50, lam=1e-6):
    """
    Computes mean and covariance of change per sample using its K nearest neighbours.

    :param y: (n_samples, dim) array of summary measurements
    :param dy: (n_params, n_samples, dim) array of derivatives per poi
    :param k: number of neighbourhood samples
    :param lam: shrinkage value to avoid degenerate covariance matrices
    :return: mu and l per data point
    """
    n_params, n_samples, dim = dy.shape
    mu = np.zeros_like(dy)
    l = np.zeros((n_params, n_samples, dim * (dim + 1) // 2))

    idx = np.tril_indices(dim)
    diag_idx = np.argwhere(idx[0] == idx[1])

    tree = KDTree(y)
    _, neigbs = tree.query(y, k)
    pbar = ProgressBar()
    print('KNN approximation of sample mean and covariance:')
    for p in pbar(range(n_params)):
        for sample_idx in range(n_samples):
            pop = dy[p, neigbs[sample_idx]]
            mu[p, sample_idx, :] = pop.mean(axis=0)
            l[p, sample_idx, :] = np.linalg.cholesky(np.cov(pop.T) + lam * np.eye(dim))[np.tril_indices(dim)]
            for i in diag_idx:
                l[p, sample_idx, i] = np.log(l[p, sample_idx, i])

    return mu, l


def l_to_sigma(l_vec):
    """
         inverse Cholesky decomposition and log transforms diagonals
           :param l_vec: (..., dim(dim+1)/2) vectors
           :return: mu sigma (... , dim, dim)
       """
    t = l_vec.shape[-1]
    dim = int((np.sqrt(8 * t + 1) - 1) / 2)  # t = dim*(dim+1)/2
    idx = np.tril_indices(dim)
    diag_idx = np.argwhere(idx[0] == idx[1])
    l_vec[..., diag_idx] = np.exp(l_vec[..., diag_idx])
    l_mat = np.zeros((*l_vec.shape[:-1], dim, dim))
    l_mat[..., idx[0], idx[1]] = l_vec
    sigma = l_mat @ l_mat.swapaxes(-2, -1)  # transpose last two dimensions

    return sigma


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
