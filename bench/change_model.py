#!/usr/bin/env python3

"""
This module contains classes and functions to train a change model and make inference on new data.
"""

import inspect
import os
import pickle
import warnings
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
from progressbar import ProgressBar
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from scipy.spatial import KDTree
from scipy.cluster.vq import whiten
from scipy.stats import rv_continuous, lognorm
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing import Callable, List, Any, Union, Sequence, Mapping
from scipy import optimize
import numba
from bench import summary_measures

BOUNDS = {'negative': (-np.inf, 0), 'positive': (0, np.inf), 'twosided': (-np.inf, np.inf)}
INTEGRAL_LIMITS = list(BOUNDS.keys())


@dataclass
class KNNChangeVector:
    """
    class for a single model of change
    """
    vec: Mapping
    mu_mdl: Pipeline
    l_mdl: Pipeline
    lim: str
    prior: float = 1
    scale: float = 0.1
    name: str = None

    def __post_init__(self):
        scale = np.round(np.linalg.norm(list(self.vec.values())), 3)
        if scale != 1:
            warnings.warn(f'Change vector {self.name} does not have a unit length')

        if self.name is None:
            self.name = dict_to_string(self.vec)

    def estimate_change(self, y: np.ndarray):
        """
        Computes predicted mu and sigma at the given point for this model of change

        :param y: sample data (..., N) for N summary measures
        :return: Tuple with

            - `mu`: (..., N)-array with mean change
            - `sigma`: (..., N, N)-array with covariance matrix of change
        """
        if y.ndim == 1:
            y = y[np.newaxis, :]

        mu = self.mu_mdl.predict(y)
        sigma, _ = l_to_sigma(self.l_mdl.predict(y))

        return mu, sigma

    def log_lh(self, dv, y, dy, sigma_n):
        """
        Computes log likelihood (P(dy|y, dv)) for the given  measurements
        :param dv: scalar or 1d array of the amount of change
        :param y: the baseline measurements 1d or 2d array with last dimension beaing the measurements.
        :param dy: chage in the meausrements , the same size as y
        :param sigma_n: noise covariance matrix per sample
        :return: loglikelhood per sample
        """
        mu, sigma_p = self.estimate_change(y)
        mean = np.squeeze(dv * mu)
        cov = (dv ** 2) * np.linalg.inv(np.squeeze(sigma_p)) + sigma_n
        return log_mvnpdf(x=dy, mean=mean, cov=cov)

    def log_prior(self, dv):
        """
        Computes the prior probability of the amount of change

        :param dv: the amount of change, scalar or 1d array.
        :return: log(p(dv))
        """
        if self.lim == 'negative':
            dv = -dv
        elif self.lim == 'twosided':
            dv = np.abs(dv)

        p = lognorm(s=np.log(10), scale=self.scale).logpdf(x=dv)  # norm(scale=scale, loc=0).pdf(x=dv)  #

        if self.lim == 'twosided':
            p -= np.log(2)

        return p

    def log_posterior(self, dv, y, dy, sigma_n):
        return self.log_prior(dv) + self.log_lh(dv, y, dy, sigma_n)


@dataclass
class MLChangeVector:
    """
    class for a single model of change trained by maximum likelihood approach
    """
    vec: Mapping
    mu_weight: np.ndarray
    sig_weight: np.ndarray
    mean_y: np.ndarray
    mu_poly_degree: int
    sigma_poly_degree: int
    summary_names: List
    lim: str
    prior: float = 1
    scale: float = 0.1
    name: str = None

    def __post_init__(self):
        scale = np.round(np.linalg.norm(list(self.vec.values())), 3)
        if scale != 1:
            warnings.warn(f'Change vector {self.name} does not have a unit length')

        if self.name is None:
            self.name = dict_to_string(self.vec)

        self.mu_feature_extractor = PolynomialFeatures(degree=self.mu_poly_degree)
        self.sigma_feature_extractor = PolynomialFeatures(degree=self.sigma_poly_degree)

        if self.mean_y is not None:
            tril_idx = np.tril_indices(self.mean_y.shape[-1])
            self.diag_idx = np.argwhere(tril_idx[0] == tril_idx[1])

    def derivative_distribution(self, y):
        y = np.atleast_2d(y)
        yf_mu = self.mu_feature_extractor.fit_transform(y - self.mean_y)
        yf_sigma = self.sigma_feature_extractor.fit_transform(y - self.mean_y)
        mu, sigma_inv, _ = regression_model(yf_mu, yf_sigma, self.mu_weight, self.sig_weight, self.diag_idx)
        sigma = np.linalg.inv(sigma_inv)
        return mu, sigma

    def log_lh(self, dv, y, dy, sigma_n):
        y, dy, sigma_n = summary_measures.normalize_summaries(y, dy, sigma_n, self.summary_names)
        mu, sigma_p = self.derivative_distribution(y)
        mean = np.squeeze(dv * mu)
        cov = (dv ** 2) * np.squeeze(sigma_p) + sigma_n
        return log_mvnpdf(x=dy, mean=mean, cov=cov)

    def log_prior(self, dv):
        if self.lim == 'negative':
            dv = -dv
        elif self.lim == 'twosided':
            dv = np.abs(dv)

        p = lognorm(s=np.log(10), scale=self.scale).logpdf(x=dv)  # norm(scale=scale, loc=0).pdf(x=dv)  #

        if self.lim == 'twosided':
            p -= np.log(2)

        return p

    def log_posterior(self, dv, y, dy, sigma_n):
        return self.log_prior(dv) + self.log_lh(dv, y, dy, sigma_n)


@dataclass
class NoChangeModel:
    summary_names: list
    name: str = 'No change'
    prior: float = 1.0

    def derivative_distribution(self, y):
        y = np.atleast_2d(y)
        return np.zeros((y.shape[0], len(self.summary_names))), \
               np.zeros((y.shape[0], len(self.summary_names), len(self.summary_names)))

    def log_posterior(self, y, dy, sigma_n):
        y, dy, sigma_n = summary_measures.normalize_summaries(y, dy, sigma_n, self.summary_names)
        return log_mvnpdf(x=dy, mean=np.zeros_like(y), cov=sigma_n)


@dataclass
class ChangeModel:
    """
    Class that contains trained models of change for all of the parameters of a given forward models.
    """
    models: List[MLChangeVector]
    model_name: str = 'unnamed'

    def __post_init__(self):
        null_model = NoChangeModel(prior=1,
                                   name='[]',
                                   summary_names= self.models[0].summary_names)

        self.models.insert(0, null_model)

    def save(self, path='./', file_name=None):
        """
        Writes the change model to disk

        :param path: directory to write to
        :param file_name: filename (defaults to model name)
        """
        if file_name is None:
            file_name = self.model_name

        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)

    def predict(self, data, delta_data, sigma_n, parallel=True):
        """
        Computes the posterior probabilities for each model of change

        :param data: numpy array (..., n_dim) containing the first group average
        :param delta_data: numpy array (..., n_dim) containing the change between groups.
        :param sigma_n: numpy array or list (..., n_dim, n_dim)
        :param parallel: flag to run infrence in parallel
        :return: posterior probabilities for each voxel (..., n_vecs)
        """

        print(f'running inference for {data.shape[0]} samples ...')
        lls, peaks = self.compute_log_likelihood(data, delta_data, sigma_n, parallel=parallel, integral_bound=10)
        priors = np.array([m.prior for m in self.models])  # the 1 is for empty set
        priors = priors / priors.sum()
        model_log_posteriors = lls + np.log(priors)
        posteriors = np.exp(model_log_posteriors)
        posteriors[posteriors.sum(axis=1) == 0, 0] = 1  # in case all integrals were zeros accept the null model.
        print('samples with nan posterior:', np.argwhere(np.isnan(posteriors).any(axis=1)))
        posteriors[np.isnan(posteriors)] = 0
        posteriors[np.isposinf(posteriors)] = np.nanmax(posteriors[np.isfinite(posteriors)]) + 1
        posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
        predictions = np.argmax(posteriors, axis=1)
        return posteriors, predictions, peaks

    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    def compute_log_likelihood(self, y, delta_y, sigma_n, integral_bound=1e3, parallel=False):
        """
        Computes log_likelihood function for all models of change.

        Compares the observed data with the result from :func:`predict`.

        :param parallel: flag to run inference in parallel
        :param y: (n_samples, n_dim) array of data
        :param delta_y: (n_samples, n_dim) array of delta data
        :param sigma_n: (n_samples, n_dim, n_dim) noise covariance per sample
        :param integral_bound
        :return: np array containing log likelihood for each sample per class
        """

        n_samples, n_dim = y.shape
        n_models = len(self.models)

        def func(sam_idx):
            log_prob = np.zeros(n_models)
            peaks = np.zeros(n_models)

            y_s = y[sam_idx].T
            dy_s = delta_y[sam_idx]
            sigma_n_s = sigma_n[sam_idx]

            if np.isnan(y_s).any() or np.isnan(dy_s).any() or np.isnan(sigma_n_s).any():
                log_prob = np.ones(n_models) / n_models
                warnings.warn("Received nan inputs at inference.")

            else:
                log_prob[0] = log_mvnpdf(x=dy_s, mean=np.zeros(n_dim), cov=sigma_n_s)

                for vec_idx, ch_mdl in enumerate(self.models[1:], 1):
                    try:
                        log_post_pdf = lambda dv: ch_mdl.log_posterior(dv, y_s, dy_s, sigma_n_s)
                        post_pdf = lambda dv: np.exp(log_post_pdf(dv))

                        if ch_mdl.lim == 'positive':
                            neg_int = 0
                        else:
                            neg_peak, lower, upper = find_range(log_post_pdf, (-integral_bound, 0))
                            if check_exp_underflow(log_post_pdf(neg_peak)):
                                neg_int = 0
                            else:
                                neg_int = quad(post_pdf, lower, upper, points=[neg_peak], epsrel=1e-3)[0]

                        if ch_mdl.lim == 'negative':
                            pos_int = 0
                        else:
                            pos_peak, lower, upper = find_range(log_post_pdf, (0, integral_bound))
                            if check_exp_underflow(pos_peak):
                                pos_int = 0
                            else:
                                pos_int = quad(post_pdf, lower, upper, points=[pos_peak], epsrel=1e-3)[0]

                        integral = pos_int + neg_int
                        if integral == 0:
                            log_prob[vec_idx] = -np.inf
                        else:
                            log_prob[vec_idx] = np.log(integral)

                        if ch_mdl.lim == 'positive':
                            peaks[vec_idx] = pos_peak
                        if ch_mdl.lim == 'negative':
                            peaks[vec_idx] = neg_peak
                        else:
                            if log_post_pdf(neg_peak) < log_post_pdf(pos_peak):
                                peaks[vec_idx] = pos_peak
                            else:
                                peaks[vec_idx] = neg_peak

                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            log_prob[vec_idx] = -np.inf
                            warnings.warn(f'noise covariance is singular for sample {sam_idx}'
                                          f'with variances {np.diag(sigma_n_s)}')
                        else:
                            raise
            if np.isnan(log_prob).any():
                print(sam_idx, np.argwhere(np.isnan(log_prob)))
            return log_prob, peaks

        if parallel:
            res = Parallel(n_jobs=-1, verbose=True)(delayed(func)(i) for i in range(n_samples))
        else:
            pbar = ProgressBar()
            res = []
            for i in pbar(range(n_samples)):
                res.append(func(i))

        log_probs = np.array([d[0] for d in res])
        peaks = np.array([d[1] for d in res])

        return log_probs, peaks

    def estimate_quality_of_fit(self, y1, dy, sigma_n, predictions, peaks):
        dv = np.array([p[i] for i, p in zip(predictions, peaks)])
        y1, dy, sigma_n = summary_measures.normalize_summaries(y1, dy, sigma_n, self.models[1].summary_names)
        dists = [self.models[p].derivative_distribution(y) for p, y in zip(predictions, y1)]
        mu = np.stack([np.squeeze(d[0]) for d in dists], axis=0)
        sigma = np.stack([np.squeeze(d[1]) for d in dists], axis=0)

        offset = dy - mu * dv[:, np.newaxis]
        sigma_inv = np.linalg.inv(sigma_n + sigma * dv[:, np.newaxis, np.newaxis] ** 2)
        deviation = np.einsum('ij,ijk,ik->i', offset, sigma_inv, offset)

        return dv, offset, deviation

    @classmethod
    def load(cls, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                mdl = pickle.load(f)
                return mdl
        else:
            raise SystemExit('model file not found')


def make_pipeline(degree: int, alpha: float) -> Pipeline:
    steps = [('features', PolynomialFeatures(degree=degree)),
             ('reg', linear_model.Ridge(alpha=alpha, fit_intercept=True))]
    return Pipeline(steps=steps)


def string_to_dict(str_):
    dict_ = {}
    str_ = str_.replace('+ ', '+')
    str_ = str_.replace('- ', '-')
    str_ = str_.replace('* ', '*')
    str_ = str_.replace(' *', '*')

    for term in str_.split(' '):
        if term[0] in ('+', '-'):
            sign = term[0]
            term = term[1:]
        else:
            sign = '+'

        if '*' in term:
            n = term.split('*')
            coef = n[0]
            pname = n[1]
        else:
            coef = '1'
            pname = term

        dict_[pname] = float(sign + coef)
    return dict_


def dict_to_string(dict_):
    str_ = ' '.join([f'{v:+1.1f}*{p}' for p, v in
                     dict_.items() if v != 0]).replace('1.0*', '')
    if str_[0] == '+':
        str_ = str_[1:]
    str_ = str_.replace(' +', ' + ')
    str_ = str_.replace(' -', ' - ')
    return str_


def parse_change_vecs(vec_texts: List[str]):
    vecs = list()
    lims = list()
    for txt in vec_texts:
        if '#' in txt:
            txt = txt[:txt.find('#')]

        sub = txt.split(',')
        this_vec = string_to_dict(sub[0])
        if len(sub) > 1:
            this_lims = sub[1].rstrip().lower().split()
            for l in this_lims:
                if l not in INTEGRAL_LIMITS:
                    raise ValueError(f'limits must be one of {INTEGRAL_LIMITS} but got {l}.')
        else:
            this_lims = ['twosided']

        vecs.append(this_vec)
        lims.append(this_lims)

    return vecs, lims


@dataclass
class Trainer:
    """
        A class for training models of change
    """
    forward_model: Callable
    """ The forward model, it must be function of the form f(args, **params) that returns a numpy array. """
    param_prior_dists: Mapping[str, rv_continuous]
    kwargs: Mapping[str, Any] = None
    change_vecs: List[Mapping] = None
    lims: List = None
    priors: Union[float, Sequence] = 1

    def __post_init__(self):

        if self.kwargs is None:
            self.kwargs = dict()

        if self.change_vecs is None:
            self.change_vecs = [{p: 1} for p in self.param_names]

        elif np.all([isinstance(p, str) for p in self.change_vecs]):
            self.change_vecs, self.lims = parse_change_vecs(self.change_vecs)
        else:
            raise ValueError(" Change vectors are not defined properly.")

        if self.lims is None:
            self.lims = [['twosided']] * len(self.change_vecs)

        self.vec_names = [dict_to_string(s) for s in self.change_vecs]
        self.n_vecs = len(self.change_vecs)

        for idx in range(self.n_vecs):
            for pname in self.change_vecs[idx].keys():
                if pname not in self.param_names:
                    raise KeyError(f"parameter {pname} is not defined as a free parameters of the forward model."
                                   f"The parameters are {self.param_names}")
            # normalize change vectors to have a unit L2 norm:
            scale = np.linalg.norm(list(self.change_vecs[idx].values()))
            self.change_vecs[idx] = {k: v / scale for k, v in self.change_vecs[idx].items()}

        if np.isscalar(self.priors):
            self.priors = [self.priors] * self.n_vecs
        elif len(self.priors) != self.n_vecs:
            raise ("priors must be either a scalar (same for all change models) "
                   "or a sequence with size of number of change models.")

        params_test = sample_params(self.param_prior_dists)
        try:
            self.n_dim = self.forward_model(**self.kwargs, **params_test).size
            tril_idx = np.tril_indices(self.n_dim)
            self.diag_idx = np.argwhere(tril_idx[0] == tril_idx[1])

        except Exception as ex:
            print("The provided function does not work with the given parameters and args.")
            raise ex
        if self.n_dim == 1:
            raise ValueError('The forward model must produce at least two dimensional output.')

    @property
    def param_names(self, ):
        return [i for p in self.param_prior_dists.keys()
                for i in ([p] if isinstance(p, str) else p)]

    def vec_to_dict(self, vec: np.ndarray):
        return {p: v for p, v in zip(self.param_names, vec) if v != 0}

    def dict_to_vec(self, dict_: Mapping):
        return np.array([dict_.get(p, 0) for p in self.param_names])

    def train_knn(self, n_samples=1000, poly_degree=2, regularization=1, k=100, dv0=1e-6, verbose=True):
        """
        Train change models (estimates w_mu and w_l) using forward model simulation

        :param verbose:
        :param n_samples: number simulated samples to estimate forward model
        :param k: nearest neighbours parameter
        :param dv0: the amount perturbation in parameter to estimate derivatives
        :param poly_degree: polynomial degree for regression
        :param regularization: ridge regression alpha
        """
        print('Generating training samples...')
        y_1, y_2 = self.generate_train_samples(n_samples, dv0)
        dy = (y_2 - y_1) / dv0

        sample_mu, sample_l = knn_estimation(y_1, dy, k=k)

        models = []
        if verbose:
            print('Training the models...')
        for idx, (vec, name, prior, lims) \
                in enumerate(zip(self.change_vecs, self.vec_names, self.priors, self.lims)):

            mu_mdl = make_pipeline(poly_degree, regularization)
            l_mdl = make_pipeline(poly_degree, regularization)
            mu_mdl.fit(y_1, sample_mu[idx])
            l_mdl.fit(y_1, sample_l[idx])
            for l in lims:
                models.append(
                    KNNChangeVector(vec=vec,
                                    mu_mdl=mu_mdl,
                                    l_mdl=l_mdl,
                                    prior=prior,
                                    lim=l,
                                    name=str(name) + '_' + l)
                )
                if verbose:
                    print(models[-1].name)

        return ChangeModel(models=models)

    def train_ml(self, n_samples=1000, mu_poly_degree=2, sigma_poly_degree=1, alpha=.1, dv0=1e-6,
                 verbose=True, summary_names=None, parallel=True):
        """
        Train change models (estimates w_mu and w_l) using forward model simulation

        :param summary_names: name of summary measure, only for diffusion training.
        :param n_samples: number simulated samples to estimate forward model
        :param dv0: the amount perturbation in parameter to estimate derivatives
        :param mu_poly_degree: polynomial degree for regressing mu
        :param sigma_poly_degree: polynomial degree for regressing sigma
        :param alpha: ridge regression regularization weight
        :param parallel: train models in parallel (on thread per model of change)

        :param verbose: flag for printing steps.
        """
        print(f'Generating {n_samples} training samples...')
        y_1, y_2 = self.generate_train_samples(n_samples, dv0)
        dy = (y_2 - y_1) / dv0
        mean_y = y_1.mean(axis=0)[np.newaxis, :]
        yf_mu = PolynomialFeatures(degree=mu_poly_degree).fit_transform(y_1 - mean_y)
        yf_sigma = PolynomialFeatures(degree=sigma_poly_degree).fit_transform(y_1 - mean_y)
        n_mu_features = yf_mu.shape[-1]
        if verbose:
            print('Training models of change ...')

        def func(idx):
            mu_weights = optimize.minimize(neg_log_likelihood,
                                           x0=np.zeros(self.n_dim * n_mu_features),
                                           method=None,
                                           args=(dy[idx], yf_mu, yf_sigma, None, self.diag_idx, 'mu', alpha))

            print(f' mu optimization for {self.vec_names[idx]}: {mu_weights.message}\n '
                  f'function value:{mu_weights.fun}, {mu_weights.nit}')

            x0 = np.zeros((self.n_dim * (self.n_dim + 1) // 2) * yf_sigma.shape[-1])
            sigma_weights = optimize.minimize(neg_log_likelihood, method='BFGS',
                                              x0=x0,
                                              args=(dy[idx], yf_mu, yf_sigma, mu_weights.x, self.diag_idx, 'sigma', alpha))

            print(f' sigma optimization for {self.vec_names[idx]}: {sigma_weights.message}\n '
                  f'function value:{sigma_weights.fun}, {sigma_weights.nit}')

            all_weights = optimize.minimize(neg_log_likelihood,
                                            method='BFGS',
                                            x0=np.concatenate([mu_weights.x, sigma_weights.x]),
                                            args=(dy[idx], yf_mu, yf_sigma, None, self.diag_idx, 'both', alpha))

            print(f' mu+sigma optimization for {self.vec_names[idx]}: {all_weights.message}\n '
                  f'function value:{all_weights.fun}, {all_weights.nit}')

            w_mu = all_weights.x[:(n_mu_features * self.n_dim)].reshape(n_mu_features, self.n_dim)
            w_sigma = all_weights.x[n_mu_features * self.n_dim:].reshape\
                (yf_sigma.shape[-1], self.n_dim * (self.n_dim + 1) // 2)

            models = []
            for l in self.lims[idx]:
                models.append(
                    MLChangeVector(vec=self.change_vecs[idx],
                                   mu_weight=w_mu,
                                   sig_weight=w_sigma,
                                   mean_y=mean_y,
                                   mu_poly_degree=mu_poly_degree,
                                   sigma_poly_degree=sigma_poly_degree,
                                   prior=self.priors[idx],
                                   lim=l,
                                   summary_names=summary_names,
                                   name=str(self.vec_names[idx]) + '_' + l)
                )
                if verbose:
                    print(f'Tained models for {models[-1].name}.')
            return models

        if parallel:
            models = Parallel(n_jobs=-1, verbose=True)(delayed(func)(i) for i in range(len(self.change_vecs)))
            models = [m for a in models for m in a]
        else:
            models = []
            for i in range(len(self.change_vecs)):
                models.extend(func(i))

        return ChangeModel(models=models)

    def generate_train_samples(self, n_samples: int, dv0: float = 1e-6) -> tuple:
        """
        generate samples to estimate derivatives.
        :param n_samples:
        :param dv0:
        :param parallel:
        :param old:
        :return: M(V) , M(V+\Delta V).
        """
        all_params = sample_params(self.param_prior_dists, n_samples=n_samples)

        for vec in self.change_vecs:
            for (k, v) in all_params.items():
                params_2 = v + vec.get(k, 0) * dv0
                if k in self.param_prior_dists and hasattr(self.param_prior_dists[k], 'pdf'):
                    invalid = self.param_prior_dists[k].pdf(params_2) == 0
                    all_params[k][invalid] -= vec.get(k, 0) * dv0

        y1 = self.forward_model(**self.kwargs, **all_params)
        y2 = []
        for vec in self.change_vecs:
            params_2 = {k: v + vec.get(k, 0) * dv0 for k, v in all_params.items()}
            y2.append(self.forward_model(**self.kwargs, **params_2))
        y2 = np.stack(y2, 0)

        nans = np.isnan(y2).any(axis=(0, 2)) | np.isnan(y1).any(axis=1)
        y1 = y1[~nans]
        y2 = y2[:, ~nans, :]
        if np.sum(nans) > 0:
            warnings.warn(f'{np.sum(nans)} nan samples were generated during training.')
        return y1, y2

    def generate_test_samples(self, n_samples=1000, effect_size=0.1, noise_level=0.0, n_repeats=1,
                              base_params=None, true_change=None, summary_names=None):
        """
        Generate signal and covariance matrix for baseline parameters and parameters shifted across change vector

        Important note: for this feature the forward model needs to have an internal noise model that 
        accepts the parameter 'noise_level'. Otherwise, gaussian white noise is added to the measurements.

        :param true_change: index of true parameter change for each sample (0 for no change), array or list
        :param base_params: parameter values for the baseline
        :param n_samples: number of samples
        :param effect_size: the amount of change
        :param noise_level: level of measurement noise
        :param n_repeats: number of repeats to estimate noise covariance.
        :return: for N samples and M observational parameters returns tuple with

            - (N, ) index array with which change vector was applied
            - (N, M) with noisy baseline data
            - (N, M) with noisy data with shifted parameters
            - (N, M, M) or (1, M, M) array with noise matrix for each sample
        """

        models = []

        for vec, name, prior, lims in zip(self.change_vecs, self.vec_names, self.priors, self.lims):
            for l in lims:
                models.append(MLChangeVector(vec=vec, prior=prior, lim=l, mu_weight=None, sig_weight=None,
                                             mean_y=None, mu_poly_degree=None, sigma_poly_degree=None,
                                             name=str(name) + ', ' + l, summary_names=summary_names))
        if true_change is None:
            # randomly choose what to change
            true_change = np.random.randint(0, len(models) + 1, n_samples)
        elif n_samples is None:
            n_samples = len(true_change)
        else:
            assert n_samples == len(true_change)

        if np.isscalar(effect_size):
            effect_size = np.ones(n_samples) * effect_size
        else:
            assert n_samples == len(effect_size)

        kwargs = self.kwargs.copy()
        has_noise_model = 'noise_level' in inspect.getfullargspec(self.forward_model).args

        if has_noise_model:
            kwargs['noise_level'] = noise_level
            sigma_n = np.zeros((n_samples, self.n_dim, self.n_dim))
        else:
            sigma_n = np.eye(self.n_dim)[None, :, :] * noise_level ** 2

        print(f'Generating {n_samples} test samples:')

        y_1 = np.zeros((n_samples, self.n_dim))
        y_2 = np.zeros_like(y_1)
        pbar = ProgressBar()
        for s_idx in pbar(range(n_samples)):
            valid = False
            while not valid:
                if base_params is None:
                    params_1 = sample_params(self.param_prior_dists)
                else:
                    params_1 = base_params

                tc = true_change[s_idx]
                if tc == 0:
                    params_2 = params_1.copy()
                    valid = True
                else:
                    lim = models[tc - 1].lim
                    sign = {'positive': 1, 'negative': -1, 'twosided': 1}[lim]
                    params_2 = {k: v + models[tc - 1].vec.get(k, 0) * effect_size[s_idx] * sign
                                for k, v in params_1.items()}

                    valid = np.all([self.param_prior_dists[k].pdf(params_2[k]) > 0 for k in params_2.keys()
                                    if k in self.param_prior_dists and hasattr(self.param_prior_dists[k], 'pdf')])

                    if not valid and base_params is not None:
                        raise ValueError("...")

            if has_noise_model:
                y_1_r = np.zeros((n_repeats, self.n_dim))
                y_2_r = np.zeros_like(y_1_r)
                for r in range(n_repeats):
                    y_1_r[r] = self.forward_model(**kwargs, **params_1)
                    y_2_r[r] = self.forward_model(**kwargs, **params_2)
                y_1[s_idx] = y_1_r[0]  # a random sample.
                y_2[s_idx] = y_2_r[0]
                sigma_n[s_idx] = np.cov((y_2_r - y_1_r).T)
            else:
                y_1[s_idx] = self.forward_model(**kwargs, **params_1) + np.random.randn(self.n_dim) * noise_level
                y_2[s_idx] = self.forward_model(**kwargs, **params_2) + np.random.randn(self.n_dim) * noise_level

        # data = Parallel(n_jobs=-1, verbose=True)(delayed(generator_func)(i) for i in range(n_samples))

        return true_change, y_1, y_2, sigma_n


# helper functions:
def knn_estimation(y, dy, k=100, lam=1e-12):
    """
    Computes mean and covariance of change per sample using its K nearest neighbours.

    :param y: (n_samples, n_dim) array of summary measurements
    :param dy: (n_vecs, n_samples, n_dim) array of derivatives per vector of change
    :param k: number of neighbourhood samples
    :param lam: shrinkage value to avoid degenerate covariance matrices relative to minimum change
    :return: mu and l per data point
    """
    # make lambda relative tominimum:
    lam *= np.abs(dy[np.nonzero(dy)]).min()

    n_vecs, n_samples, dim = dy.shape
    mu = np.zeros_like(dy)
    tril = np.zeros((n_vecs, n_samples, dim * (dim + 1) // 2))

    idx = np.tril_indices(dim)
    diag_idx = np.argwhere(idx[0] == idx[1])

    y_whitened = whiten(y)
    tree = KDTree(y_whitened)
    dists, neigbs = tree.query(y_whitened, k)
    weights = 1 / (dists + 1)
    pbar = ProgressBar()
    print('KNN approximation of sample means and covariances:')
    for vec_idx in pbar(range(n_vecs)):
        for sample_idx in range(n_samples):
            pop = dy[vec_idx, neigbs[sample_idx]]
            mu[vec_idx, sample_idx] = np.average(pop, axis=0, weights=weights[sample_idx])
            try:
                c = np.cov(pop.T, aweights=weights[sample_idx])
                tril[vec_idx, sample_idx] = np.linalg.cholesky(c + lam * np.eye(dim))[np.tril_indices(dim)]
            except Exception as ex:
                print(ex)
                raise ex

            for i in diag_idx:
                tril[vec_idx, sample_idx, i] = np.log(tril[vec_idx, sample_idx, i])

    return mu, tril


def l_to_sigma(l_vec, diag_idx, numba=True):
    """
         inverse Cholesky decomposition and log transforms diagonals
           :param l_vec: (..., dim(dim+1)/2) vectors
           :return: sigma (... , dim, dim) det_sigma (..., dim)
       """

    t = l_vec.shape[-1]
    dim = int((np.sqrt(8 * t + 1) - 1) / 2)  # t = dim*(dim+1)/2

    if numba:
        sigma = np.zeros((l_vec.shape[0], dim, dim))
        _mat_lower_diagonal(l_vec, sigma)
     # transpose last two dimensions:
    log_dets = 2 * np.squeeze(l_vec[..., diag_idx]).sum(axis=-1)
    return sigma, log_dets


@numba.jit(nopython=True)
def _mat_lower_diagonal(l_vec, l_sigma):
    """Multiplies a lower diagonal matrix with its transpose.

    Numba helper function used in l_to_sigma.
    """
    n, t = l_vec.shape
    n2, dim, dim2 = l_sigma.shape
    assert dim == dim2
    assert n == n2
    assert t == dim * (dim + 1) / 2
    a_i = np.zeros(n)
    a_j = np.zeros(n)
    isum = 0
    for i in range(dim):
        isum += i
        for k in range(i + 1):
            a_i[()] = l_vec[:, isum + k]
            if i == k:
                a_i[()] = np.exp(a_i)
            jsum = 0
            for j in range(dim):
                jsum += j
                if i <= j:
                    if i == j:
                        a_j[()] = a_i
                    else:
                        a_j[()] = l_vec[:, jsum + k]
                        if j == k:
                            a_j[()] = np.exp(a_j)
                    l_sigma[:, i, j] += a_i * a_j
        for j in range(i):
            l_sigma[:, i, j] = l_sigma[:, j, i]


def log_mvnpdf(x, mean, cov):
    """
    log of multivariate normal distribution. identical output to scipy.stats.multivariate_normal(mean, cov).logpdf(x) but faster.
    :param x: input numpy array
    :param mean: mean of the distribution numpy array same size of x
    :param cov: covariance of distribution, numpy array or scalar
    :return scalar
    """
    cov = np.atleast_2d(cov)
    d = mean.shape[-1]
    expo = -0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
    nc = -0.5 * np.log(((2 * np.pi) ** d) * abs(np.linalg.det(cov.astype(float))))

    # var2 = np.log(multivariate_normal(mean=mean, cov=cov).pdf(x))
    return expo + nc


def find_range(f: Callable, bounds, scale=1e-3):
    """
     find the range for integration
    :param f: function in logarithmic scale, e.g. log_posterior
    :param bounds:
    :param scale: the ratio of limits to peak
    :param search_rad: radious to search for limits
    :return: peak, lower limit and higher limit of the function
    """
    neg_f = lambda dv: -f(dv)
    np.seterr(invalid='raise')
    peak = minimize_scalar(neg_f, bounds=bounds, method='bounded').x

    f_norm = lambda dv: f(dv) - (f(peak) + np.log(scale))
    lower, upper = bounds
    if f_norm(lower) < 0:  # to check the function is not flat.
        try:
            lower = root_scalar(f_norm, bracket=[lower, peak], method='brentq').root
        except Exception as e:
            print(f"Error of type {e.__class__} occurred while finding the lower limit of integral."
                  f"The lower bound is used instead")

    if f_norm(upper) < 0:
        try:
            upper = root_scalar(f_norm, bracket=[peak, upper], method='brentq').root
        except Exception as e:
            print(f"Error of type {e.__class__} occurred, while finding higher limit of integral."
                  f"The upper bound is used instead")

    return peak, lower, upper


def check_exp_underflow(x):
    """
    Checks if underflow happens for calculating exp(x)
    :param x: float
    :return: 1 if underflow error happens for x
    """
    with np.errstate(under='raise'):
        try:
            np.exp(x)
            return False
        except FloatingPointError:
            return True


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


def neg_log_likelihood(weights, dy, yf_mu, yf_sigma, w_mu, diag_idx, opt_param='mu', alpha=0.1):
    """
     Computes the negative log-liklihood for a set of weights given y and dy
    :param weights: vectorized weights (d * n_features), if fixed sigma is False + d(d+1)/2 * n_features
    :param yf: features extracted from summary measurements (n, n_features)
    :param dy: derivatives (n, d)
    :param alpha: regularization weight
    :param fixed_sigma: flag for fixing sigma
    :return: -log(P(dy | yf, weights)) + alpha * |weights|^2
    """
    n_samples, n_dim = dy.shape

    if opt_param == 'mu':
        _, n_features = yf_mu.shape
        w_mu = weights.reshape(n_features, n_dim)
        mu, _, _ = regression_model(yf_mu, yf_sigma, w_mu, None, diag_idx)
        offset = dy - mu
        nll = np.linalg.norm(offset) ** 2 + alpha * np.mean(w_mu[1:, :] ** 2)

    elif opt_param == 'sigma':
        _, n_features = yf_sigma.shape
        w_sigma = weights.reshape(n_features, n_dim * (n_dim + 1) // 2)
        w_mu_ = w_mu.reshape(yf_mu.shape[-1], n_dim)
        mu, sigma_inv, ldet = regression_model(yf_mu, yf_sigma, w_mu_, w_sigma, diag_idx)
        offset = dy - mu
        nll = np.mean(-ldet + np.einsum('ij,ijk,ik->i', offset, sigma_inv, offset)) + \
              alpha * np.mean(w_sigma[1:, :] ** 2)

    elif opt_param == 'both':
        n_mu_features = yf_mu.shape[-1]
        n_sig_features = yf_sigma.shape[-1]
        w_mu = weights[:n_mu_features * n_dim].reshape(n_mu_features, n_dim)
        w_sigma = weights[n_mu_features * n_dim:].reshape(n_sig_features, n_dim * (n_dim + 1) // 2)
        mu, sigma_inv, ldet = regression_model(yf_mu, yf_sigma, w_mu, w_sigma, diag_idx)
        offset = dy - mu
        nll = np.mean(-ldet +
                      np.einsum('ij,ijk,ik->i', offset, sigma_inv, offset)) + \
              alpha * (np.mean(w_mu[1:, :] ** 2) + np.mean(w_sigma[1:, :] ** 2))
    return nll


def regression_model(yf_mu, yf_sigma, w_mu, w_sigma, diag_idx, lam=0):
    """
    Given some measurements and regression weights, computes the hyperparameters of derivatives (mean and covariance)
    :param yf_mu: feature vector (..., n_features)
    :param w_mu: weight matrix for mu (d , n_features)
    :param w_sigma: (d(d+1)/2 , n_features)
    :param lam: shrinkage weight
    :return:
    """
    n_dim = w_mu.shape[1]
    mu = yf_mu @ w_mu
    if w_sigma is None:
        sigma = np.eye(n_dim)
        log_det = 0
    else:
        sigma, log_det = l_to_sigma(yf_sigma @ w_sigma, diag_idx)
        sigma = sigma + lam * np.eye(n_dim)
    return mu, sigma, log_det


def plot_conf_mat(conf_mat, param_names, f_name=None, title=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(conf_mat.T, annot=True, fmt="2.2f", vmin=0, vmax=1,
                     annot_kws={'size': 12})

    plt.tight_layout()
    ax.set_xticklabels(labels=param_names, rotation=45, fontdict={'size': 12})
    ax.set_yticklabels(labels=param_names, rotation=45, fontdict={'size': 12})
    ax.set_xlabel('Actual change', fontdict={'size': 16})
    ax.set_ylabel('Predicted Change', fontdict={'size': 16})
    if title is not None:
        plt.title(title)
    if f_name is not None:
        plt.savefig(f_name, bbox_inches='tight')
    plt.show()


def sample_params(param_prior_dists, n_samples=1):
    """Sample from the prior distributions

    Prior distributions are given by a dictionary from parameter names to a sampling function or scipy distribution.
    Independent parameters can map to a 1D prior distribution or we can map from a list of dependent parameters to a multi-dimensional distribution.
    """
    params = {}
    for p, dist in param_prior_dists.items():
        func = getattr(dist, 'rvs', dist)
        if isinstance(p, str):
            params[p] = func(n_samples)
        else:
            for single_p, samples in zip(p, func(n_samples)):
                params[single_p] = samples
    return params
