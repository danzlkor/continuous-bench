"""
This module contains classes and functions to train a change model and make inference on new data.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import rv_continuous
from typing import Callable, List, Any, Union, Sequence, Mapping
from progressbar import ProgressBar
from dataclasses import dataclass
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline
import os
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
import warnings
import pickle


@dataclass
class ChangeVector:
    """
    class for a single model of change
    """
    vec: np.ndarray
    mu_mdl: Pipeline
    l_mdl: Pipeline
    lim: str = 'b'
    sigma_v: float = 0.1
    prior: float = 1
    name: str = 'unnamed'

    def __post_init__(self):
        self.vec = self.vec / np.linalg.norm(self.vec)

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
        sigma = l_to_sigma(self.l_mdl.predict(y))

        return mu, sigma


@dataclass
class ChangeModel:
    models: List[ChangeVector]
    model_name: str

    def __post_init__(self):
        if self.model_name is None:
            self.model_name = 'unnamed'

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

    def predict(self, data, delta_data, sigma_n):
        """
        Computes the posterior probabilities for each model of change

        :param data: numpy array (..., d) containing the first group average
        :param delta_data: numpy array (..., d) containing the change between groups.
        :param sigma_n: numpy array or list (..., d, d)
        :return: posterior probabilities for each voxel (..., n_params)
        """

        print(f'running inference for {data.shape[0]} samples ...')
        lls = self.compute_log_likelihood(data, delta_data, sigma_n)
        priors = np.array([1.] + [m.prior for m in self.models])  # the 1 is for empty set
        priors = priors / priors.sum()
        model_log_posteriors = lls + np.log(priors)
        posteriors = np.exp(model_log_posteriors)
        posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
        predictions = np.argmax(posteriors, axis=1)
        return posteriors, predictions

    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    def compute_log_likelihood(self, y, delta_y, sigma_n):
        """
        Computes log_likelihood function for all models of change.

        Compares the observed data with the result from :func:`predict`.

        :param y: (n_samples, n_x) array of summary measurements
        :param delta_y: (n_samples, n_x) array of delta data
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

            if np.isnan(y_s).any() or np.isnan(dy_s).any() or np.isnan(sigma_n_s).any():
                log_prob[sam_idx, :] = np.zeros(n_models)
            else:
                log_prob[sam_idx, 0] = log_mvnpdf(x=dy_s, mean=np.zeros(n_x), cov=sigma_n_s)

                for vec_idx, ch_mdl in enumerate(self.models, 1):
                    try:
                        mu, sigma_p = ch_mdl.estimate_change(y_s)
                        fun = lambda dv: posterior_dv(dv, dy_s,
                                                      mu, sigma_p, sigma_n_s,
                                                      ch_mdl.sigma_v, ch_mdl.lim)

                        peak, low, high = find_range(fun)
                        fun2 = lambda dv: np.exp(fun(dv))
                        integral = quad(fun2, low, high, epsrel=1e-3)[0]
                        log_prob[sam_idx, vec_idx] = np.log(integral) if integral > 0 else -np.inf

                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            log_prob[sam_idx, vec_idx] = -1e3
                            warnings.warn(f'noise covariance is singular for sample {sam_idx}'
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


def make_pipeline(degree: int, alpha: float) -> Pipeline:
    steps = [('features', PolynomialFeatures(degree=degree)),
             ('reg', linear_model.Ridge(alpha=alpha, fit_intercept=False))]
    return Pipeline(steps=steps)


def string_to_dict(str_):
    dict_ = {}
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


def dict_to_str(dict_):
    return ' '.join([f'{v:+1.1f}*{p}' for p, v in dict_.items() if v != 0]).replace('1.0', '')


@dataclass
class Trainer:
    """
        A class for training models of change
    """
    forward_model: Callable
    """ The forward model, it must be function of the form f(args, **params) that returns a numpy array. """
    args: Any
    param_prior_dists: Mapping[str, rv_continuous]
    change_vecs: List[Mapping] = None
    sigma_v: Union[float, List[float], np.ndarray] = 0.1
    priors: Union[float, Sequence] = 1

    def __post_init__(self):

        self.param_names = list(self.param_prior_dists.keys())

        if self.change_vecs is None:
            self.change_vecs = [{p: c} for p in self.param_names for c in (-1, 1)]
            self.vec_names = [dict_to_str(s) for s in self.change_vecs]
        elif np.all([p is dict for p in self.change_vecs]):
            self.vec_names = [dict_to_str(s) for s in self.change_vecs]
        elif np.all([isinstance(p, str) for p in self.change_vecs]):
            self.vec_names = self.change_vecs
            self.change_vecs = [string_to_dict(str(s)) for s in self.vec_names]
        else:
            raise ValueError(" Change vectors are not defined properly.")

        for d in self.change_vecs:
            for pname in d.keys():
                if pname not in self.param_names:
                    raise KeyError(f"parameter {pname} is not defined as a free parameters of the forward model."
                                   f"The parameters are {self.param_names}")

        self.n_vecs = len(self.change_vecs)

        if np.isscalar(self.sigma_v):
            self.sigma_v = [self.sigma_v] * self.n_vecs
        elif len(self.sigma_v) != self.n_vecs:
            raise ("sigma_v must be either a scalar (same for all change models) "
                   "or a sequence with size of number of change models.")

        if np.isscalar(self.priors):
            self.priors = [self.priors] * self.n_vecs
        elif len(self.priors) != self.n_vecs:
            raise ("priors must be either a scalar (same for all change models) "
                   "or a sequence with size of number of change models.")

        params_test = {p: v.mean() for p, v in self.param_prior_dists.items()}
        try:
            self.n_y = self.forward_model(**self.args, **params_test).size
        except Exception as ex:
            print("The provided function does not work with the given parameters and args.")
            raise ex
        if self.n_y == 1:
            raise ValueError('The forward model must produce at least two dimensional output.')

    def vec_to_dict(self, vec: np.ndarray):
        return {p: v for p, v in zip(self.param_names, vec) if v != 0}

    def dict_to_vec(self, dict_: Mapping):
        return np.array([dict_.get(p, 0) for p in self.param_names])

    def train(self, n_samples=10000, poly_degree=2, regularization=1, k=100, dv0=1e-6, model_name=None):
        """
          Train change models (estimates w_mu and w_l) using forward model simulation
            :param n_samples: number simulated samples to estimate forward model
            :param k: nearest neighbours parameter
            :param dv0: the amount perturbation in parameter to estimate derivatives
            :param poly_degree: polynomial degree for regression
            :param regularization: ridge regression alpha
            :param model_name: name of the model.
          """
        models = []
        for vec, name, sigma_v, prior in zip(self.change_vecs, self.vec_names, self.sigma_v, self.priors):
            models.append(ChangeVector(vec=self.dict_to_vec(vec),
                                       mu_mdl=make_pipeline(poly_degree, regularization),
                                       l_mdl=make_pipeline(poly_degree, regularization),
                                       sigma_v=sigma_v,
                                       prior=prior,
                                       name=str(name)))

        y_1, y_2 = self.generate_samples(n_samples, dv0)
        dy = (y_2 - y_1) / dv0
        sample_mu, sample_l = knn_estimation(y_1, dy, k=k)

        for idx in range(self.n_vecs):
            models[idx].mu_mdl.fit(y_1, sample_mu[idx])
            models[idx].l_mdl.fit(y_1, sample_l[idx])
            print(f'Model of change for vector {models[idx].name} trained successfully.')

        return ChangeModel(models=models, model_name=model_name)

    def generate_samples(self, n_samples: int, eff_size: float) -> tuple:
        y_1 = np.zeros((n_samples, self.n_y))
        y_2 = np.zeros((self.n_vecs, n_samples, self.n_y))

        for s_idx in range(n_samples):
            params_1 = {p: v.rvs() for p, v in self.param_prior_dists.items()}
            y_1[s_idx] = self.forward_model(**self.args, **params_1)
            for v_idx, vec in enumerate(self.change_vecs):
                params_2 = {k: v + dv * eff_size for (k, v), dv in zip(params_1.items(), self.dict_to_vec(vec))}
                y_2[v_idx, s_idx] = self.forward_model(**self.args, **params_2)
        return y_1, y_2

    def generate_test_samples(self, n_samples=1000, effect_size=0.1, noise_level=0.0, n_repeats=100):
        y_1 = np.zeros((n_samples, self.n_y))
        y_2 = np.zeros((n_samples, self.n_y))
        sigma_n = np.zeros((n_samples, self.n_y, self.n_y))
        true_change = np.random.randint(0, self.n_vecs + 1, n_samples)
        args = (*self.args[:-1], noise_level)

        y_1_r = np.zeros((n_repeats, self.n_y))
        y_2_r = np.zeros_like(y_1_r)
        for s_idx in range(n_samples):
            params_1 = {p: v.rvs() for p, v in self.param_prior_dists.items()}
            if true_change[s_idx] == 0:
                params_2 = params_1
            else:
                # noinspection PyTypeChecker
                params_2 = {k: v + dv * effect_size for (k, v), dv in
                            zip(params_1.items(),
                                self.vec_to_dict(self.change_vecs[true_change[s_idx] - 1]))}

            for r in range(n_repeats):
                y_1_r[r] = self.forward_model(args, **params_1)
                y_2_r[r] = self.forward_model(args, **params_2)

            y_1[s_idx] = y_1_r.mean(axis=0)
            y_2[s_idx] = y_2_r.mean(axis=0)
            sigma_n[s_idx] = np.cov(y_1_r - y_2_r)
        return y_1, y_2, sigma_n


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
    low_tr = np.zeros((n_params, n_samples, dim * (dim + 1) // 2))

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
            low_tr[p, sample_idx, :] = np.linalg.cholesky(np.cov(pop.T) + lam * np.eye(dim))[np.tril_indices(dim)]
            for i in diag_idx:
                low_tr[p, sample_idx, i] = np.log(low_tr[p, sample_idx, i])

    return mu, low_tr


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


def posterior_dv(dv, dy, mu, sigma_p, sigma_n, sigma_v, lims):
    if lims == 'b':
        log_prior = log_mvnpdf(x=dv, mean=np.zeros(1), cov=sigma_v ** 2)
    elif (lims == 'n' and dv > 0) or (lims == 'p' and dv < 0):
        return -np.inf
    else:
        log_prior = log_mvnpdf(x=dv, mean=np.zeros(1), cov=sigma_v ** 2) * 2

    mean = np.squeeze(mu * dv)
    cov = np.squeeze(sigma_p) * dv ** 2 + sigma_n
    log_lh = log_mvnpdf(x=dy, mean=mean, cov=cov)

    return log_lh + log_prior


def find_range(f: Callable, scale=1e2, search_rad=1):
    minus_f = lambda dv: -f(dv)
    peak = minimize_scalar(minus_f).x
    f2 = lambda dv: f(dv) - (f(peak) -np.log(scale))
    try:
        lower = root_scalar(f2, bracket=[-search_rad + peak, peak], method='brentq').root
    except Exception as e:
        print(f"Error of type {e.__class__} occurred, while finding lower limit of integral."
              f"The peak - {search_rad} is used instead")
        lower = -search_rad + peak
    try:
        upper = root_scalar(f2, bracket=[peak, search_rad + peak], method='brentq').root
    except Exception as e:
        print(f"Error of type {e.__class__} occurred, while finding higher limit of integral."
              f"The peak +{search_rad} is used instead")
        upper = search_rad + peak

    return peak, lower, upper


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
