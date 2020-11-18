"""
This module contains classes and functions to train a change model for a given function.
"""

import numpy as np
import pickle
from scipy.spatial import KDTree
from scipy.stats import rv_continuous
from typing import Callable, List, Any, Mapping
from progressbar import ProgressBar
from dataclasses import dataclass
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline
import os


@dataclass
class ChangeVector:
    mu_mdl: Pipeline
    l_mdl: Pipeline
    prior: int = 1
    
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


class MyPipeline(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, [('features', PolynomialFeatures(degree=2)),
                                 ('reg', linear_model.Ridge(alpha=0.5, fit_intercept=False))])


@dataclass
class ChangeModel:
    """
    Main class for models of change
    """
    forward_model: Callable
    x: Any
    vecs: List[np.ndarray]
    priors: Mapping[str, rv_continuous]
    poly_degree: int = 2
    regularization: float = 0.5

    def __post_init__(self):
        self.models = [ChangeVector(mu_mdl=MyPipeline(),
                                    l_mdl=MyPipeline())
                       for _ in range(len(self.vecs))]

    def save(self, path='./', file_name=None):
        if file_name is None:
            file_name = getattr(self.forward_model, '__name__', __default='unnamed')

        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                mdl = pickle.load(f)
                return mdl
        else:
            raise SystemExit('model file not found')

    def train(self, n_samples=10000, k=100, dv0=1e-6):
        """
          Train change models (estimates w_mu and w_l) using forward model simulation
            :param n_samples: number simulated samples to estimate forward model
            :param k: nearest neighbours parameter
            :param dv0: the amount perturbation in parameter to estimate derivatives
            :param poly_degree: polynomial degree for regression
            :param regularization: ridge regression alpha
          """

        params_test = {p: v.mean() for p, v in self.priors.items()}
        n_y = len(self.forward_model(self.x, **params_test))
        n_vec = len(self.vecs)

        y_1 = np.zeros((n_samples, n_y))
        y_2 = np.zeros((n_vec, n_samples, n_y))

        for s_idx in range(n_samples):
            params_1 = {p: v.rvs() for p, v in self.priors.items()}
            y_1[s_idx] = np.array([self.forward_model(x_, **params_1) for x_ in self.x])
            for v_idx, vec in enumerate(self.vecs):
                params_2 = {k: v + dv * dv0 for (k, v), dv in zip(params_1.items(), vec)}
                y_2[v_idx, s_idx] = self.forward_model(self.x, **params_2)

        dy = (y_2 - y_1) / dv0
        sample_mu, sample_l = knn_estimation(y_1, dy, k=k)

        for idx in range(n_vec):
            self.models[idx].mu_mdl.fit(y_1, sample_mu[idx])
            self.models[idx].l_mdl.fit(y_1, sample_l[idx])


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
