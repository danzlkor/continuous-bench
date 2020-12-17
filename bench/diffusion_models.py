"""
This module contains definition of some microstructural diffusion models and a prior distribution for their parameters.
"""

import numpy as np
from scipy import stats
import numba
from bench import summary_measures

# prior distributions:

dif_coeff = 1.7  # unit: um^2/ms

prior_distributions = dict(
    ball={'d_iso': stats.truncnorm(loc=3, scale=.1, a=-3 / 0.1, b=np.Inf)},
    stick={'d_a': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf)},

    cigar={'d_a': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
           'd_r': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf)
           },

    watson_zeppelin_numerical={
        'd_a': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
        'd_r': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
        'odi': stats.uniform(loc=.5, scale=.4)
    },

    bingham_zeppelin={'d_a': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                      'd_r': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                      'odi': stats.uniform(loc=.5, scale=.4),
                      'odi2': stats.uniform(loc=.5, scale=.4),
                      'psi': stats.uniform(loc=np.pi / 2, scale=np.pi / 4),
                      },

    ball_stick={'s_iso': stats.truncnorm(loc=.5, scale=.2, a=-.5 / .2, b=np.Inf),
                's_a': stats.truncnorm(loc=.5, scale=.2, a=-.5 / .2, b=np.Inf),
                'd_iso': stats.truncnorm(loc=3, scale=.1, a=-3 / 0.1, b=np.Inf),
                'd_a': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                },

    watson_noddi={'s_iso': stats.gamma(a=1, scale=1 / 2),
                  's_in': stats.truncnorm(loc=.5, scale=.2, a=-.5 / .2, b=np.Inf),
                  's_ex': stats.truncnorm(loc=.5, scale=.2, a=-.5 / .2, b=np.Inf),
                  'd_iso': stats.truncnorm(loc=3, scale=.1, a=-3 / .1, b=np.Inf),
                  'd_a_in': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                  'd_a_ex': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                  'tortuosity': stats.uniform(loc=0.1, scale=.8),
                  'odi': stats.uniform(loc=.1, scale=.8),
                  },

    bingham_noddi={'s_iso': stats.gamma(a=1, scale=1 / 2),
                   's_in': stats.truncnorm(loc=.5, scale=.2, a=-.5 / .2, b=np.Inf),
                   's_ex': stats.truncnorm(loc=.5, scale=.2, a=-.5 / .2, b=np.Inf),
                   'd_iso': stats.truncnorm(loc=3, scale=.1, a=-3 / .1, b=np.Inf),
                   'd_a_in': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                   'd_a_ex': stats.truncnorm(loc=dif_coeff, scale=.3, a=-dif_coeff / 0.3, b=np.Inf),
                   'tortuosity': stats.uniform(loc=0.1, scale=.8),
                   'odi': stats.uniform(loc=.2, scale=.6),
                   'odi_ratio': stats.uniform(loc=.1, scale=.8),
                   # 'psi': stats.uniform(loc=np.pi/2, scale=np.pi/2)
                   }
)


# basic compartment definitions:
def ball(bval=0, bvec=np.array([0, 0, 1]), d_iso=1., s0=1.0):
    """
    Simulates diffusion signal for isotropic diffusion

    This is the equation for this compartment:

    .. math::

        s = e^{-b d_{iso}}

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_iso: diffusion coefficient
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    assert s0 >= 0, 's0 cant be negative'
    assert d_iso >= 0, 'diso cant be negative'
    if np.isscalar(bval):
        bval = bval * np.ones(bvec.shape[0])

    return s0 * np.exp(-bval * d_iso)


def stick(bval=0, bvec=np.array([0, 0, 1]), d_a=1., theta=0., phi=0.0, s0=1.0):
    """
    Simulates diffusion signal from single stick model

    The attenuation for the stick is :math:`s=e^{-b d_a (\vec{g} \vec{n})^2}`.

    Simpler equation :math:`a=b`.

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_a: axial diffusion coefficient
    :param theta: angle from z-axis
    :param phi: angle from x axis in xy-plane
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    assert d_a >= 0, 'd_a can\'t be negative'
    assert s0 >= 0, 's0 cant be negative'

    orientation = spherical2cart(theta, phi)
    return s0 * np.exp(-bval * (d_a * bvec.dot(orientation) ** 2))


def cigar(bval=0, bvec=np.array([0, 0, 1]), theta=0., phi=0,
          d_a=1., d_r=0., s0=1.0):
    """
    Simulates diffusion signal from single stick model

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param theta: angle from z-axis
    :param phi: angle from x axis in xy-plane
    :param d_a: axial diffusion coefficient
    :param d_r: radial diffusion coefficient
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    assert d_a >= 0, 'd_a cant be negative'
    assert d_r >= 0, 'd_r cant be negative'
    assert s0 >= 0, 's0 cant be negative'

    orientation = spherical2cart(theta, phi)
    return s0 * np.exp(-bval * (d_r + (d_a - d_r) * bvec.dot(orientation) ** 2))


def bingham_zeppelin(bval=0, bvec=np.array([[0, 0, 1]]), d_a=1., d_r=0.,
                     odi=1., odi2=None, theta=0., phi=0., psi=0., s0=1.):
    """
    Simulates diffusion signal for a bingham-distributed ODF

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_a: axial diffusion coefficient
    :param d_r: radial diffusion coefficient
    :param odi: first dispersion coefficient
    :param odi2: second dispersion coefficient
    :param theta: theta for main diffusion direction
    :param phi: phi for main diffusion direction
    :param psi: first dispersion orientation
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    assert s0 >= 0, 's0 cannot be negative'
    if odi2 is None:
        odi2 = odi  # make it watson distribution.
    assert odi >= odi2 > 0, 'odis must be positive and in order'

    if bvec.ndim == 1:
        bvec = bvec[np.newaxis, :]

    if np.isscalar(bval):
        bval = bval * np.ones(bvec.shape[0])

    k1 = 1 / np.tan(odi * np.pi / 2)
    k2 = 1 / np.tan(odi2 * np.pi / 2)

    r_psi = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    r_theta = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    r_phi = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    b_diag = -np.diag([k1, k2, 0])
    r = r_psi @ r_theta @ r_phi
    bing_mat = r.T @ b_diag @ r
    s = []

    denom = hyp_sapprox(np.linalg.eigvalsh(bing_mat)[::-1])
    for bval_, g in zip(bval, bvec):
        q = bing_mat - bval_ * (d_a - d_r) * g[:, np.newaxis].dot(g[np.newaxis, :])
        num = hyp_sapprox(np.linalg.eigvalsh(q)[::-1]) * np.exp(-d_r * bval_)
        s.append(num)
    return s0 * np.array(s) / denom


def watson_zeppelin_numerical(bval=0, bvec=np.array([[0, 0, 1]]), d_a=1., d_r=0.,
                              odi=1, theta=0., phi=0., s0=1., n_samples=10000):
    """
    Simulates diffusion signal for a watson distribution with numerical integration

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_a: axial diffusion coefficient
    :param d_r: radial diffusion coefficient
    :param odi: first dispersion coefficient
    :param theta: theta for main diffusion direction
    :param phi: phi for main diffusion direction
    :param s0: attenuation for b=0
    :param n_samples: resolution of the surface integral
    :return: simulated signal (M,)
    """
    assert odi > 0, 'odis must be positive'

    if np.isscalar(bval):
        bval = bval * np.ones(bvec.shape[0])

    if bvec.ndim == 1:
        bvec = bvec[np.newaxis, :]

    k = 1 / np.tan(odi * np.pi / 2)
    mu = np.array(spherical2cart(theta, phi))

    theta_samples, phi_samples = uniform_sampling_sphere(n_samples=n_samples)
    normal_samples = np.array(spherical2cart(theta_samples, phi_samples)).T
    wat_pdf_samples = np.exp(k * normal_samples.dot(mu) ** 2)
    wat_pdf_samples = wat_pdf_samples / wat_pdf_samples.sum()

    s = np.zeros_like(bval)
    for g_i, (b, g) in enumerate(zip(bval, bvec)):
        resp = cigar(b, g, d_a=d_a, d_r=d_r, s0=s0, theta=theta_samples, phi=phi_samples)
        s[g_i] = (resp * wat_pdf_samples).sum()
    return np.array(s)


# multi-compartment models:

def ball_stick(bval=0, bvec=np.array([0, 0, 1]), theta=0., phi=0.0, d_a=dif_coeff / 2, d_iso=dif_coeff,
               s_iso=1, s_a=1, s0=1.0):
    """
    Simulates diffusion signal from ball and stick model

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param theta: angle from z-axis
    :param phi: angle from x axis in xy-plane
    :param d_a: axial diffusion coefficient
    :param d_iso: radial diffusion coefficient
    :param s_iso: signal fraction of isotropic diffusion
    :param s_a:  signal fraction of anisotropic diffusion
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    assert s_iso >= 0, 'volume fraction cant be negative'
    assert s_a >= 0, 'volume fraction cant be negative'

    return s_a * stick(bval, bvec, d_a, theta, phi, s0) + s_iso * ball(bval, bvec, d_iso, s0)


def watson_noddi(bval=0, bvec=np.array([0, 0, 1]),
                 s_iso=0.5, s_in=0.5, s_ex=.5,
                 d_iso=1., d_a_in=1., d_a_ex=1.,
                 tortuosity=.5, odi=.5,
                 theta=0., phi=0., s0=1.):
    """
    Simulates diffusion signal with Watson dispressed NODDI model

    :param bval: b-values
    :param bvec: (,3) gradient directions(x, y, z)
    :param s_iso: signal fraction of isotropic diffusion
    :param s_in: signal fraction of intra-axonal diffusion
    :param s_ex: signal fraction of extra-axonal water
    :param d_iso: isotropic diffusion coefficient
    :param d_a_in: axial diffusion coefficient
    :param d_a_ex: axial diffusion coefficient for extra-axonal compartment
    :param tortuosity: ratio of radial to axial diffusivity
    :param odi: dispersion parameter of watson distribution
    :param theta: orientation of stick from z axis
    :param phi: orientation of stick from x axis
    :param s0: attenuation for b=0
    :return: (M,) diffusion signal
        """
    assert s0 >= 0, 's0 cant be negative'
    a_iso = ball(bval=bval, bvec=bvec, d_iso=d_iso, s0=s_iso)
    a_int = bingham_zeppelin(bval=bval, bvec=bvec, d_a=d_a_in, d_r=0,
                             odi=odi, odi2=odi, 
                             psi=0, theta=theta, phi=phi, s0=s_in)
    
    a_ext = bingham_zeppelin(bval=bval, bvec=bvec, d_a=d_a_ex,
                             d_r=d_a_ex * tortuosity,
                             odi=odi, odi2=odi, 
                             psi=0, theta=theta, phi=phi, s0=s_ex)

    return (a_iso + a_int + a_ext) * s0


def bingham_noddi(bval=0, bvec=np.array([0, 0, 1]),
                  s_iso=0.5, s_in=0.5, s_ex=.5,
                  d_iso=1., d_a_in=1., d_a_ex=1., tortuosity=0.5,
                  odi=1, odi_ratio=1, psi=0, theta=0., phi=0., s0=1.):
    """
    Simulates diffusion signal with Bingham dispressed NODDI model

    :param bval: b-values
    :param bvec: (,3) gradient directions(x, y, z)
    :param s_iso: signal fraction of isotropic diffusion
    :param s_in: signal fraction of intra-axonal diffusion
    :param s_ex: signal fraction of extra-axonal water
    :param d_iso: isotropic diffusion coefficient
    :param d_a_in: axial diffusion coefficient
    :param d_a_ex: axial diffusion coefficient for extra-axonal compartment
    :param tortuosity: ratio for radial diffusion coefficient for intra-axonal compartment
    :param odi: dispersion parameter of bingham distribution
    :param odi_ratio: ratio for dispersion parameter of bingham distribution
    :param psi: fanning orientation
    :param theta: orientation of stick from z axis
    :param phi: orientation of stick from x axis
    :param s0: attenuation for b=0
    :return: (M,) diffusion signal
        """

    a_iso = ball(bval=bval, bvec=bvec, d_iso=d_iso, s0=s_iso)
    a_int = bingham_zeppelin(bval=bval, bvec=bvec, d_a=d_a_in, d_r=0,
                             odi=odi, odi2=odi * odi_ratio,
                             psi=psi, theta=theta, phi=phi, s0=s_in)
    a_ext = bingham_zeppelin(bval=bval, bvec=bvec, d_a=d_a_ex, d_r=d_a_ex * tortuosity,
                             odi=odi, odi2=odi * odi_ratio,
                             psi=psi, theta=theta, phi=phi, s0=s_ex)

    return (a_iso + a_int + a_ext) * s0


# helper functions:
def spherical2cart(theta, phi, r=1):
    """
    Converts spherical to cartesian coordinates
    :param theta: angel from z axis
    :param phi: angle from x axis
    :param r: radius
    :return: tuple [x, y, z]-coordinates
    """
    z = r * np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    return x, y, z


def cart2spherical(n):
    """
    Converts to spherical coordinates
    :param n: (:, 3) array containing vectors in (x,y,z) coordinates
    :return: tuple with (phi, theta, r)-coordinates
    """
    r = np.sqrt(np.sum(n ** 2, axis=1))
    theta = np.arccos(n[:, 2] / r)
    phi = np.arctan2(n[:, 1], n[:, 0])
    phi[r == 0] = 0
    theta[r == 0] = 0
    return phi, theta, r


def uniform_grid_sphere(n_theta, n_phi=None):
    """
    Generates uniformly distributed grid over the surface of sphere:

    :param n_theta: number of theta grids
    :param n_phi: number of phi_grids
    :return: grid of theta and phi
    """
    if n_phi is None:
        n_phi = n_theta

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    cos_theta = np.linspace(-1, 1, n_theta)
    theta = np.arccos(cos_theta)

    pairs = np.array([(t, p) for t in theta for p in phi])
    theta, phi = pairs.T
    return theta, phi


def uniform_sampling_sphere(n_samples):
    """
    Generates uniformly distributed samples over the surface of sphere:

    :param n_samples: number of theta grids
    :return: samples of theta and phi
    """

    phi = np.random.uniform(0, 2 * np.pi, n_samples)

    cos_theta = np.random.uniform(-1, 1, n_samples)
    theta = np.arccos(cos_theta)
    return theta, phi


def plot_response_function(response, shells, idx_shells, bvecs, res=40, maxs=5):
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from matplotlib import cm
    fig = plt.figure(figsize=(12, 8))

    for shell_idx in np.arange(1, 1):
        sig = - np.log(response[idx_shells == shell_idx]) / shells[shell_idx].bval
        dirs = bvecs[idx_shells == shell_idx]
        dirs = np.vstack((dirs, -dirs))
        sig = np.append(sig, sig)
        _, phi, theta = cart2spherical(dirs)

        p, t = np.meshgrid(np.linspace(-np.pi, np.pi, res), np.linspace(0, np.pi, res))
        s = griddata((phi, theta), sig, (p, t), method='nearest')
        x = np.sin(p) * np.cos(t)
        y = np.sin(p) * np.sin(t)
        z = np.cos(p)

        ax = fig.add_subplot(2, 2, shell_idx, projection='3d')
        plt.set_cmap('jet')
        plot = ax.plot_surface(
            x, y, z, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False, alpha=.8, facecolors=cm.jet(s / maxs))
        plot.set_clim(vmin=0, vmax=maxs)
        fig.colorbar(plot, shrink=0.5, aspect=2)
        plt.title(f"bval={shells[shell_idx].bval}")
        ax.set_xlim3d(-maxs, maxs), ax.set_ylim3d(-maxs, maxs), ax.set_zlim3d(-maxs, maxs)
        ax.view_init(azim=0, elev=0)
    plt.show()


@numba.jit(nopython=True)
def find_t(l1, l2, l3):
    """
    Helper function for hyp_Sapprox

    Args:
        l1: float
            negative first eigenvalue
        l2: float
            negative second eigenvalue
        l3: float
            negative third eigenvalue

    Returns: float
        I guess the return value is t

    """
    a3 = l1 * l2 + l2 * l3 + l1 * l3
    a2 = 1.5 - l1 - l2 - l3
    a1 = a3 - l1 - l2 - l3
    a0 = 0.5 * (a3 - 2 * l1 * l2 * l3)

    inv3 = 1. / 3.
    p = (a1 - a2 * a2 * inv3) * inv3
    q = (-9 * a2 * a1 + 27 * a0 + 2 * a2 * a2 * a2) / 54
    d = q * q + p * p * p
    offset = a2 * inv3

    if d > 0:
        ee = np.sqrt(d)
        z1 = (-q + ee) ** inv3 + (-q - ee) ** inv3 - offset
        z2 = z1
        z3 = z1
    elif d < 0:
        ee = np.sqrt(-d)
        angle = 2 * inv3 * np.arctan(ee / (np.sqrt(q * q + ee * ee) - q))
        sqrt3 = np.sqrt(3.)
        c = np.cos(angle)
        s = np.sin(angle)
        ee = np.sqrt(-p)
        z1 = 2 * ee * c - offset
        z2 = -ee * (c + sqrt3 * s) - offset
        z3 = -ee * (c - sqrt3 * s) - offset
    else:
        tmp = -q ** inv3
        z1 = 2 * tmp - offset
        if p != 0 or q != 0:
            z2 = tmp - offset
        else:
            z2 = z1
        z3 = z2
    if z1 < z2 and z1 < z3:
        return z1
    elif z2 < z3:
        return z2
    else:
        return z3


@numba.jit(nopython=True)
def hyp_sapprox(x):
    """
    Computes 1F1(1/2; 3/2; M) where ``x`` are the eigenvalues from M

    see ``der_hyp_Sapprox`` to only numerically estimate the derivative

    Args:
        x: (3, ) float np.ndarray
            eigenvalues in descending order

    Returns: float
        Result of the hypergeometric function

    """
    if x[0] == 0 and x[1] == 0 and x[2] == 0:
        return 1
    else:
        t = find_t(-x[0], -x[1], -x[2])
        r = 1.
        k2 = 0.
        k3 = 0.
        k4 = 0.

        for idx in range(3):
            r /= np.sqrt(-x[idx] - t)
            k2 += 0.5 * (x[idx] + t) ** -2
            k3 -= (x[idx] + t) ** -3
            k4 += 3 * (x[idx] + t) ** -4

        tau = k4 / (8 * k2 * k2) - 5 * k3 * k3 / (24 * k2 ** 3)
        c1 = (np.sqrt(2 / k2) * np.pi * r * np.exp(-t)) * np.exp(tau) / (4 * np.pi)
        return c1


def simulate_signal(model, acq, params):
    """
    simulates diffusion MRI signals for test from the specified model
        :param model: function object from diffusion models
        :param acq: Acquisition object containing acquisition parameters
        :param params: dictionary of model parameter
        :return: diffusion signal (ndirs, 1)
    """
    n_directions = acq.bvecs.shape[0]
    signal = np.zeros(n_directions)
    for shell_idx, single_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        acquisition_params = {'bval': single_shell.bval, 'bvec': acq.bvecs[dir_idx, :], 's0': 1}
        signal[dir_idx] = model(**acquisition_params, **params)

    return signal


def bench_decorator(model):
    def func(acq, sph_degree, noise_level, **params):
        sig = simulate_signal(model, acq, params)
        noise = np.random.randn(*sig.shape) * noise_level
        sm = summary_measures.compute_summary(sig + noise, acq, sph_degree=sph_degree)
        sm = np.stack(list(sm.values()))
        return sm
    func.__name__ = model.__name__
    return func
