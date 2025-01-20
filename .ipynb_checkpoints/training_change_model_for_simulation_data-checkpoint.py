#!/usr/bin/env python

"""
This script trains a change model for diffusion MRI (dMRI) data based on acquisition parameters.
It processes acquisition files (.npz) containing b-values, b-vectors, and other parameters, and outputs a trained change model. This particular change model is for simulation data.

Usage:
    python training_change_model_for_simulation_data.py -ad <acq_dir> -num <num_samples> -dc <diff_coeff> -od <output_dir>

Arguments:
    -ad, --acq_dir       Path to the .npz file containing acquisition parameters (required).
    -num, --num_samples  Number of samples to train the model (required).
    -od, --output_dir    Directory to save the trained model (required).

Outputs:
    Trained change model saved in the specified directory
    
"""

import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import argparse
from scipy import stats
import numpy as np
from bench import summary_measures
from bench import change_model


def main():
    p = argparse.ArgumentParser(description="Training extended standard model to describe simulation data")

    p.add_argument('-ad', '--acq_dir',
                   required=True, nargs='+', type=str, metavar='<str>',
                   help='npz file with acquisition params')
    p.add_argument('-num', '--num_samples',
                   required=True, nargs='+', type=int,
                   help='Number of samples to train our model on')
    p.add_argument('-od', '--output_dir',
                   required=True, nargs='+', type=str, metavar='<str>',
                   help='directory to save trained model to')

    # Parse arguments
    args = p.parse_args()
    args.acq_dir = ' '.join(args.acq_dir)
    args.output_dir = ' '.join(args.output_dir)
    output_dir = args.output_dir
    acq_npz = args.acq_dir
    num_samples = np.array(args.num_samples)[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load acquisition parameters
    data = np.load(acq_npz)
    bvals = data["a"]
    bvecs = data["b"]
    pulse_duration = data["c"]
    diffusion_time = data["d"]
    gradient_strength = data["e"]


    # Define priors for the microstructural parameters
    priors = {
        ('s_in', 's_ex', 's_dot', 's_sphere', 's_iso'): sample_signal_with_ball,
        'odi': stats.uniform(loc=0.01, scale=0.50),
        'd_a_in': stats.uniform(loc=0.50, scale=1.5),
        'd_a_ex': stats.uniform(loc=0.50, scale=1.5),
        'd_r_ex': stats.uniform(loc=0.50, scale=1.5),
        'd_iso': stats.uniform(loc=0.50, scale=2.5),
        'd_sphere': stats.uniform(loc=0.50, scale=1.5),
        'rad_sphere': stats.uniform(loc=0.01, scale=10)}

    # Set up the model
    forward_model, sm_names = summary_decorator(
        model=extended_standard_model,
        bval=bvals,
        bvec=bvecs,
        pulse_duration=pulse_duration,
        diffusion_time=diffusion_time,
        G=gradient_strength,
        summary_type='sh',
        neglect_b0=True
    )


    # Train the change model
    tr = change_model.Trainer(forward_model, priors, summary_names=sm_names,
                              change_vecs=[{'odi': 1},
                                           {'d_a_in': 1},
                                           {'d_a_ex': 1},
                                           {'d_r_ex': 1},
                                           {'d_iso': 1},
                                           {'d_sphere': 1},
                                           {'rad_sphere': 1},
                                           {'s_dot': -1 / np.sqrt(2), 's_sphere': 1 / np.sqrt(2)},
                                           {'s_dot': -1 / np.sqrt(2), 's_in': 1 / np.sqrt(2)},
                                           {'s_dot': -1 / np.sqrt(2), 's_ex': 1 / np.sqrt(2)},
                                           {'s_ex': -1 / np.sqrt(2), 's_in': 1 / np.sqrt(2)},
                                           {'s_in': -1 / np.sqrt(2), 's_sphere': 1 / np.sqrt(2)},
                                           {'s_ex': -1 / np.sqrt(2), 's_sphere': 1 / np.sqrt(2)},
                                           {'s_iso': -1 / np.sqrt(2), 's_sphere': 1 / np.sqrt(2)},
                                           {'s_iso': -1 / np.sqrt(2), 's_in': 1 / np.sqrt(2)},
                                           {'s_iso': -1 / np.sqrt(2), 's_ex': 1 / np.sqrt(2)},
                                           {'s_iso': -1 / np.sqrt(2), 's_dot': 1 / np.sqrt(2)}])
    
    ch_mdl = tr.train_ml(n_samples=num_samples, verbose=True, parallel=False)

    # Save the trained model
    ch_mdl.save(file_name="change_model_for_simulation_data", path=output_dir)


## === Other functions === ##

def sample_signal_with_ball(n_samples):
        
    """
    Generate signal fraction parameters for different compartment models and normalise them.

    Parameters:
        n_samples (int): Number of samples to generate.

    Returns:
        tuple: Normalized signal fractions for intra-axonal, extra-axonal, dot-like, spherical, and isotropic compartments.
    
    """

    s_in = stats.truncnorm(loc=.60, scale=.2, a=-.4 / .2, b=1 / 0.2).rvs(n_samples)
    s_ex = stats.truncnorm(loc=.40, scale=.2, a=-.4 / .2, b=1 / 0.2).rvs(n_samples)
    s_sphere = stats.uniform(loc=0, scale=0.10).rvs(n_samples)
    s_iso = stats.uniform(loc=0, scale=0.10).rvs(n_samples)
    s_dot = stats.uniform(loc=0, scale=0.10).rvs(n_samples)
    
    # Normalise signal fractions to ensure their sum equals 1
    norm = (1 - s_sphere - s_dot - s_iso) / (s_in + s_ex)

    return s_in * norm, s_ex * norm, s_dot, s_sphere, s_iso


# Gyromagnetic ratio constants for signal computation
gamma = 2.6752218744 * 1e8
gamma_ms = gamma * 1e-3  # [ms]^-1 *[T]^-1

# Precomputed roots for spherical diffusion calculations, derived from Camino software
#  60 first roots from the equation (am*x)j3/2'(am*x)- 1/2 J3/2(am*x)=0
am = np.array([2.08157597781810, 5.94036999057271, 9.20584014293667,
               12.4044450219020, 15.5792364103872, 18.7426455847748,
               21.8996964794928, 25.0528252809930, 28.2033610039524,
               31.3520917265645, 34.4995149213670, 37.6459603230864,
               40.7916552312719, 43.9367614714198, 47.0813974121542,
               50.2256516491831, 53.3695918204908, 56.5132704621986,
               59.6567290035279, 62.8000005565198, 65.9431119046553,
               69.0860849466452, 72.2289377620154, 75.3716854092873,
               78.5143405319308, 81.6569138240367, 84.7994143922025,
               87.9418500396598, 91.0842274914688, 94.2265525745684,
               97.3688303629010, 100.511065295271, 103.653261271734,
               106.795421732944, 109.937549725876, 113.079647958579,
               116.221718846033, 116.221718846033, 119.363764548757,
               122.505787005472, 125.647787960854, 128.789768989223,
               131.931731514843, 135.073676829384, 138.215606107009,
               141.357520417437, 144.499420737305, 147.641307960079,
               150.783182904724, 153.925046323312, 157.066898907715,
               166.492397790874, 169.634212946261, 172.776020008465,
               175.917819411203, 179.059611557741, 182.201396823524,
               185.343175558534, 188.484948089409, 191.626714721361])


def compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius):
    """
    Compute the Gaussian phase distribution sum (GPDsum) for diffusion in restricted spheres.

    Parameters:
        am_r (ndarray): Precomputed roots divided by the sphere radius.
        pulse_duration (float): Duration of the gradient pulse (ms).
        diffusion_time (float): Total diffusion time (ms).
        diffusivity (float): Diffusion coefficient.
        radius (float): Sphere radius (um).

    Returns:
        ndarray: GPDsum values for each input set of parameters.
    """
    dam = diffusivity * am_r * am_r

    e11 = -dam * pulse_duration

    e2 = -dam * diffusion_time

    dif = diffusion_time - pulse_duration

    e3 = -dam * dif

    plus = diffusion_time + pulse_duration

    e4 = -dam * plus
    nom = 2 * dam * pulse_duration - 2 + (2 * np.exp(e11)) + (2 * np.exp(e2)) - np.exp(e3) - np.exp(e4)

    denom = dam ** 2 * am_r ** 2 * (radius ** 2 * am_r ** 2 - 2)

    return np.sum(nom / denom, 1)


def sphere(pulse_duration, diffusion_time, G, radius, diffusivity, S0):
    """
    Predict the signal attenuation of water restricted within a spherical compartment.

    Parameters:
        pulse_duration (ndarray): Duration of gradient pulses (ms).
        diffusion_time (ndarray): Total diffusion time (ms).
        G (float): Gradient strength (?T).
        radius (ndarray): Sphere radius (um).
        diffusivity (ndarray): Diffusion coefficient inside the sphere.
        S0 (ndarray): Base signal at b=0.

    Returns:
        ndarray: Predicted signal attenuation.
    """
    ## === refactoring === ##

    pulse_duration = np.repeat(pulse_duration[np.newaxis, :], am.shape[0], axis=0)
    pulse_duration = np.repeat(pulse_duration[np.newaxis, :], diffusivity.shape[0], axis=0)

    diffusion_time = np.repeat(diffusion_time[np.newaxis, :], am.shape[0], axis=0)
    diffusion_time = np.repeat(diffusion_time[np.newaxis, :], diffusivity.shape[0], axis=0)

    diffusivity = np.repeat(diffusivity[:, np.newaxis], am.shape[0], axis=1)
    diffusivity = np.repeat(diffusivity[:, :, np.newaxis], diffusion_time.shape[-1], axis=2)

    radius = np.repeat(radius[:, np.newaxis], am.shape[0], axis=1)
    radius = np.repeat(radius[:, :, np.newaxis], diffusion_time.shape[-1], axis=2)

    S0 = np.repeat(S0[:, np.newaxis], diffusion_time.shape[-1], axis=1)

    ## === compute sum === ##

    G_T_per_micron = G * 1e-3 * 1e-6  # [T] * [um]^-1
    am_r = am[:, np.newaxis] / radius

    GPDsum = compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius)

    log_att = -2. * gamma_ms ** 2 * G_T_per_micron ** 2 * GPDsum

    return S0 * np.exp(log_att)


## with 2nd priors
from bench.diffusion_models import bingham_zeppelin, ball


def extended_standard_model(bval,
                            bvec,
                            pulse_duration,
                            diffusion_time,
                            G,
                            s_in,
                            s_ex,
                            s_dot,
                            s_sphere,
                            s_iso,
                            odi,
                            d_a_in,
                            d_a_ex,
                            d_r_ex,
                            d_iso,
                            d_sphere,
                            rad_sphere,
                            theta=0.,
                            phi=0.,
                            s0=1.):
    """
    Compute the extended standard diffusion model combining intra-axonal, extra-axonal, spherical, isotropic, and dot
    compartments.

    Parameters:
        bval (ndarray): Array of b-values.
        bvec (ndarray): Array of b-vectors.
        pulse_duration (float): Duration of gradient pulses (ms).
        diffusion_time (float): Total diffusion time (ms).
        G (float): Gradient strength (?T).
        s_in (float): Signal fraction for intra-axonal compartment.
        s_ex (float): Signal fraction for extra-axonal compartment.
        s_dot (float): Signal fraction for dot compartment.
        s_sphere (float): Signal fraction for spherical compartment.
        s_iso (float): Signal fraction for isotropic compartment.
        odi (float): Orientation dispersion index.
        d_a_in (float): Axial diffusivity for intra-axonal compartment.
        d_a_ex (float): Axial diffusivity for extra-axonal compartment.
        d_r_ex (float): Radial diffusivity for extra-axonal compartment.
        d_iso (float): Isotropic diffusivity.
        d_sphere (float): Diffusivity inside the sphere.
        rad_sphere (float): Radius of the sphere (um).
        theta (float): Polar angle for orientation (default 0).
        phi (float): Azimuthal angle for orientation (default 0).
        s0 (float): Baseline signal intensity (default 1).

    Returns:
        ndarray: Predicted diffusion signal.
    """
    
    a_int = bingham_zeppelin(bval=bval, bvec=bvec, d_a=d_a_in, d_r=0,
                             odi=odi, odi2=odi,
                             psi=0, theta=theta, phi=phi, s0=s_in)

    a_ext = bingham_zeppelin(bval=bval, bvec=bvec, d_a=d_a_ex,
                             d_r=d_r_ex,
                             odi=odi, odi2=odi,
                             psi=0, theta=theta, phi=phi, s0=s_ex)

    a_sphere = sphere(pulse_duration=pulse_duration,
                      diffusion_time=diffusion_time,
                      G=G,
                      radius=rad_sphere,
                      diffusivity=d_sphere,
                      S0=s_sphere)

    a_iso = ball(bval=bval, bvec=bvec, d_iso=d_iso, s0=s_iso)

    s_dot_full = np.repeat(s_dot[:, np.newaxis], a_ext.shape[-1], axis=1)

    return (a_int + a_ext + a_sphere + s_dot_full + a_iso) * s0


def summary_decorator(model,
                      bval,
                      bvec,
                      pulse_duration,
                      diffusion_time,
                      G,
                      summary_type='sh', shm_degree=2, neglect_b0=True):
    """
    Decorate a diffusion model to add noise and directly calculate summary measurements.

    Parameters:
        model (function): A diffusion model function.
        bval (ndarray): Array of b-values.
        bvec (ndarray): Array of b-vectors.
        pulse_duration (float): Duration of gradient pulses (ms).
        diffusion_time (float): Total diffusion time (ms).
        G (float): Gradient strength (?T).
        summary_type (str): Type of summary measurement ('sh', 'dt', or 'sig').
        shm_degree (int): Degree for spherical harmonics (default 2).
        neglect_b0 (bool): Whether to exclude b0 signals from the summary measurements (default True).

    Returns:
        tuple: A function to compute summary measurements and a list of summary measure names.
    """
        
    if summary_type == 'sh':
        if neglect_b0:
            def func(noise_std=0.0, **params):
                sig = model(bval, bvec, pulse_duration, diffusion_time, G, **params)
                noise = np.random.randn(*sig.shape) * noise_std
                sm = summary_measures.fit_shm(sig + noise, bval, bvec, shm_degree=shm_degree)[..., 1:] #here is where you neglect b0
                return sm
            names = summary_measures.summary_names(bval, summarytype='sh', shm_degree=shm_degree)[1:]
            print(names)
        else:
            def func(noise_std=0.0, **params):
                sig = model(bval, bvec, pulse_duration, diffusion_time, G, **params)
                noise = np.random.randn(*sig.shape) * noise_std
                sm = summary_measures.fit_shm(sig + noise, bval, bvec, shm_degree=shm_degree)
                return sm
            names = summary_measures.summary_names(bval, summarytype='sh', shm_degree=shm_degree)
            print(names)

    elif summary_type == 'sig':
        def func(noise_std=0.0, **params):
            sig = model(bval, bvec, pulse_duration, diffusion_time, G, **params)
            noise = np.random.randn(*sig.shape) * noise_std
            return sig + noise

        names = [f'{b:1.3f}-{g[0]:1.3f},{g[1]:1.3f},{g[2]:1.3f}' for b, g in zip(bval, bvec)]
    else:
        raise ValueError(f'Summary measurement type {summary_type} is not defined. Must be shm, dtm or sig.')
    func.__name__ = model.__name__
    return func, names


if __name__ == '__main__':
    main()
