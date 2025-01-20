from bench import summary_measures


def set_up_GLM(sm, dictionary_of_axes):
    
    """
    Sets up the General Linear Model (GLM) for estimating relationships between variables.

    Args:
        sm (np.ndarray): Summary measure array with shape (number_voxels, summary_measures).
        dictionary_of_axes (dict): Dictionary containing axes of change as keys and corresponding
                                   variable arrays as values.

    Returns:
        y (np.ndarray): The dependent variable array (summary measures).
        X (np.ndarray): The design matrix incorporating the axes of change and baseline.
    """


    summary_measure_array = np.copy(sm)
    number_voxels = summary_measure_array.shape[0]
    y = summary_measure_array

    ## y = X @ beta (voxels x sm  = voxels x change axes @ change axes x sm)

    axes_of_change = [x for x in dictionary_of_axes.keys()]

    X = np.ones(number_voxels)

    # generate design matrix (X)
    for axe in axes_of_change:
        axis_of_change = dictionary_of_axes[axe]
        X = np.column_stack((X, axis_of_change - axis_of_change.mean()))

    return y, X


def get_analytical_betas_and_noise_covariance(y, X, effect_size_dict):
    
    """
    Computes analytical estimates of regression coefficients (betas) and noise covariance.

    Args:
        y (np.ndarray): Dependent variable array (voxels x summary_measures).
        X (np.ndarray): Design matrix (voxels x predictors).
        effect_size_dict (dict): Dictionary mapping axes of change to effect sizes.

    Returns:
        beta_dict (dict): Dictionary of estimated regression coefficients for each axis.
        covar_dict (dict): Dictionary of noise covariance matrices for each axis.
    """
    
    X_least_squares = np.linalg.inv(X.T @ X) @ X.T
    betas_bar = X_least_squares @ y

    axes_of_effect_sizes = [x for x in effect_size_dict.keys()]

    ## rows to select axes

    selection_matrix = np.eye(X.shape[-1])
    selection_dict = {}
    for idx, axe in enumerate(axes_of_effect_sizes):
        selection_dict[axe] = selection_matrix[idx, :]
        selection_dict[axe] = selection_dict[axe][np.newaxis, :]
        #selection_dict[axe] = selection_dict[axe][:, np.newaxis]

    res = (y - X @ betas_bar)

    ## compute the noise covariance and betas
    beta_dict = {}
    covar_dict = {}

    for idx, axe in enumerate(axes_of_effect_sizes):

        if axe in ["baseline"]:

            beta_dict[axe] = (selection_dict[axe] @ betas_bar) * effect_size_dict[axe]

        else:

            covar_dict[axe] = (res.T @ np.diag(
                np.diag(
                    (X_least_squares).T @ (selection_dict[axe]).T @ selection_dict[axe] @ X_least_squares)) @ res) * effect_size_dict[axe] ** 2
            beta_dict[axe] = (selection_dict[axe] @ betas_bar) * effect_size_dict[axe]

    return beta_dict, covar_dict

def continuous_change(summary_measure_array, dictionary_of_axes):
    
    """
    Estimates continuous changes in summary measures across specified axes of change.

    Args:
        summary_measure_array (np.ndarray): Summary measure array (number_voxels x summary_measures).
        dictionary_of_axes (dict): Dictionary containing axes of change as keys and corresponding
                                   variable arrays as values.

    Returns:
        baseline_y (np.ndarray): Baseline summary measure (without effects of changes).
        delta_ys (dict): Dictionary of changes in summary measures for each axis.
    """

    number_voxels = summary_measure_array.shape[0]
    y = summary_measure_array

    ## y = X @ beta (voxels x sm  = voxels x change axes @ change axes x sm)

    axes_of_change = [x for x in dictionary_of_axes.keys()]

    X = np.ones(number_voxels)

    # generate design matrix (X)
    for axe in axes_of_change:
        axis_of_change = dictionary_of_axes[axe]

        X = np.column_stack((X, axis_of_change - axis_of_change.mean()))
        # X = np.column_stack((X, axis_of_change))

    # regression

    # betas = np.linalg.pinv(X) @ (y)
    betas = np.linalg.pinv(X) @ (y)

    # designate directions of change

    delta_ys = {}

    for idx, axe in enumerate(axes_of_change):
        delta_ys[axe] = betas[idx + 1, :][np.newaxis, ...]

    # designate baseline
    baseline_y = betas[0, :]
    baseline_y = baseline_y[np.newaxis, ...]

    return baseline_y, delta_ys


def add_noise(data, noise_type='gaussian', SNR=1, S0=1):
    """Add noise to signal

    For Gaussian :
            Snoise = S0 + N(0,sigma)
    For Rician :
            Snoise = sqrt( Real^2 + Imag^2  )
            where Real = S0+N(0,sigma) and Imag=N(0,sigma)
    In both case SNR is defined as SNR = S0/sigma

    Args:
        data : array-like
        noise_type : str
                     accepted values are 'gaussian', 'rician', or 'none'
        SNR : float
        S0  : float
    """
    if noise_type not in ['none', 'gaussian', 'rician']:
        raise (Exception(f'Unknown noiise type {noise_type}'))
    if noise_type == 'none':
        return data + 0.
    else:
        sigma = S0 / SNR
        if noise_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
            return data + noise
        elif noise_type == 'rician':
            noise_r = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
            noise_i = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
            return np.sqrt((data + noise_r) ** 2 + noise_i ** 2)


## === for lower amount of voxels (not enough for bootstrapping; used for numerical simulations w/ limited substrates === ##
# For each substrate: 1) add noise to it, 2) fit summary measure, 3) append it to make the dependent matrix.
# For each new noisy summary measure, take it perform high amount of iterations to get a a beta for each. Then do the noise covariance.

def fit_shm_with_noise(signal, bvalues, gradient_directions, shm_degree, SNR, noise_type="gaussian"):
    """
    Fits spherical harmonics to noisy diffusion MRI signal data.

    Args:
        signal (np.ndarray): Input signal data.
        bvalues (np.ndarray): Array of b-values.
        gradient_directions (np.ndarray): Array of gradient directions.
        shm_degree (int): Degree of spherical harmonics to fit.
        SNR (float): Signal-to-noise ratio.
        noise_type (str): Type of noise ('gaussian' or 'rician').

    Returns:
        np.ndarray: Summary measures fitted to the noisy signal.
    """    
    noisy_signal = add_noise(signal, noise_type=noise_type, SNR=SNR, S0=1)
    sm = summary_measures.fit_shm(noisy_signal, bvalues, gradient_directions, shm_degree=shm_degree)

    return sm


def estimate_continuous_noise_covariance_for_low_voxels(signals, substrates, dictionary_of_axes, sm_names, iterations,
                                                        bvalues, gradient_directions, SNR, effect_size_dict,
                                                        sm="shm", shm_degree=2, noise_type="gaussian",
                                                        neglect_b0=False):
    """
    Estimates noise covariance for continuous changes in low-voxel scenarios.

    Args:
        signals (dict): Dictionary of signals for each substrate.
        substrates (list): List of substrate identifiers.
        dictionary_of_axes (dict): Axes of change with associated variables.
        sm_names (list): Names of summary measures.
        iterations (int): Number of iterations for covariance estimation.
        bvalues (np.ndarray): Array of b-values.
        gradient_directions (np.ndarray): Gradient directions.
        SNR (float): Signal-to-noise ratio.
        effect_size_dict (dict): Effect sizes for axes of change.
        sm (str): Type of summary measure ('shm').
        shm_degree (int): Degree of spherical harmonics.
        noise_type (str): Type of noise ('gaussian' or 'rician').
        neglect_b0 (bool): Whether to neglect b0 terms.

    Returns:
        dict: Noise covariance matrices for each axis of change.
    """


    dictionary_of_betas = {}
    dictionary_of_noise_covar = {}

    # get the noise covariance for each beta

    axes_of_change = [x for x in dictionary_of_axes.keys()]

    for axe in axes_of_change:
        dictionary_of_betas[axe] = np.zeros((iterations, len(sm_names)))
        dictionary_of_noise_covar[axe] = np.zeros((iterations, len(sm_names)))

    for i in range(iterations):

        if sm == "shm":

            for j, substrate in enumerate(substrates):

                noisy_sm = fit_shm_with_noise(signals[substrate], bvalues, gradient_directions, shm_degree, SNR,
                                              noise_type=noise_type)

                # concatenate

                if j == 0:

                    grp_sm = noisy_sm

                else:

                    grp_sm = np.concatenate((grp_sm, noisy_sm), axis=0)

            # infer changes across all substrate
            baseline_y, delta_ys = continuous_change(grp_sm, dictionary_of_axes)

            for axe in axes_of_change:  # iterate over all change axes and compute beta for deriving covariance

                #print("{} : {}".format(axe,delta_ys[axe].shape))

                if neglect_b0:
                    dictionary_of_betas[axe][i, :] = delta_ys[axe][:,1:] * effect_size_dict[axe] #to neglect b0
                else:
                    dictionary_of_betas[axe][i, :] = delta_ys[axe] * effect_size_dict[axe]

    for axe in axes_of_change:
        # print(dictionary_of_betas[axe].shape)

        dictionary_of_noise_covar[axe] = np.cov(dictionary_of_betas[axe].T)

    return dictionary_of_noise_covar


def fit_group_shm_with_noise_and_continuous_noise_covariance(signals, substrates, dictionary_of_axes, sm_names, bvalues,
                                                             gradient_directions, shm_degree, SNR, effect_size_dict, iterations=100,
                                                             noise_type="gaussian",
                                                             neglect_b0=False):
    """
    Fits spherical harmonics to group-level signals with added noise and estimates noise covariance matrices.

    Args:
        signals (dict): Dictionary of signal data for each substrate.
        substrates (list): List of substrate identifiers.
        dictionary_of_axes (dict): Axes of change and corresponding variables.
        sm_names (list): Names of summary measures.
        bvalues (np.ndarray): Array of b-values.
        gradient_directions (np.ndarray): Array of gradient directions.
        shm_degree (int): Degree of spherical harmonics for fitting.
        SNR (float): Signal-to-noise ratio.
        effect_size_dict (dict): Effect sizes for axes of change.
        iterations (int): Number of iterations for noise covariance estimation.
        noise_type (str): Type of noise ('gaussian' or 'rician').
        neglect_b0 (bool): Whether to exclude b0 term in the analysis.

    Returns:
        tuple: 
            - grp_sm (np.ndarray): Group-level summary measures.
            - dictionary_of_noise_covar (dict): Noise covariance matrices for each axis of change.
            - noisy_signal_dictionary (dict): Noisy signal data for each substrate.
    """

    noisy_signal_dictionary = {}

    for j, substrate in enumerate(substrates):  # iterate over substrates to generate SM

        signal = signals[substrate]

        # add noise to each voxel

        noisy_signal = add_noise(signal, noise_type=noise_type, SNR=SNR, S0=1)
        noisy_signal_dictionary[substrate] = noisy_signal

        sm = summary_measures.fit_shm(noisy_signal, bvalues, gradient_directions, shm_degree=shm_degree)

        # concatenate

        if j == 0:

            grp_sm = sm

        else:

            grp_sm = np.concatenate((grp_sm, sm), axis=0)

    if neglect_b0:
        grp_sm = grp_sm[:,1:]

    # for noise covariance
    dictionary_of_noise_covar = estimate_continuous_noise_covariance_for_low_voxels(noisy_signal_dictionary,
                                                                                    substrates,
                                                                                    dictionary_of_axes,
                                                                                    sm_names,
                                                                                    iterations,
                                                                                    bvalues,
                                                                                    gradient_directions,
                                                                                    SNR,
                                                                                    effect_size_dict=effect_size_dict,
                                                                                    sm="shm",
                                                                                    shm_degree=2,
                                                                                    noise_type=noise_type,
                                                                                    neglect_b0=neglect_b0)

    return grp_sm, dictionary_of_noise_covar, noisy_signal_dictionary


## === for high amount of voxels (enough for bootstrapping; used for analytical signals) === ##

def estimate_continuous_noise_covariance_from_data_with_effect_size(signals,
                                                                    dictionary_of_axes,
                                                                    sm_names,
                                                                    bvalues,
                                                                    bvecs,
                                                                    samples=200,
                                                                    iterations=100,
                                                                    shm_degree=2,
                                                                    effect_size=1):
    """
    Estimates noise covariance for continuous changes in scenarios with sufficient voxel samples.

    Args:
        signals (np.ndarray): Input signal data.
        dictionary_of_axes (dict): Axes of change and corresponding variables.
        sm_names (list): Names of summary measures.
        bvalues (np.ndarray): Array of b-values.
        bvecs (np.ndarray): Array of gradient directions.
        samples (int): Number of voxel samples per iteration.
        iterations (int): Number of iterations for noise covariance estimation.
        shm_degree (int): Degree of spherical harmonics for fitting.
        effect_size (float): Scaling factor for effect size.

    Returns:
        dict: Noise covariance matrices for each axis of change.
    """

    dictionary_of_betas = {}
    dictionary_of_noise_covar = {}

    # get the noise covariance for each beta

    axes_of_change = [x for x in dictionary_of_axes.keys()]

    for axe in axes_of_change:
        dictionary_of_betas[axe] = np.zeros((iterations, len(sm_names)))
        dictionary_of_noise_covar[axe] = np.zeros((iterations, len(sm_names)))

    for i in range(iterations):

        indices_to_sample = np.random.choice(signals.shape[0], samples, replace=True)

        ## === Sample diffusion signal and summarise === ##

        sampled_signals = signals[indices_to_sample, :]  # randomly sample diffusion signal

        sampled_sm = summary_measures.fit_shm(sampled_signals, bvalues, bvecs, shm_degree=shm_degree)

        ## === Sample from axes of chage === ##

        sampled_dictionary_of_axes = {}

        for axe in axes_of_change:
            sampled_dictionary_of_axes[axe] = dictionary_of_axes[axe][indices_to_sample]

        ## === Infer change based on sampled data === ##

        baseline_y, delta_ys = continuous_change(sampled_sm, sampled_dictionary_of_axes)

        for axe in axes_of_change:  # iterate over all change axes and compute beta for deriving covariance

            dictionary_of_betas[axe][i, :] = delta_ys[axe] * effect_size

    for axe in axes_of_change:
        dictionary_of_noise_covar[axe] = np.cov(dictionary_of_betas[axe].T)

    return dictionary_of_noise_covar


def estimate_continuous_noise_covariance_and_betas(signals,
                                                   dictionary_of_axes,
                                                   sm_names,
                                                   bvalues,
                                                   bvecs,
                                                   samples=200,
                                                   iterations=100,
                                                   shm_degree=2,
                                                   effect_size_dict=None,
                                                   neglect_b0=False):
    """
    Estimates regression coefficients (betas) and noise covariance matrices for continuous changes.

    Args:
        signals (np.ndarray): Input signal data.
        dictionary_of_axes (dict): Axes of change and corresponding variables.
        sm_names (list): Names of summary measures.
        bvalues (np.ndarray): Array of b-values.
        bvecs (np.ndarray): Array of gradient directions.
        samples (int): Number of voxel samples per iteration.
        iterations (int): Number of iterations for covariance estimation.
        shm_degree (int): Degree of spherical harmonics for fitting.
        effect_size_dict (dict, optional): Effect sizes for axes of change. Defaults to uniform effect sizes.
        neglect_b0 (bool): Whether to exclude b0 terms in the analysis.

    Returns:
        tuple:
            - dictionary_of_betas (dict): Regression coefficients for each axis of change.
            - dictionary_of_noise_covar (dict): Noise covariance matrices for each axis of change.
    """

    dictionary_of_betas = {}
    dictionary_of_noise_covar = {}

    # set the baseline for the beta

    dictionary_of_betas["baseline"] = np.zeros((iterations, len(sm_names)))

    # get the noise covariance for each beta

    axes_of_change = [x for x in dictionary_of_axes.keys()]

    if effect_size_dict is None:  # set effect size of 1
        effect_size_dict = {}
        for axe in axes_of_change:
            effect_size_dict[axe] = 1

    for axe in axes_of_change:
        dictionary_of_betas[axe] = np.zeros((iterations, len(sm_names)))
        dictionary_of_noise_covar[axe] = np.zeros((iterations, len(sm_names)))

    for i in range(iterations):

        indices_to_sample = np.random.choice(signals.shape[0], samples, replace=True)

        ## === Sample diffusion signal and summarise === ##

        sampled_signals = signals[indices_to_sample, :]  # randomly sample diffusion signal

        sampled_sm = summary_measures.fit_shm(sampled_signals, bvalues, bvecs, shm_degree=shm_degree)

        ## === Sample from axes of chage === ##

        sampled_dictionary_of_axes = {}

        for axe in axes_of_change:
            sampled_dictionary_of_axes[axe] = dictionary_of_axes[axe][indices_to_sample]

        ## === Infer change based on sampled data === ##

        baseline_y, delta_ys = continuous_change(sampled_sm, sampled_dictionary_of_axes)

        #print(baseline_y.shape)

        if neglect_b0:
            dictionary_of_betas["baseline"][i, :] = baseline_y[:,1:]
        else:
            dictionary_of_betas["baseline"][i, :] = baseline_y

        for axe in axes_of_change:  # iterate over all change axes and compute beta for deriving covariance

            if neglect_b0:
                dictionary_of_betas[axe][i, :] = delta_ys[axe][:, 1:] * effect_size_dict[axe]  # to neglect b0
            else:
                dictionary_of_betas[axe][i, :] = delta_ys[axe] * effect_size_dict[axe]

    for axe in axes_of_change:
        dictionary_of_noise_covar[axe] = np.cov(dictionary_of_betas[axe].T)  # estimate noise covariance

    return dictionary_of_betas, dictionary_of_noise_covar


## === plotting === ##

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def default_template():
    """
    Sets the default plotting template for matplotlib.

    Modifies matplotlib's global parameters for consistent figure appearance, such as font style,
    background colors, and tick sizes.
    """

    mpl.rcParams["figure.facecolor"] = "FFFFFF"
    mpl.rcParams["axes.facecolor"] = "FFFFFF"
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.labelcolor'] = "000000"
    mpl.rcParams['axes.edgecolor'] = "000000"

    mpl.rcParams['ytick.color'] = "000000"
    mpl.rcParams['xtick.color'] = "000000"
    mpl.rcParams['ytick.labelsize'] = 50
    mpl.rcParams['xtick.labelsize'] = 50


def plot_probs(probs, free_params, title, ymax=0.4, xticksize=10, figsize=None):

    """
    Plots bar charts to visualise probabilities associated with different free parameters.

    Args:
        probs (np.ndarray): Array of probabilities for each parameter.
        free_params (list): List of parameter names corresponding to the probabilities.
        title (str): Title for the plot.
        ymax (float): Maximum value for the y-axis. Defaults to 0.4.
        xticksize (int): Font size for x-axis tick labels. Defaults to 10.
        figsize (tuple, optional): Figure size. Defaults to (20, 12).

    Returns:
        tuple:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (matplotlib.axes.Axes): The axes object.
    """

    if figsize is None:
        figsize=(20,12)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_ylim(0, ymax)

    colors = {"None": "c",
              "nochange": "c",
              "s_in": 'b',
              "s_ex": 'k',
              "s_dot": "brown",
              "s_sphere": "orchid",
              "d_sphere": 'purple',
              "s_iso": 'brown',
              "d_iso": 'lightseagreen',
              "rad_sphere": 'gold',
              "odi": 'r',
              "d_a_in": 'g',
              "d_a_ex": 'm',
              "d_r_ex": 'y',
              "tortuosity": 'y',
              "-s_dot + s_sphere": "darkorange",
              "-0.7*s_dot + 0.7*s_sphere": "darkorange",
              "-0.6*s_dot + 0.8*s_sphere": "darkorange",
              "-0.7*s_in + 0.7*s_ex": "coral",
              "-0.7*s_ex + 0.7*s_in": "coral",
              "-0.7*s_dot + 0.7*s_in": "indigo",
              "-0.7*s_dot + 0.7*s_ex": "black",
              "-0.7*s_ex + 0.7*s_sphere": "pink",
              "-0.7*s_in + 0.7*s_sphere": "chocolate",
              "-0.7*s_iso + 0.7*s_sphere": "darkorange",
              "-0.7*s_iso + 0.7*s_in": "indigo",
              "-0.7*s_iso + 0.7*s_ex": "black",
              "-0.7*s_iso + 0.7*s_dot": "brown",
              }

    notation = {"None": "$None$",
                "nochange": "$None$",
                "s_in": "$f_{in}$",
                "s_ex": "$f_{ex}$",
                "s_dot": "$f_{dot}$",
                "s_iso": "$f_{ball}$",
                "s_sphere": "$f_{sph}$",
                "d_sphere": '$D_{sph}$',
                "rad_sphere": '$R_{sph}$',
                "d_iso": '$D_{ball}$',
                "odi": "$ODI$",
                "d_a_in": "$D_{a,in}$",
                "d_a_ex": "$D_{a,ex}$",
                "tortuosity": r"$\tau$",
                "d_r_ex": "$D_{r,ex}$",
                "-s_dot + s_sphere": r"$f_{sph}$-$f_{dot}$",
                "-0.7*s_dot + 0.7*s_sphere": r"$f_{dot}$-$f_{sph}$",
                "-0.7*s_dot + 0.7*s_ex": r"$f_{dot}$-$f_{ex}$",
                "-0.7*s_dot + 0.7*s_in": r"$f_{dot}$-$f_{in}$",
                "-0.6*s_dot + 0.8*s_sphere": r"$f_{sph}$-$f_{dot}$",
                "-0.7*s_in + 0.7*s_ex": r"$f_{ex}$-$f_{in}$",
                "-0.7*s_ex + 0.7*s_in": r"$f_{in}$-$f_{ex}$",
                "-0.7*s_ex + 0.7*s_sphere": r"$f_{ex}$-$f_{sph}$",
                "-0.7*s_in + 0.7*s_sphere": r"$f_{in}$-$f_{sph}$",
                "-s_iso + s_sphere": r"$f_{sph}$-$f_{ball}$",
                "-0.7*s_iso + 0.7*s_sphere": r"$f_{sph}$-$f_{ball}$",
                "-0.7*s_iso + 0.7*s_ex": r"$f_{ex}$-$f_{ball}$",
                "-0.7*s_iso + 0.7*s_in": r"$f_{in}$-$f_{ball}$",
                "-0.7*s_iso + 0.7*s_dot": r"$f_{dot}$-$f_{ball}$"
                }


    probs = np.squeeze(probs)

    for i, x in enumerate(free_params):
        ax.bar(notation[x], probs[i], color=colors[x], width=0.80)

    ax.set_title(title, fontsize=15)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', which='major', labelsize=xticksize)
    ax.tick_params(axis='y', which='major', labelsize=18)

    fig.tight_layout()

    return fig, ax


def plot_changes(names, delta, error=None, color="blue"):

    """
    Plots bar charts to visualize changes (e.g., effect sizes) with optional error bars.

    Args:
        names (list): List of parameter names.
        delta (np.ndarray): Array of changes (effect sizes) for each parameter.
        error (np.ndarray, optional): Array of error values for each parameter. Defaults to None.
        color (str): Color of the bars. Defaults to "blue".
    """
    
    if error is None:
        plt.bar(x=names, height=np.squeeze(delta))

    else:
        plt.bar(x=names, height=np.squeeze(delta), yerr=np.squeeze(error), align='center', alpha=0.5,
                ecolor='black', color=color, capsize=10)


## === models that you added === ##


# Constants:
gamma = 2.6752218744 * 1e8
# gamma = 42.577478518*1e6     # [sec]^-1 * [T]^-1
gamma_ms = gamma * 1e-3  # [ms]^-1 *[T]^-1

# From Camino source
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
    Computes the gaussian phase distribution (GPD) sum for restricted diffusion in a sphere.

    Args:
        am_r (np.ndarray): Array of scaled Bessel function roots divided by sphere radius.
        pulse_duration (np.ndarray): Gradient pulse duration (ms).
        diffusion_time (np.ndarray): Diffusion time (ms).
        diffusivity (np.ndarray): Diffusion coefficient inside the sphere.
        radius (np.ndarray): Sphere radius (μm).

    Returns:
        np.ndarray: Computed GPD sum for restricted diffusion.
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
    Predicts signal of water restricted in  a sphere.
    Args:
        pulse_duration: applied gradient time (in ms)
        diffusion_time: diffusion time (in ms)
        G: gradient strength (in ?)
        radius: radius of sphere (in um)
        diffusivity: diffusion coefficient inside the sphere
        S0: base signal at b=0
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


from bench.diffusion_models import bingham_zeppelin


def dot_sphere_watson_stick_zeppelin(bval,
                                     bvec,
                                     pulse_duration,
                                     diffusion_time,
                                     G,
                                     s_in,
                                     s_ex,
                                     s_dot,
                                     s_sphere,
                                     odi,
                                     d_a_in,
                                     d_a_ex,
                                     d_sphere,
                                     rad_sphere,
                                     d_r_ex,
                                     theta=0.,
                                     phi=0.,
                                     s0=1.):
    """
    Simulates diffusion signals from a combination of dot, sphere, Watson-dispersed sticks, and zeppelin models.

    Args:
        bval (np.ndarray): Array of b-values.
        bvec (np.ndarray): Array of gradient directions.
        pulse_duration (float): Gradient pulse duration (ms).
        diffusion_time (float): Diffusion time (ms).
        G (float): Gradient strength.
        s_in (float): Signal fraction for intracellular space.
        s_ex (float): Signal fraction for extracellular space.
        s_dot (float): Signal fraction for isotropic component.
        s_sphere (float): Signal fraction for restricted sphere.
        odi (float): Orientation dispersion index.
        d_a_in (float): Axial diffusivity (intracellular).
        d_a_ex (float): Axial diffusivity (extracellular).
        d_sphere (float): Diffusivity inside spheres.
        rad_sphere (float): Sphere radius (μm).
        d_r_ex (float): Radial diffusivity (extracellular).
        theta (float): Azimuthal angle (rad).
        phi (float): Polar angle (rad).
        s0 (float): Base signal intensity.

    Returns:
        np.ndarray: Simulated diffusion signal.
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

    s_dot_full = np.repeat(s_dot[:, np.newaxis], a_ext.shape[-1], axis=1)

    return (a_int + a_ext + a_sphere + s_dot_full) * s0


def dot_sphere_watson_stick_zeppelin_without_s_ex(bval,
                                                  bvec,
                                                  pulse_duration,
                                                  diffusion_time,
                                                  G,
                                                  s_in,
                                                  s_dot,
                                                  s_sphere,
                                                  odi,
                                                  d_a_in,
                                                  d_a_ex,
                                                  d_sphere,
                                                  rad_sphere,
                                                  d_r_ex,
                                                  theta=0.,
                                                  phi=0.,
                                                  s0=1.):
    """
    Simulates diffusion signals from a model combining dot, sphere, Watson-dispersed sticks, and zeppelin,
    with the extracellular fraction inferred from other components. Assumes a "fixed" extra-axonal signal fraction.

    Args:
        bval (np.ndarray): Array of b-values.
        bvec (np.ndarray): Array of gradient directions.
        pulse_duration (float): Gradient pulse duration (ms).
        diffusion_time (float): Diffusion time (ms).
        G (float): Gradient strength.
        s_in (float): Signal fraction for intracellular space.
        s_dot (float): Signal fraction for isotropic component.
        s_sphere (float): Signal fraction for restricted sphere.
        odi (float): Orientation dispersion index.
        d_a_in (float): Axial diffusivity (intracellular).
        d_a_ex (float): Axial diffusivity (extracellular).
        d_sphere (float): Diffusivity inside spheres.
        rad_sphere (float): Sphere radius (μm).
        d_r_ex (float): Radial diffusivity (extracellular).
        theta (float): Azimuthal angle (rad).
        phi (float): Polar angle (rad).
        s0 (float): Base signal intensity.

    Returns:
        np.ndarray: Simulated diffusion signal.
    """

    s_ex = 1 - s_in - s_sphere - s_dot

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

    s_dot_full = np.repeat(s_dot[:, np.newaxis], a_ext.shape[-1], axis=1)

    return (a_int + a_ext + a_sphere + s_dot_full) * s0


## === Modifield summary decorator to include pulse duration and different acquisition parameters === ##

def summary_decorator(model,
                      bval,
                      bvec,
                      pulse_duration,
                      diffusion_time,
                      G,
                      summary_type='sh', shm_degree=2):
    """
    Decorates a diffusion model to add noise and directly calculate a summary measurement. The return function can
    be used like sm = f(micro params, noise_std= default to zero).

    :param bvec: array of b-values
    :param bval: array of b-vectors
    :param model: a diffusion model (function)
    :param summary_type: either 'sh' (spherical harmonics), 'dt' (diffusion tensor model), 'sig' (raw signal)
    :param shm_degree: degree for spherical harmonics
    :return:
        function f that maps microstructural parameters to summary measurements, name of the summary measures
    """
    if summary_type == 'sh':
        def func(noise_std=0.0, **params):
            sig = model(bval, bvec, pulse_duration, diffusion_time, G, **params)
            noise = np.random.randn(*sig.shape) * noise_std
            sm = summary_measures.fit_shm(sig + noise, bval, bvec, shm_degree=shm_degree)
            return sm

        names = summary_measures.summary_names(bval, summarytype='sh', shm_degree=shm_degree)

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
