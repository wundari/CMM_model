import numpy as np
from numba import njit, prange


# %%
@njit(fastmath=True)
def compute_disp_tuning_corr(x, y):
    # standardize
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()

    x_standardized = (x - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    # compute correlation
    corr = np.dot(x_standardized, y_standardized) / (len(x) * len(y))

    return corr


# %%
@njit(parallel=True)
def _compute_disp_tuning_corr_layer(disp_tuning_layer):
    """
    For each neuron in a network layer, compute the Pearson correlation
    between non-crds disparity tuning and cRDS disparity tuning.

    Args:
        disp_tuning_layer (np.array [n_neurons, dotMatch, disp]):
            disparity tuning of neurons in a DNN layer for every
            dotMatch

    Returns:
        disp_tuning_corr_layer (np.array [n_neurons, n_dotMatch - 1]):
            the Pearson correlation between non-crds disparity tuning
            and cRDS disparity tuning
    """
    # disp_tuning_layer [n_neurons, len(dotMatch_list), len(disp_ct_pix_list)]

    n_neurons, n_dotMatch, _ = disp_tuning_layer.shape

    disp_tuning_corr_layer = np.zeros((n_neurons, n_dotMatch - 1), dtype=np.float32)
    for i in prange(n_neurons):
        # get disparity tuning for crds
        disp_tuning_crds = disp_tuning_layer[i, -1]
        for j in range(n_dotMatch - 1):
            disp_tuning = disp_tuning_layer[i, j]
            # compute correlation between non-crds disp_tuning vs disp_tuning_crds
            corr = compute_disp_tuning_corr(disp_tuning, disp_tuning_crds)

            disp_tuning_corr_layer[i, j] = corr

    return disp_tuning_corr_layer


# %%
@njit(fastmath=True)
def compute_auc(y, dx):
    auc = 0
    for i in range(len(y) - 1):
        y0 = y[i]
        y1 = y[i + 1]

        auc += 0.5 * (y0 + y1) * dx

    return auc


@njit(parallel=True)
def compute_auc_disp_tuning_corr_layer(disp_tuning_corr_layer, dx):
    """
    compute the area under curve (auc) of disparity tuning which is the
    function of dotMatch.
    The auc is computed for the above and below corr = 0.

    Args:
        disp_tuning_corr_layer (np.array [n_neurons, n_dotMatch - 1]):
            the Pearson correlation between non-crds disparity tuning
            and cRDS disparity tuning
        dx (float): the width (height) of trapezoids for calculating auc

    Returns:
        auc_neg_layer <np.array [n_neurons]>: the auc for corr < 0
        auc_pos_layer <np.array [n_neurons]>: the auc for corr > 0
    """
    n_neurons = disp_tuning_corr_layer.shape[0]

    auc_neg_layer = np.zeros(n_neurons, dtype=np.float32)
    auc_pos_layer = np.zeros(n_neurons, dtype=np.float32)
    for n in prange(n_neurons):
        disp_tuning_neuron = disp_tuning_corr_layer[n]

        auc_neg = 0  # auc below corr = 0
        auc_pos = 0  # auc above corr = 0
        for i in range(len(disp_tuning_neuron) - 1):
            y0 = disp_tuning_neuron[i]
            y1 = disp_tuning_neuron[i + 1]
            # check if the correlation is below or above 0
            if (y0 < 0) and (y1 < 0):
                auc_neg += 0.5 * (y0 + y1) * dx

            elif (y0 > 0) and (y1 > 0):
                auc_pos += 0.5 * (y0 + y1) * dx

            elif (y0 < 0) and (y1 > 0):
                auc_neg -= 0.5 * dx * (y0**2) / (y1 - y0)
                auc_pos += 0.5 * dx * (y1**2) / (y1 - y0)

            elif (y0 > 0) and (y1 < 0):
                auc_neg -= 0.5 * dx * (y1**2) / (y0 - y1)
                auc_pos += 0.5 * dx * (y0**2) / (y0 - y1)

        auc_neg_layer[n] = auc_neg
        auc_pos_layer[n] = auc_pos

    return auc_neg_layer, auc_pos_layer


# %%
@njit
def get_bin_edges(data, bins):
    a_min = data.min()
    a_max = data.max()
    bin_edges = np.zeros((bins + 1), dtype=np.float32)
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # avoid roundoff error on last point
    return bin_edges


@njit
def compute_bin_index(x, bin_edges):
    # assume uniform bin
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    if x == a_max:
        return n - 1

    bin_idx = int(n * (x - a_min) / (a_max - a_min))

    if bin_idx < 0 or bin_idx >= n:
        return None
    else:
        return bin_idx


@njit
def histogram_1d(data, bins):
    hist = np.zeros((bins), dtype=np.intp)
    bin_edges = get_bin_edges(data, bins)

    for x in data.flat:
        bin_idx = compute_bin_index(x, bin_edges)
        if bin_idx is not None:
            hist[int(bin_idx)] += 1

    return hist, bin_edges


@njit
def histogram2d(data1, data2, bins):
    hist = np.zeros((bins, bins), dtype=np.intp)
    bin_edges1 = get_bin_edges(data1, bins)
    bin_edges2 = get_bin_edges(data2, bins)

    for x, y in zip(data1.flat, data2.flat):
        bin_idx1 = compute_bin_index(x, bin_edges1)
        bin_idx2 = compute_bin_index(y, bin_edges2)
        if (bin_idx1 is not None) and (bin_idx2 is not None):
            hist[int(bin_idx1), int(bin_idx2)] += 1

    return hist, bin_edges1, bin_edges2


@njit
def compute_MI_disp_tuning_neuron(x, y, bins):
    """
    Compute mutual information (MI) between two disparity tuning

    Args:
        x ([type]): [description]
        y ([type]): [description]
        bins ([type]): [description]

    Returns:
        [type]: [description]
    """

    hist_2d, _, _ = histogram2d(x.ravel(), y.ravel(), bins)

    # joint prob distribution
    pxy = hist_2d / np.sum(hist_2d)

    # marginal distribution
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]

    nonzeros_idx = np.argwhere(pxy > 0)
    MI = 0
    for i, j in nonzeros_idx:
        MI += pxy[i, j] * np.log(pxy[i, j] / px_py[i, j])

    return MI


@njit(parallel=True)
def compute_MI_disp_tuning_layer(disp_tuning_layer, bins):
    n_neurons, n_dotMatch = disp_tuning_layer.shape[0:2]

    MI_layer = np.zeros((n_neurons, n_dotMatch - 1), dtype=np.float32)
    for n in prange(n_neurons):
        disp_tuning_crds = disp_tuning_layer[n, -1]
        for dm in range(n_dotMatch - 1):
            disp_tuning_non_crds = disp_tuning_layer[n, dm]

            MI = compute_MI_disp_tuning_neuron(
                disp_tuning_crds, disp_tuning_non_crds, bins
            )

            MI_layer[n, dm] = MI

    return MI_layer
