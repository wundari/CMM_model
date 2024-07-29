"""
working dir: bw18_005_2/Codes/Python/CMM
Created Date: 2021-11-24 22:57:28
Author: Bayu G. Wundari
-----
Last Modified: 2022-07-25 13:35:28
Modified By: Bayu G. Wundari

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------

script for simulating cross-correlation and cross-matching with specific 
    parameters in disparity column model for each brain region
"""

import numpy as np

# enable intel cpu optimization for scikit
from sklearnex import patch_sklearn

patch_sklearn()

import math
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import sem
from scipy.optimize import nnls
from scipy.stats import kendalltau, pearsonr
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge, LinearRegression
from timeit import default_timer as timer

from Common.Common import General
from Common.Common import PlotGeneral

from numba import njit, prange, cuda, float32, int8
from numba.cuda.cudadrv import enums


@njit
def _create_square_window(disp_mu_x, f, size_rds_bg_pix):

    ################## convert disparity in deg to voxel ######################
    # disp_mu_in_pixel = 9 # the number of pixels associated with 0.2 deg
    screen_height = 35.5  # screen height (cm)
    cm_per_pix = screen_height / 1024  # screen pixel resolution
    view_dist = 96  # viewing distance (cm)
    h = 2 * view_dist * math.tan((disp_mu_x / 2) * np.pi / 180)
    disp_mu_in_pixel = int(h / cm_per_pix)

    ########################## compute fwhm in pixel #########################
    # compute fwhm, pixel
    sigma = (np.sqrt(np.log(2) / 2) / (np.pi * f)) * (
        (2**1.5 + 1) / (2**1.5 - 1)
    )  # Read, nature 2007
    # sigma = 0.39/f
    # convert to pixel
    fwhm_pix = int(
        2.355 * sigma * size_rds_bg_pix
    )  # ref: https://mathworld.wolfram.com/GaussianFunction.html
    # fwhm_pix = fwhm(f)

    # define square window w
    w = np.zeros((size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8)
    # w = np.zeros((size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8)

    # use np.max to prevent id_row_start < 0
    temp_start = int(size_rds_bg_pix / 2 - fwhm_pix / 2)
    id_row_start = max([0, temp_start])
    # id_row_start = np.int32(size_rds_bg_pix/2 - fwhm_pix/2)
    # w[0, 0] = id_row_start

    # use np.min to prevent id_row_end > size_rds_bg_pix
    temp_end = temp_start + fwhm_pix
    id_row_end = min([size_rds_bg_pix, temp_end])
    # id_row_end = np.int32(id_row_start + fwhm_pix)
    # w[0, 1] = id_row_end + 2 # plus 2 pixel to prevent zero area

    # use np.max to prevent id_col_start < 0
    temp_start = int(size_rds_bg_pix / 2 - fwhm_pix / 2 - disp_mu_in_pixel)
    # temp_start = np.int32(size_rds_bg_pix/2 - fwhm_pix/2 - disp_mu_in_pixel)
    id_col_start = max([0, temp_start])
    # w[1, 0] = id_col_start

    # use np.min to prevent id_col_end > size_rds_bg_pix
    temp_end = temp_start + fwhm_pix
    id_col_end = min([size_rds_bg_pix, temp_end])
    # id_col_end = min([size_rds_bg_pix, temp_end])
    # w[1, 1] = id_col_end + 2 # plus 2 pixel to prevent zero area

    w[id_row_start:id_row_end, id_col_start:id_col_end] = int(1)
    # plt.imshow(w)

    return w


@njit
def _create_window_in_singleVox_singleFreq(
    dispPref_in_vox, size_rds_bg_pix, neurons_per_vox, f
):
    """
    create a batch of square windows in a single voxel for a given
    spatial frequency referenced by the iterator f.

    Parameters
    ----------

    Returns
    -------
    w_batch : [neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
        a batch of square windows.



    Parameters
    ----------
    neurons_per_vox : scalar
        the number of neurons in a voxel.

    dispPref_in_vox : [neurons_in_vox] np.array
        a list of disparity magnitude (deg) contained in a voxel of simulated
        disparity column map

    f : scalar
        spatial frequency in f_batch.

    Returns
    -------
    window_in_vox: <[neurons_per_vox, size_rds_bg[0], size_rds_bg[1]] np.array>
                    RFs contained in a simulated voxel "vox".

    """

    # f_this = self.f_batch[f]

    # get the number of RF in a voxel
    # [size_rds_bg, neurons_per_vox*size_rds_bg]
    window_in_singleVox = np.zeros(
        (neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8
    )

    for n in range(neurons_per_vox):

        disp_mu_x_this = dispPref_in_vox[n]

        # get the RF centroid of each neuron in the simulated voxel.
        # it is assumed that all neurons in the voxel has the same centroid.
        # pos_rf_x = self.rf_centroid[0] # RF x coordinate
        # pos_rf_y = self.rf_centroid[1] # RF y coordinate
        # rf_centroid_this = (pos_rf_x, pos_rf_y)
        window_in_singleVox[n] = _create_square_window(
            disp_mu_x_this, f, size_rds_bg_pix
        )

    return window_in_singleVox


@njit(parallel=True)
def _create_window_in_allVox(
    dispColMap_bootstrap,
    f_batch,
    nVox_to_analyze,
    size_rds_bg_pix,
    neurons_per_vox,
    i_bootstrap,
):
    """
    Create a batch of square window for disparity neurons contained in a single simulated
    voxel for all spatial frequency listed in self.f_batch.

    each window with spatial frequency f listed in self.f_batch is repeated
    neurons_per_vox times. Thus, there are len(f_batch) * neurons_per_vox
    windows in total.

    the order of repetition:
        [f1_disp1 f1_disp2 f1_disp3...    f2_disp1 f2_disp2 f2_disp3...    f3_disp1 f3 f3...]
            neurons_per_vox                        neurons_per_vox          neurons_per_vox

    Outputs:
        - window_in_vox: <[nVox_to_analyze * nRF_in_vox,
                           size_rds_bg[0], size_rds_bg[1]] np.array>
                    RFs contained in nVox_to_analyze simulated voxels.

        note:
            nRF_in_vox = number of RF contained in a single voxel
                nRF_in_vox = len(f_batch) * neurons_per_vox

        note: vox_channels is similar to number of voxels used in cell column model
              rfDisp_channels is similar to the number of RFs in a voxel

    """

    n_rf_in_allVox = nVox_to_analyze * len(f_batch) * neurons_per_vox
    window_in_allVox = np.zeros(
        (n_rf_in_allVox, size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8
    )  # RFs in all vox

    for v in range(nVox_to_analyze):
        dispPref_in_vox = dispColMap_bootstrap[i_bootstrap, v]

        for f in prange(len(f_batch)):
            f_this = f_batch[f]

            # [neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
            window_in_singleVox = _create_window_in_singleVox_singleFreq(
                dispPref_in_vox, size_rds_bg_pix, neurons_per_vox, f_this
            )

            id_start = (v * len(f_batch) * neurons_per_vox) + f * neurons_per_vox
            id_end = id_start + neurons_per_vox
            window_in_allVox[id_start:id_end] = window_in_singleVox

    return window_in_allVox


@njit(parallel=True)
def _generate_id_rds_bootstrap(n_rds, n_trial, n_bootstrap):

    id_rds_allBootstrap = np.zeros((n_bootstrap, 2, n_trial), dtype=np.int32)
    for i in prange(n_bootstrap):
        id_rds_allBootstrap[i] = np.random.choice(
            np.arange(n_rds), (2, n_trial), replace=False
        )

    return id_rds_allBootstrap


@njit(parallel=True)
def _generate_rds_batch_trial(
    rds_L, rds_R, id_rds, size_rds_bg_pix, n_rds_single_trial, n_trial_batch
):
    """

    Parameters
    ----------
    rds_L : [n_rds, 2, size_rds_bg_pix, size_rds_bg_pix] np.int8
        rds images for the left eye
        n_rds = 10240

        the second axis ("2") indicates crossed and uncrossed rds images

    rds_R : [n_rds, 2, size_rds_bg_pix, size_rds_bg_pix] np.int8
        rds images for the right eye
        n_rds = 10240

        the second axis ("2") indicates crossed and uncrossed rds images

    id_rds : [n_trial_batch] np.array
        random integer number for choosing images in rds_L and rds_R

        id_rds = np.random.choice(np.arange(10240), n_trial_batch, replace=False).

    size_rds_bg_pix : scalar
        the size of rds in pixel (120)

    n_rds_single_trial : scalar
        the number of rds images to be copied or repeated for parallel
        processing with cuda. All RFs must receive the same rds, therefore
        the rds must be copied or repeated for parallel processing.

        n_rds_single_trial = nVox_to_analyze * len(f_batch) * neurons_per_vox

    n_trial_batch : scalar
        the number of trial batches.

        e.g: 2

    Returns
    -------
    rds_L_CU_trial : [2 * nVox_to_analyze * len(f_batch) * neurons_per_vox,
                       size_rds_bg_pix, size_rds_bg_pix]
                    np.int8

                    left rds, containes crossed and uncrossed disparity

    rds_R_CU_trial : [2 * nVox_to_analyze * len(f_batch) * neurons_per_vox,
                       size_rds_bg_pix, size_rds_bg_pix]
                    np.int8

                    right rds, containes crossed and uncrossed disparity

        a very big rds matrix. Each stores 2 RDSs in the following order:
            rds_L_CU_trial[0 : n_trial_batch] -> rds_left_crossed, trial 1:n_trial_batch
            rds_L_CU_trial[n_trial_batch : 2*n_trial_batch] -> rds_left_uncrossed, trial 1:n_trial_batch

            rds_R_CU_trial[0 : n_trial_batch] -> rds_right_crossed, trial 1:n_trial_batch
            rds_R_CU_trial[n_trial_batch : 2*n_trial_batch] -> rds_right_uncrossed, trial 1:n_trial_batch

    """

    rds_L_CU = np.zeros(
        (n_trial_batch * 2, size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8
    )
    rds_R_CU = np.zeros(
        (n_trial_batch * 2, size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8
    )

    for i in prange(n_trial_batch):

        # rds left
        rds_left_crossed = rds_L[id_rds[i], 0]
        rds_left_uncrossed = rds_L[id_rds[i], 1]

        # rds right
        rds_right_crossed = rds_R[id_rds[i], 0]
        rds_right_uncrossed = rds_R[id_rds[i], 1]

        # rds left crossed
        rds_L_CU[i] = rds_left_crossed

        # rds left uncrossed
        rds_L_CU[i + n_trial_batch] = rds_left_uncrossed

        # rds right crossed
        rds_R_CU[i] = rds_right_crossed

        # rds right uncrossed
        rds_R_CU[i + n_trial_batch] = rds_right_uncrossed

    return rds_L_CU, rds_R_CU


@njit
def _deg2pix(disp_mu_x):
    ################## convert disparity in deg to voxel ######################
    # disp_mu_in_pixel = 9 # the number of pixels associated with 0.2 deg
    screen_height = 35.5  # screen height (cm)
    cm_per_pix = screen_height / 1024  # screen pixel resolution
    view_dist = 96  # viewing distance (cm)
    h = 2 * view_dist * math.tan((disp_mu_x / 2) * np.pi / 180)

    disp_mu_in_pixel = np.int32(h / cm_per_pix)

    return disp_mu_in_pixel


@njit(parallel=True)
def _dispCol_in_pixel(dispCol):

    n_bootstrap, nVox, n_neurons = dispCol.shape

    dispCol_in_pix = np.zeros((n_bootstrap, nVox, n_neurons), dtype=np.int8)

    for b in prange(n_bootstrap):
        for v in range(nVox):
            for i in range(n_neurons):

                dispCol_in_pix[b, v, i] = _deg2pix(dispCol[b, v, i])

    return dispCol_in_pix


@njit(parallel=True)
def _vectorize_disparity(dispCol_in_pix, n_f_batch):
    """
    repeat and vectorize the dispCol for parallel processing with cuda

    the order of repetition:
        [f1_disp1 f1_disp2 f1_disp3...    f2_disp1 f2_disp2 f2_disp3...    f3_disp1 f3 f3...]
                neurons_per_vox                  neurons_per_vox          neurons_per_vox

    Parameters
    ----------
    dispCol_in_pix : [n_bootstrap, nVox_to_analyze, neurons_per_vox] np.array
        disparity columnar structure containing disparity information in pixel for each
        neuron in the simulated voxels.

    n_f_batch : scalar
        len(f_batch)

    Returns
    -------
    dispCol_vect : [n_bootstrap, n_repeat], np.int8

                n_repeat = nVox_to_analyze * len(f_batch) * neurons_per_vox

        vectorized repeated disparity column.

    """

    n_bootstrap, nVox, n_neurons = dispCol_in_pix.shape
    n_repeat = n_f_batch * nVox * n_neurons
    dispCol_vect = np.zeros((n_bootstrap, n_repeat), dtype=np.int8)

    for b in range(n_bootstrap):
        for v in prange(nVox):
            for i in range(n_f_batch):
                id_start = v * n_f_batch * n_neurons + i * n_neurons
                id_end = id_start + n_neurons

                dispCol_vect[b, id_start:id_end] = dispCol_in_pix[b, v]

    return dispCol_vect


@cuda.jit
def _monoResp_cuda(rds_CU_trial_gpu, window_allVox_gpu, dispCol_vect_gpu, monoResp_gpu):
    """
    compute monocular response with njit

    Parameters
    ----------
    rds_L_CU_trial_gpu : [n_trial_batch * 2 * nVox_to_analyze * len(f_batch) * neurons_per_vox,
                           size_rds_bg_pix, size_rds_bg_pix] np.array
        rds for the left eye

        the second axis ("2") indicates crossed and uncrossed rds images

    rds_R_CU_trial_gpu : [n_trial_batch * 2 * nVox_to_analyze * len(f_batch) * neurons_per_vox,
                           size_rds_bg_pix, size_rds_bg_pix] np.array
        rds for the right eye

        the second axis ("2") indicates crossed and uncrossed rds images

        a very big rds matrix. Each stores 2 RDSs in the following order:
            rds_L_CU_trial[0 : n_trial_batch] -> rds_left_crossed, trial 1:n_trial_batch
            rds_L_CU_trial[n_trial_batch : 2*n_trial_batch] -> rds_left_uncrossed, trial 1:n_trial_batch

            rds_R_CU_trial[0 : n_trial_batch] -> rds_right_crossed, trial 1:n_trial_batch
            rds_R_CU_trial[n_trial_batch : 2*n_trial_batch] -> rds_right_uncrossed, trial 1:n_trial_batch


    window_allVox_gpu : [nVox * len(f_batch) * neurons_per_vox,
                           size_rds_bg_pix, size_rds_bg_pix] np.array
        rf for a single bootstrap.

    dispCol_vect_gpu : [n_repeat], np.int8

                n_repeat = nVox_to_analyze * len(f_batch) * neurons_per_vox

        vectorized repeated disparity column in pixel.

    Returns
    -------
    monoResp : [n_trial_batch * 2 * nVox * len(f_batch) * neurons_per_vox,
                           size_rds_bg_pix, size_rds_bg_pix]
                np.array
        monocular response.

    """
    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # tz = cuda.threadIdx.z

    # define shared memory
    # cache_rds = cuda.shared.array(shape=(8, 8), dtype=int8)
    # cache_rf = cuda.shared.array(shape=(8, 8), dtype=int8)
    # cache_shift = cuda.shared.array(shape=(8, 8), dtype=int8)

    # initialize cache
    # cache_rds[ty, tx] = 0
    # cache_rf[ty, tx] = 0
    # cuda.syncthreads()

    if (
        (z < window_allVox_gpu.shape[0])
        and (y < window_allVox_gpu.shape[1])
        and (x < window_allVox_gpu.shape[2])
    ):

        # reset monoResp_gpu
        monoResp_gpu[z, y, x] = 0
        cuda.syncthreads()

        # iterate over rds_trial
        for id_z_rds in range(rds_CU_trial_gpu.shape[0]):

            # cache_rds[ty, tx] = rds_CU_trial_gpu[id_z_rds, y, x]
            # cuda.syncthreads()

            # iterate over rf
            for id_z_rf in range(z, window_allVox_gpu.shape[0], stride_z):

                # cache_rf[ty, tx] = window_allVox_gpu[id_z_rf, y, x]
                # cuda.syncthreads()

                idx = id_z_rf + (id_z_rds * window_allVox_gpu.shape[0])
                # monoResp_gpu[idx, y, x + dispCol_vect_gpu[id_z_rf]] = \
                #     cache_rds[ty, tx] * cache_rf[ty, tx]
                monoResp_gpu[idx, y, x + dispCol_vect_gpu[id_z_rf]] = (
                    rds_CU_trial_gpu[id_z_rds, y, x] * window_allVox_gpu[id_z_rf, y, x]
                )

                cuda.syncthreads()


@njit
def _shift_monoResp(monoResp, shift_pix):
    """
    shift monocular response according to disparity shift_pix

    Parameters
    ----------
    monoResp : TYPE
        DESCRIPTION.
    shift_pix : TYPE
        DESCRIPTION.

    Returns
    -------
    img_shift : TYPE
        DESCRIPTION.

    """

    n_row, n_col = monoResp.shape

    img_shift = np.zeros((n_row, n_col), dtype=np.int8)

    for col in range(n_col):
        idx = col + shift_pix
        if idx >= 0 and idx < n_col:
            img_shift[:, col + shift_pix] = monoResp[:, col]

    return img_shift


@njit(parallel=True)
def monoResp_check(rds_CU_trial_cpu, window_allVox_cpu, dispCol_vect):

    n_rf = window_allVox_cpu.shape[0]
    n_rds = rds_CU_trial_cpu.shape[0]
    n_mono = n_rf * n_rds

    monoResp_check_cpu = np.zeros((n_mono, 120, 120), dtype=np.int8)

    # n_epoch = np.int32(n_mono/window_allVox_cpu.shape[0])
    for t in range(n_rds):

        for i in prange(n_rf):

            monoResp = rds_CU_trial_cpu[t] * window_allVox_cpu[i]
            idx = i + t * n_rf
            monoResp_check_cpu[idx] = _shift_monoResp(monoResp, dispCol_vect[i])

    return monoResp_check_cpu


@njit(parallel=True)
def monoResp_check_without_shift(rds_CU_trial_cpu, window_allVox_cpu):

    n_rf = window_allVox_cpu.shape[0]
    n_rds = rds_CU_trial_cpu.shape[0]
    n_mono = n_rf * n_rds

    monoResp_check_cpu = np.zeros((n_mono, 120, 120), dtype=np.int8)

    for t in range(n_rds):
        for i in prange(n_rf):

            monoResp = rds_CU_trial_cpu[t] * window_allVox_cpu[i]
            idx = i + t * n_rf
            monoResp_check_cpu[idx] = monoResp

    return monoResp_check_cpu


@cuda.jit
def _binoResp_corr_cuda(monoResp_L_gpu, monoResp_R_gpu, binoResp_corr_gpu):
    """
    compute binocular response according to cross-correlation with cuda.

    Parameters
    ----------
    monoResp_L_gpu : TYPE
        DESCRIPTION.
    monoResp_R_gpu : TYPE
        DESCRIPTION.
    binoResp_corr_gpu : TYPE
        DESCRIPTION.
    binoResp_match_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # tz = cuda.threadIdx.z

    # define shared memory
    cache_left = cuda.shared.array(shape=(8, 8), dtype=float32)
    cache_right = cuda.shared.array(shape=(8, 8), dtype=float32)

    # initialize cache
    cache_left[ty, tx] = 0
    cache_right[ty, tx] = 0

    if (
        (z < monoResp_L_gpu.shape[0])
        and (y < monoResp_L_gpu.shape[1])
        and (x < monoResp_L_gpu.shape[2])
    ):

        for idx in range(z, monoResp_L_gpu.shape[0], stride_z):

            cache_left[ty, tx] = monoResp_L_gpu[idx, y, x]
            cache_right[ty, tx] = monoResp_R_gpu[idx, y, x]

            # compute cross-correlation
            binoResp_corr_gpu[idx, y, x] = cache_left[ty, tx] * cache_right[ty, tx]


@cuda.jit
def _binoResp_match_cuda(monoResp_L_gpu, monoResp_R_gpu, binoResp_match_gpu):
    """
    compute binocular response according to cross-matching with cuda.

    Parameters
    ----------
    monoResp_L_gpu : TYPE
        DESCRIPTION.
    monoResp_R_gpu : TYPE
        DESCRIPTION.
    binoResp_corr_gpu : TYPE
        DESCRIPTION.
    binoResp_match_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # tz = cuda.threadIdx.z

    # define shared memory
    cache_left = cuda.shared.array(shape=(8, 8), dtype=float32)
    cache_right = cuda.shared.array(shape=(8, 8), dtype=float32)

    # initialize cache
    cache_left[ty, tx] = 0
    cache_right[ty, tx] = 0

    if (
        (z < monoResp_L_gpu.shape[0])
        and (y < monoResp_L_gpu.shape[1])
        and (x < monoResp_L_gpu.shape[2])
    ):

        for idx in range(z, monoResp_L_gpu.shape[0], stride_z):

            cache_left[ty, tx] = monoResp_L_gpu[idx, y, x]
            cache_right[ty, tx] = monoResp_R_gpu[idx, y, x]

            # compute cross-matching
            binoResp_match_gpu[idx, y, x] = cache_left[ty, tx] * cache_right[ty, tx]
            if binoResp_match_gpu[idx, y, x] < 0:
                binoResp_match_gpu[idx, y, x] = 0


@cuda.jit
def _binoResp_cuda(
    monoResp_L_gpu, monoResp_R_gpu, binoResp_corr_gpu, binoResp_match_gpu
):
    """
    compute binocular response for both cross-correlation and
    cross-matching with cuda.

    Parameters
    ----------
    monoResp_L_gpu : TYPE
        DESCRIPTION.
    monoResp_R_gpu : TYPE
        DESCRIPTION.
    binoResp_corr_gpu : TYPE
        DESCRIPTION.
    binoResp_match_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # tz = cuda.threadIdx.z

    # define shared memory
    cache_left = cuda.shared.array(shape=(8, 8), dtype=float32)
    cache_right = cuda.shared.array(shape=(8, 8), dtype=float32)

    # initialize cache
    cache_left[ty, tx] = 0
    cache_right[ty, tx] = 0

    if (
        (z < monoResp_L_gpu.shape[0])
        and (y < monoResp_L_gpu.shape[1])
        and (x < monoResp_L_gpu.shape[2])
    ):

        for idx in range(z, monoResp_L_gpu.shape[0], stride_z):

            cache_left[ty, tx] = monoResp_L_gpu[idx, y, x]
            cache_right[ty, tx] = monoResp_R_gpu[idx, y, x]

            # compute cross-correlation
            binoResp_corr_gpu[idx, y, x] = cache_left[ty, tx] * cache_right[ty, tx]

            # compute cross-matching
            binoResp_match_gpu[idx, y, x] = cache_left[ty, tx] * cache_right[ty, tx]
            if binoResp_match_gpu[idx, y, x] < 0:
                binoResp_match_gpu[idx, y, x] = 0


@njit(parallel=True)
def binoResp_check(monoResp_L_cpu, monoResp_R_cpu):

    binoResp = monoResp_L_cpu * monoResp_R_cpu

    return binoResp


@cuda.jit
def _corr_partial_cuda(binoResp_corr_gpu, partial_corr_gpu):
    """
    reduce matrix binoResp_corr_gpu to [4, 4]

    Parameters
    ----------
    binoResp_corr_gpu : TYPE
        DESCRIPTION.

    partial_corr_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache_corr = cuda.shared.array(shape=(32, 32), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x  # column
    ty = cuda.threadIdx.y  # row

    for idx in range(z, binoResp_corr_gpu.shape[0], stride_z):
        # initialize cache, copy binoResp_gpu to the cache
        cache_corr[ty, tx] = 0

        if (
            (z < binoResp_corr_gpu.shape[0])
            and (y < binoResp_corr_gpu.shape[1])
            and (x < binoResp_corr_gpu.shape[2])
        ):  # only access the indexes within binoResp_gpu

            cache_corr[ty, tx] += binoResp_corr_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(16)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache_corr[ty, tx] += cache_corr[ty, tx + i]

            cuda.syncthreads()
            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(16)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache_corr[ty, tx] += cache_corr[ty + i, tx]

            cuda.syncthreads()
            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            partial_corr_gpu[idx, cuda.blockIdx.y, cuda.blockIdx.x] = cache_corr[0, 0]


@cuda.jit
def _match_partial_cuda(binoResp_match_gpu, partial_match_gpu):
    """
    reduce matrix binoResp_match_gpu to [4, 4]

    Parameters
    ----------
    binoResp_match_gpu : TYPE
        DESCRIPTION.

    partial_match_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache_match = cuda.shared.array(shape=(32, 32), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x  # column
    ty = cuda.threadIdx.y  # row

    for idx in range(z, binoResp_match_gpu.shape[0], stride_z):
        # initialize cache, copy binoResp_gpu to the cache
        cache_match[ty, tx] = 0
        if (
            (z < binoResp_match_gpu.shape[0])
            and (y < binoResp_match_gpu.shape[1])
            and (x < binoResp_match_gpu.shape[2])
        ):  # only access the indexes within binoResp_gpu

            cache_match[ty, tx] += binoResp_match_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(16)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache_match[ty, tx] += cache_match[ty, tx + i]

            cuda.syncthreads()
            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(16)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache_match[ty, tx] += cache_match[ty + i, tx]

            cuda.syncthreads()
            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            partial_match_gpu[idx, cuda.blockIdx.y, cuda.blockIdx.x] = cache_match[0, 0]


@cuda.jit
def _cmm_partial_cuda(
    binoResp_corr_gpu, binoResp_match_gpu, partial_corr_gpu, partial_match_gpu
):
    """
    reduce matrix binoResp_corr_gpu and binoResp_match_gpu to [4, 4]

    Parameters
    ----------
    binoResp_corr_gpu : TYPE
        DESCRIPTION.
    binoResp_match_gpu : TYPE
        DESCRIPTION.
    partial_corr_gpu : TYPE
        DESCRIPTION.
    partial_match_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache_corr = cuda.shared.array(shape=(32, 32), dtype=float32)
    cache_match = cuda.shared.array(shape=(32, 32), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x  # column
    ty = cuda.threadIdx.y  # row

    for idx in range(z, binoResp_corr_gpu.shape[0], stride_z):
        # initialize cache, copy binoResp_gpu to the cache
        cache_corr[ty, tx] = 0
        cache_match[ty, tx] = 0
        if (
            (z < binoResp_corr_gpu.shape[0])
            and (y < binoResp_corr_gpu.shape[1])
            and (x < binoResp_corr_gpu.shape[2])
        ):  # only access the indexes within binoResp_gpu

            cache_corr[ty, tx] += binoResp_corr_gpu[idx, y, x]
            cache_match[ty, tx] += binoResp_match_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(16)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache_corr[ty, tx] += cache_corr[ty, tx + i]
                cache_match[ty, tx] += cache_match[ty, tx + i]

            cuda.syncthreads()
            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(16)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache_corr[ty, tx] += cache_corr[ty + i, tx]
                cache_match[ty, tx] += cache_match[ty + i, tx]

            cuda.syncthreads()
            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            partial_corr_gpu[idx, cuda.blockIdx.y, cuda.blockIdx.x] = cache_corr[0, 0]
            partial_match_gpu[idx, cuda.blockIdx.y, cuda.blockIdx.x] = cache_match[0, 0]


@cuda.jit
def _corr_final_cuda(partial_corr_gpu, rf_area_gpu, corrResp_gpu):
    """
    reduce matrix partial_corr_gpu to get the sum of the matrix

    Parameters
    ----------
    partial_corr_gpu : TYPE
        DESCRIPTION.

    corrResp_gpu : TYPE
        DESCRIPTION.

    rf_area_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache_corr = cuda.shared.array(shape=(4, 4), dtype=float32)
    cache_area = cuda.shared.array(shape=(1, 1), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    cache_area[0, 0] = 0
    if z < rf_area_gpu.shape[0]:
        cache_area[0, 0] = rf_area_gpu[z]

    for idx in range(z, partial_corr_gpu.shape[0], stride_z):
        # reset cache, and initialize
        cache_corr[ty, tx] = 0

        if (
            (z < partial_corr_gpu.shape[0])
            and (y < partial_corr_gpu.shape[1])
            and (x < partial_corr_gpu.shape[2])
        ):  # only access the indexes within a_gpu
            cache_corr[ty, tx] += partial_corr_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(2)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache_corr[ty, tx] += cache_corr[ty, tx + i]

            cuda.syncthreads()
            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(2)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache_corr[ty, tx] += cache_corr[ty + i, tx]

            cuda.syncthreads()
            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            corrResp_gpu[idx] = cache_corr[0, 0] / cache_area[0, 0]


@cuda.jit
def _match_final_cuda(partial_match_gpu, rf_area_gpu, matchResp_gpu):
    """
    reduce matrix partial_match_gpu to get the sum of the matrix

    Parameters
    ----------
    partial_match_gpu : TYPE
        DESCRIPTION.

    matchResp_gpu : TYPE
        DESCRIPTION.

    rf_area_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache_match = cuda.shared.array(shape=(4, 4), dtype=float32)
    cache_area = cuda.shared.array(shape=(1, 1), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    cache_area[0, 0] = 0
    if z < rf_area_gpu.shape[0]:
        cache_area[0, 0] = rf_area_gpu[z]

    for idx in range(z, partial_match_gpu.shape[0], stride_z):
        # reset cache, and initialize
        cache_match[ty, tx] = 0

        if (
            (z < partial_match_gpu.shape[0])
            and (y < partial_match_gpu.shape[1])
            and (x < partial_match_gpu.shape[2])
        ):  # only access the indexes within a_gpu

            cache_match[ty, tx] += partial_match_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(2)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache_match[ty, tx] += cache_match[ty, tx + i]

            cuda.syncthreads()

            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(2)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache_match[ty, tx] += cache_match[ty + i, tx]

            cuda.syncthreads()

            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            matchResp_gpu[idx] = cache_match[0, 0] / cache_area[0, 0]


@cuda.jit
def _cmm_final_cuda(
    partial_corr_gpu, partial_match_gpu, corrResp_gpu, matchResp_gpu, rf_area_gpu
):
    """
    reduce matrix partial_corr_gpu and partial_match_gpu to get the sum of
    matrix

    Parameters
    ----------
    partial_corr_gpu : TYPE
        DESCRIPTION.
    partial_match_gpu : TYPE
        DESCRIPTION.
    corrResp_gpu : TYPE
        DESCRIPTION.
    matchResp_gpu : TYPE
        DESCRIPTION.
    rf_area_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache_corr = cuda.shared.array(shape=(4, 4), dtype=float32)
    cache_match = cuda.shared.array(shape=(4, 4), dtype=float32)
    cache_area = cuda.shared.array(shape=(1, 1), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    cache_area[0, 0] = 0
    if z < rf_area_gpu.shape[0]:
        cache_area[0, 0] = rf_area_gpu[z]

    for idx in range(z, partial_corr_gpu.shape[0], stride_z):
        # reset cache, and initialize
        cache_corr[ty, tx] = 0
        cache_match[ty, tx] = 0

        if (
            (z < partial_corr_gpu.shape[0])
            and (y < partial_corr_gpu.shape[1])
            and (x < partial_corr_gpu.shape[2])
        ):  # only access the indexes within a_gpu
            cache_corr[ty, tx] += partial_corr_gpu[idx, y, x]
            cache_match[ty, tx] += partial_match_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(2)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache_corr[ty, tx] += cache_corr[ty, tx + i]
                cache_match[ty, tx] += cache_match[ty, tx + i]

            cuda.syncthreads()

            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(2)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache_corr[ty, tx] += cache_corr[ty + i, tx]
                cache_match[ty, tx] += cache_match[ty + i, tx]

            cuda.syncthreads()

            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            corrResp_gpu[idx] = cache_corr[0, 0] / cache_area[0, 0]
            matchResp_gpu[idx] = cache_match[0, 0] / cache_area[0, 0]


@njit(parallel=True)
def _compute_rf_area(window_allVox_cpu):

    n_rf = window_allVox_cpu.shape[0]
    rf_area = np.zeros(n_rf, dtype=np.float32)
    for i in prange(n_rf):
        rf_area[i] = np.sum(window_allVox_cpu[i])

    return rf_area


@cuda.jit
def _rf_area_partial_cuda(rf_gpu, partial_area_gpu):
    """
    reduce matrix rf_gpu to get the partial sum of this matrix.
    This step is the first step to compute the area of RF

    Parameters
    ----------
    rf_gpu : TYPE
        DESCRIPTION.
    partial_area_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache = cuda.shared.array(shape=(32, 32), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x  # column
    ty = cuda.threadIdx.y  # row

    for idx in range(z, rf_gpu.shape[0], stride_z):
        # initialize cache, copy binoResp_gpu to the cache
        cache[ty, tx] = 0
        if (
            (z < rf_gpu.shape[0]) and (y < rf_gpu.shape[1]) and (x < rf_gpu.shape[2])
        ):  # only access the indexes within binoResp_gpu

            cache[ty, tx] += rf_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(16)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache[ty, tx] += cache[ty, tx + i]

            cuda.syncthreads()
            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(16)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache[ty, tx] += cache[ty + i, tx]

            cuda.syncthreads()
            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            partial_area_gpu[idx, cuda.blockIdx.y, cuda.blockIdx.x] = cache[0, 0]


@cuda.jit
def _rf_area_final_cuda(partial_area_gpu, area_gpu):
    """
    the final step in reducing matrix partial_area_gpu.

    Parameters
    ----------
    partial_area_gpu : TYPE
        DESCRIPTION.
    area_gpu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # define shared memory
    # tpb = int8(8)
    cache = cuda.shared.array(shape=(4, 4), dtype=float32)

    x, y, z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    for idx in range(z, partial_area_gpu.shape[0], stride_z):
        # initialize cache
        cache[ty, tx] = 0
        if (
            (z < partial_area_gpu.shape[0])
            and (y < partial_area_gpu.shape[1])
            and (x < partial_area_gpu.shape[2])
        ):  # only access the indexes within a_gpu
            cache[ty, tx] += partial_area_gpu[idx, y, x]

        # reduce along x (column)
        i = int8(2)  # threads per block
        while i > 0:
            if tx < i:  # fill in cache whose column index is less than i
                cache[ty, tx] += cache[ty, tx + i]

            cuda.syncthreads()

            i = int8(i >> 1)

        # reduce along y (row)
        i = int8(2)  # threads per block
        while i > 0:
            if ty < i:  # fill in cache whose row index is less than i
                cache[ty, tx] += cache[ty + i, tx]

            cuda.syncthreads()

            i = int8(i >> 1)

        if (ty == 0) and (tx == 0):
            area_gpu[idx] = cache[0, 0]


@njit(parallel=True)
def _crossCorr_cpu(monoResp_cpu):

    temp = np.zeros((120, 120), dtype=np.int8)
    cmmResp_cpu = np.zeros((16, 250, 50), dtype=np.float32)

    for t in prange(monoResp_cpu.shape[0]):
        for v in prange(250):

            row_start = v * 250
            row_end = row_start + 250

            for i in range(50):

                col_start = i * 50
                col_end = col_start + 50

                temp = monoResp_cpu[t, row_start:row_end, col_start:col_end]
                cmmResp_cpu[t, v, i] = temp.sum()

    return cmmResp_cpu


@njit(parallel=True)
def _preprocess_cmm(
    cmmResp, f_batch, nVox_to_analyze, neurons_per_vox, n_trial_batch, n_trial_total
):
    """
    preprocessing cmmResp matrix by breaking it down according to crossed
    and uncrossed

    Parameters
    ----------
    cmmResp : TYPE
        DESCRIPTION.
    f_batch : TYPE
        DESCRIPTION.
    nVox_to_analyze : TYPE
        DESCRIPTION.
    neurons_per_vox : TYPE
        DESCRIPTION.
    n_trial_total : TYPE
        DESCRIPTION.

    Returns
    -------
    cmmResp_crossed : [n_bootstrap, n_trial_total, nVox_to_analyze,
                        n_rf_in_singleVox] np.array
        DESCRIPTION.
    cmmResp_uncrossed : TYPE
        DESCRIPTION.

    """

    n_bootstrap = cmmResp.shape[0]
    n_rf_in_singleVox = len(f_batch) * neurons_per_vox
    n_rf_in_singleTrial = nVox_to_analyze * n_rf_in_singleVox
    offset = n_trial_batch * n_rf_in_singleTrial
    n_epoch = np.int32(n_trial_total / n_trial_batch)

    cmmResp_crossed = np.zeros(
        (n_bootstrap, n_trial_total, nVox_to_analyze), dtype=np.float32
    )
    cmmResp_uncrossed = np.zeros(
        (n_bootstrap, n_trial_total, nVox_to_analyze), dtype=np.float32
    )

    for b in prange(n_bootstrap):
        for e in range(n_epoch):

            # get crossed-disp response
            id_start_crossed = 2 * e * offset
            id_end_crossed = id_start_crossed + offset
            resp_crossed = cmmResp[b, id_start_crossed:id_end_crossed]

            # get uncrossed-disp response
            id_start_uncrossed = (2 * e + 1) * offset
            id_end_uncrossed = id_start_uncrossed + offset
            resp_uncrossed = cmmResp[b, id_start_uncrossed:id_end_uncrossed]

            for t in range(n_trial_batch):
                for v in range(nVox_to_analyze):
                    id_start1 = (t * n_rf_in_singleTrial) + v * n_rf_in_singleVox
                    id_end1 = id_start1 + n_rf_in_singleVox

                    ## sum neural responses across rf -> reduce memory usage
                    cmmResp_crossed[b, t + e * n_trial_batch, v] = np.sum(
                        resp_crossed[id_start1:id_end1]
                    )

                    # for t in range(n_trial_batch):
                    #     for v in range(nVox_to_analyze):
                    id_start2 = (t * n_rf_in_singleTrial) + v * n_rf_in_singleVox
                    id_end2 = id_start2 + n_rf_in_singleVox

                    ## sum neural responses across rf -> reduce memory usage
                    cmmResp_uncrossed[b, t + e * n_trial_batch, v] = np.sum(
                        resp_uncrossed[id_start2:id_end2]
                    )

    return cmmResp_crossed, cmmResp_uncrossed


def _create_rdm_fmri_vec(rdm_fmri_all):

    n_sbjID, n_ROIs, _, _ = np.shape(rdm_fmri_all)

    rdm_fmri_vec_all = np.zeros((n_sbjID, n_ROIs, 15), dtype=np.float32)
    for sbj in prange(n_sbjID):

        for roi in range(n_ROIs):

            rdm_fmri_roi = rdm_fmri_all[sbj, roi]

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_above -= np.sum(rdm_fmri_above) / np.shape(rdm_fmri_above)[0]

            # normalize by dividing max value
            rdm_fmri_vec = rdm_fmri_above / np.max(rdm_fmri_above)
            # rdm_fmri_vec = rdm_fmri_above / math.sqrt(np.sum(rdm_fmri_above**2))

            rdm_fmri_vec_all[sbj, roi] = rdm_fmri_vec

    return rdm_fmri_vec_all


@njit(fastmath=True)
def _cdist(A, B, mtd):

    # assert A.shape[1] == B.shape[1]

    # allocate distance matrix
    C = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)

    # get init value with the same datatype as matrix A
    init_val_arr = np.zeros(1, np.float32)
    init_val = init_val_arr[0]

    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            dist = init_val

            for k in range(A.shape[1]):
                dist += (A[i, k] - B[j, k]) ** 2

            if mtd == "euclidean":
                C[i, j] = np.sqrt(dist)
            elif mtd == "sqeuclidean":
                C[i, j] = dist

    return C


def _generate_train_idx_leave_one_out(n_sbjID):

    train_id = np.zeros((n_sbjID, n_sbjID - 1), dtype=np.int8)
    idx = np.arange(n_sbjID).astype(np.int8)

    for i in range(n_sbjID):

        train_id[i] = np.delete(idx, i)

    return train_id


@njit(parallel=True)
def _repeat_in_row_2d(A, n_repeat):

    # n_dim = len(A.shape)
    # if len(A.shape) == 2:
    # if n_dim == 2:
    n_row = A.shape[0]
    n_col = A.shape[1]

    A_repeat = np.empty((n_row * n_repeat, n_col), dtype=np.float32)
    for i in prange(n_repeat):

        id_start = i * n_row
        id_end = id_start + n_row

        A_repeat[id_start:id_end] = A[0:n_row]

    return A_repeat


def _fit_leave_one_out_singleBootstrap(
    rdm_fmri_vec_all, rdm_corr, rdm_match, train_id, i_bootstrap
):

    n_sbjID, n_ROIs, _ = np.shape(rdm_fmri_vec_all)

    # get rdm_corr and rdm_match at i_bootstrap
    rdm_corr_bootstrap = rdm_corr[i_bootstrap]
    rdm_match_bootstrap = rdm_match[i_bootstrap]

    # prepare regressor
    # get above diagonal elements
    rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
    rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

    #  mean-zeroing
    rdm_corr_above -= np.mean(rdm_corr_above)
    rdm_match_above -= np.mean(rdm_match_above)

    # normalize by dividing max value
    rdm_corr_vec = rdm_corr_above / np.max(rdm_corr_above)
    rdm_match_vec = rdm_match_above / np.max(rdm_match_above)
    # rdm_corr_vec = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))
    # rdm_match_vec = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

    # set up regressor, [3, 15]
    x = np.vstack([rdm_corr_vec, rdm_match_vec, np.ones(len(rdm_corr_vec))])

    w_cmm_group_roi = np.zeros((n_ROIs, n_sbjID, 3), dtype=np.float32)
    loss_group_roi = np.zeros((n_ROIs, n_sbjID), dtype=np.float32)
    # pearson_group_roi = np.zeros((n_ROIs, n_sbjID), dtype=np.float32)

    for roi in range(n_ROIs):

        for sbj in range(n_sbjID):

            rdm_fmri_vec_train = rdm_fmri_vec_all[train_id[sbj], roi].flatten()

            # repeat x, [3*len(train_id[sbj]), 15]
            x_repeat = _repeat_in_row_2d(x.T, len(train_id[sbj]))

            # start fitting
            # w = solve_pg(x.T, rdm_fmri_vec)
            # w_cmm_single_bootstrap[roi] = w

            # start fitting using Ridge regression
            # forces the coefficients to be positive, use solver="lbfgs"
            clf = Ridge(alpha=1.0, solver="lbfgs", positive=True)

            # start fitting using OLS
            # clf = LinearRegression(positive=True)

            # start fitting using Lasso regression
            # clf = Lasso(alpha=1)

            clf.fit(x_repeat, rdm_fmri_vec_train)
            w = clf.coef_
            w_cmm_group_roi[roi, sbj] = w

            # test model
            y_true = rdm_fmri_vec_all[sbj, roi]
            y_pred = np.matmul(w, x)

            # compute lost
            loss_group_roi[roi, sbj] = np.sqrt(np.sum((y_true - y_pred) ** 2))
            # loss_group_roi[roi, sbj] = kendalltau(y_true, y_pred)[0]
            # loss_group_roi[roi, sbj] = spearmanr(y_true, y_pred)[0]
            # loss_group_roi[roi, sbj] = pearsonr(y_true, y_pred)[0]

    return w_cmm_group_roi, loss_group_roi


def _fit_leave_one_out(rdm_fmri_vec_all, rdm_corr, rdm_match):

    n_sbjID, n_ROIs, _ = rdm_fmri_vec_all.shape
    n_bootstrap = rdm_corr.shape[0]
    train_id = _generate_train_idx_leave_one_out(n_sbjID)

    # a = _fit_leave_one_out_singleBootstrap(rdm_fmri_vec_all,
    #                                     rdm_corr, rdm_match,
    #                                     train_id,
    #                                     0)

    w_cmm_list = []
    w_cmm_list.append(
        Parallel(n_jobs=-1)(
            delayed(_fit_leave_one_out_singleBootstrap)(
                rdm_fmri_vec_all, rdm_corr, rdm_match, train_id, i_bootstrap
            )
            for i_bootstrap in range(n_bootstrap)
        )
    )

    # unpack
    w_cmm_group = np.zeros((n_bootstrap, n_ROIs, n_sbjID, 3), dtype=np.float32)
    loss_group = np.zeros((n_bootstrap, n_ROIs, n_sbjID), dtype=np.float32)

    for i in range(n_bootstrap):

        temp = w_cmm_list[0][i][0]
        w_cmm_group[i] = temp

        temp = w_cmm_list[0][i][1]
        loss_group[i] = temp

    return w_cmm_group, loss_group


def _compute_w_cmm_group(rdm_fmri_all, noise_dispCol_sigma_list, n_bootstrap, mtd):

    rdm_fmri_all = cmm.rdm_fmri_all

    n_sbjID, n_ROIs, _, _ = np.shape(rdm_fmri_all)

    rdm_fmri_vec_all = _create_rdm_fmri_vec(rdm_fmri_all)

    w_cmm_group_all = np.zeros(
        (len(noise_dispCol_sigma_list), n_bootstrap, n_ROIs, n_sbjID, 3),
        dtype=np.float32,
    )
    loss_group_all = np.zeros(
        (len(noise_dispCol_sigma_list), n_bootstrap, n_ROIs, n_sbjID), dtype=np.float32
    )

    # rdm_corr_all = np.zeros((len(noise_dispCol_sigma_list),
    #                          n_bootstrap,
    #                          6, 6), dtype=np.float32)

    # rdm_match_all = np.zeros((len(noise_dispCol_sigma_list),
    #                          n_bootstrap,
    #                          6, 6), dtype=np.float32)

    for d in range(len(noise_dispCol_sigma_list)):

        sawtooth_noise_std = noise_dispCol_sigma_list[d]

        # generate rdm_corr and rdm_match associated with sawtooth_noise_std
        corrResp_crossed_ards = np.load(
            "../../../../Data/CMM/corrResp_crossed_ards_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )
        corrResp_uncrossed_ards = np.load(
            "../../../../Data/CMM/corrResp_uncrossed_ards_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )

        matchResp_crossed_ards = np.load(
            "../../../../Data/CMM/matchResp_crossed_ards_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )
        matchResp_uncrossed_ards = np.load(
            "../../../../Data/CMM/matchResp_uncrossed_ards_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )

        corrResp_crossed_hmrds = np.load(
            "../../../../Data/CMM/corrResp_crossed_hmrds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )
        corrResp_uncrossed_hmrds = np.load(
            "../../../../Data/CMM/corrResp_uncrossed_hmrds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )

        matchResp_crossed_hmrds = np.load(
            "../../../../Data/CMM/matchResp_crossed_hmrds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )
        matchResp_uncrossed_hmrds = np.load(
            "../../../../Data/CMM/matchResp_uncrossed_hmrds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )

        corrResp_crossed_crds = np.load(
            "../../../../Data/CMM/corrResp_crossed_crds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )
        corrResp_uncrossed_crds = np.load(
            "../../../../Data/CMM/corrResp_uncrossed_crds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )

        matchResp_crossed_crds = np.load(
            "../../../../Data/CMM/matchResp_crossed_crds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )
        matchResp_uncrossed_crds = np.load(
            "../../../../Data/CMM/matchResp_uncrossed_crds_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            )
        )

        # compute rdm models
        # mtd = "sqeuclidean"
        rdm_corr, rdm_match = cmm.compute_rdm_cmm(
            corrResp_crossed_ards,
            corrResp_uncrossed_ards,
            corrResp_crossed_hmrds,
            corrResp_uncrossed_hmrds,
            corrResp_crossed_crds,
            corrResp_uncrossed_crds,
            matchResp_crossed_ards,
            matchResp_uncrossed_ards,
            matchResp_crossed_hmrds,
            matchResp_uncrossed_hmrds,
            matchResp_crossed_crds,
            matchResp_uncrossed_crds,
            n_bootstrap,
            mtd,
        )

        # rdm_corr_all[d] = rdm_corr
        # rdm_match_all[d] = rdm_match

        # rdm_corr_avg = rdm_corr_all.mean(axis=1)
        # rdm_match_avg = rdm_match_all.mean(axis=1)

        ## save rdm
        np.save(
            "../../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            ),
            rdm_corr,
        )
        np.save(
            "../../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                sawtooth_noise_std
            ),
            rdm_match,
        )

        # plt.imshow(np.mean(rdm_corr, axis=0), cmap="jet"), plt.show()
        # plt.imshow(np.mean(rdm_match, axis=0), cmap="jet"), plt.show()

        ## compute w_cmm
        w_cmm_group, loss_group = _fit_leave_one_out(
            rdm_fmri_vec_all, rdm_corr, rdm_match
        )

        w_cmm_group_all[d] = w_cmm_group
        loss_group_all[d] = loss_group

    return w_cmm_group_all, loss_group_all


def _find_w_cmm_group_best(w_cmm_group_all, loss_group_all):

    _, n_bootstrap, n_ROIs, n_sbjID = np.shape(loss_group_all)

    # average loss_group_all across sbjID and bootstrap
    loss_group_avg = np.mean(np.mean(loss_group_all, axis=3), axis=1)

    # find the best w_cmm_group associated with the max pearson_group_all
    loss_min = np.zeros((n_bootstrap, n_ROIs, n_sbjID), dtype=np.float32)
    w_cmm_best = np.zeros((n_bootstrap, n_ROIs, n_sbjID, 3), dtype=np.float32)
    id_best = np.zeros(n_ROIs, dtype=np.int8)

    for roi in range(n_ROIs):

        # average across sbjID
        temp = loss_group_avg[:, roi]
        idx = np.where(temp == np.min(temp))
        id_best[roi] = idx[0][0]  # id for dispColNoise

        loss_min[:, roi] = loss_group_all[idx[0][0], :, roi]
        w_cmm_best[:, roi] = w_cmm_group_all[idx[0][0], :, roi]

    return w_cmm_best, loss_min, id_best


class RDS(General):

    def __init__(
        self, n_rds_trial, rDot, dotDens, size_rds_bg_deg, size_rds_ct_deg, deg_per_pix
    ):

        super().__init__()

        # self.n_epoch = n_epoch
        # self.n_batch = n_batch
        # get the number of rds used for simulation. Total n_rds = 10240
        self.n_rds_trial = n_rds_trial

        # # self.rDot = 0.045 # dot radius
        # # self.dotDens = 0.25 # dot density
        # # self.size_rds_bg_deg = 2.5 # rds size, deg
        # # self.size_rds_ct_deg = 1.25 # center rds size, deg

        ## parameters for 2-D rds
        self.rDot = rDot  # dot radius
        self.dotDens = dotDens  # dot density
        self.size_rds_bg_deg = size_rds_bg_deg  # rds size, deg
        self.size_rds_ct_deg = size_rds_ct_deg  # center rds size, deg
        self.size_rds_bg_pix = self._compute_deg2pix(size_rds_bg_deg)
        self.size_rds_ct_pix = self._compute_deg2pix(size_rds_ct_deg)
        self.size_rds_bg = (self.size_rds_bg_pix, self.size_rds_bg_pix)
        self.size_rds_ct = (self.size_rds_ct_pix, self.size_rds_ct_pix)

        # disparity tuning axis in deg
        # self.deg_per_pix = 0.02 # deg per pix
        # step = 1*self.deg_per_pix
        # self.disp_ct_deg = np.round(np.arange(-0.25, (0.25 + step), step), 2)
        # self.disp_ct_pix = General._compute_deg2pix(self.disp_ct_deg) # disparity tuning axis in pix

    def load_rds(self, rds_type):
        """

        load rds that has crossed and uncrossed disparity

        Args:
            rds_type (str): type of rds: "ards", "hmrds", "crds".

        Returns:
            rds dimension:
            <[n_rds_trial, crossed_uncrossed, size_rds, size_rds] np.array>

            ex: [10240, 2, 120, 120]

        """
        print("load rds: {}".format(rds_type))
        self.L = np.zeros((self.n_rds_trial, 2, 120, 120), dtype=np.int8)
        self.R = np.zeros((self.n_rds_trial, 2, 120, 120), dtype=np.int8)
        # self.rds_bg = np.zeros((n_rds_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix),
        #                        dtype=np.float32)

        if rds_type == "ards":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/ards_L_crossed.npy")

            # generate n_rds_trial of random integers
            rdx_idx = np.random.randint(0, temp.shape[0], self.n_rds_trial)

            self.L[:, 0] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/ards_L_uncrossed.npy")
            self.L[:, 1] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/ards_R_crossed.npy")
            self.R[:, 0] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/ards_R_uncrossed.npy")
            self.R[:, 1] = temp[rdx_idx]

        elif rds_type == "hmrds":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/hmrds_L_crossed.npy")

            # generate n_rds_trial of random integers
            rdx_idx = np.random.randint(0, temp.shape[0], self.n_rds_trial)

            self.L[:, 0] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/hmrds_L_uncrossed.npy")
            self.L[:, 1] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/hmrds_R_crossed.npy")
            self.R[:, 0] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/hmrds_R_uncrossed.npy")
            self.R[:, 1] = temp[rdx_idx]

        elif rds_type == "crds":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/crds_L_crossed.npy")

            # generate n_rds_trial of random integers
            rdx_idx = np.random.randint(0, temp.shape[0], self.n_rds_trial)

            self.L[:, 0] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/crds_L_uncrossed.npy")
            self.L[:, 1] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/crds_R_crossed.npy")
            self.R[:, 0] = temp[rdx_idx]

            temp = np.load("../../../Data/rds/crds_R_uncrossed.npy")
            self.R[:, 1] = temp[rdx_idx]

        self.rds_type = rds_type

        ## load urds
        print("load urds")
        self.u_L = np.zeros((self.n_rds_trial, 2, 120, 120), dtype=np.int8)
        self.u_R = np.zeros((self.n_rds_trial, 2, 120, 120), dtype=np.int8)

        temp = np.load("../../../Data/rds/urds_L_crossed.npy")

        # generate n_rds_trial of random integers
        rdx_idx = np.random.randint(0, temp.shape[0], self.n_rds_trial)

        self.u_L[:, 0] = temp[rdx_idx]

        temp = np.load("../../../Data/rds/urds_L_uncrossed.npy")
        self.u_L[:, 1] = temp[rdx_idx]

        temp = np.load("../../../Data/rds/urds_R_crossed.npy")
        self.u_R[:, 0] = temp[rdx_idx]

        temp = np.load("../../../Data/rds/urds_R_uncrossed.npy")
        self.u_R[:, 1] = temp[rdx_idx]

        # update size_rds_bg_pix
        self.size_rds_bg_pix = 120

        ## load rds_background
        # print("load rds_bg")
        # temp = np.load("../../../Data/rds/rds_bg.npy")
        # self.rds_bg[:, 0] = temp[0:n_rds_trial]

    def set_rds(self, rds_type_new):

        print("set new rds: {}".format(rds_type_new))
        self.L = np.zeros((self.n_rds_trial, 2, 120, 120), dtype=np.int8)
        self.R = np.zeros((self.n_rds_trial, 2, 120, 120), dtype=np.int8)
        if rds_type_new == "ards":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/ards_L_crossed.npy")
            self.L[:, 0] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/ards_L_uncrossed.npy")
            self.L[:, 1] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/ards_R_crossed.npy")
            self.R[:, 0] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/ards_R_uncrossed.npy")
            self.R[:, 1] = temp[0 : self.n_rds_trial]

        elif rds_type_new == "hmrds":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/hmrds_L_crossed.npy")
            self.L[:, 0] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/hmrds_L_uncrossed.npy")
            self.L[:, 1] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/hmrds_R_crossed.npy")
            self.R[:, 0] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/hmrds_R_uncrossed.npy")
            self.R[:, 1] = temp[0 : self.n_rds_trial]

        elif rds_type_new == "crds":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/crds_L_crossed.npy")
            self.L[:, 0] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/crds_L_uncrossed.npy")
            self.L[:, 1] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/crds_R_crossed.npy")
            self.R[:, 0] = temp[0 : self.n_rds_trial]

            temp = np.load("../../../Data/rds/crds_R_uncrossed.npy")
            self.R[:, 1] = temp[0 : self.n_rds_trial]

        self.rds_type = rds_type_new

        # update size_rds_bg_pix
        self.size_rds_bg_pix = 120

    def resize_rds(self, new_dim):

        # n_rds_trial = 10240
        # new_dim = (100, 100)
        rds_resized_left = np.zeros(
            (self.n_rds_trial, 2, new_dim[0], new_dim[1]), dtype=np.int8
        )
        rds_resized_right = np.zeros(
            (self.n_rds_trial, 2, new_dim[0], new_dim[1]), dtype=np.int8
        )

        for n in range(self.n_rds_trial):

            ## left images
            temp = self.L[n, 0]  # crossed-disparity
            rds_resized_left[n, 0] = cv2.resize(
                temp, new_dim, interpolation=cv2.INTER_NEAREST
            )

            temp = self.L[n, 1]  # uncrossed-disparity
            rds_resized_left[n, 1] = cv2.resize(
                temp, new_dim, interpolation=cv2.INTER_NEAREST
            )

            ## right images
            temp = self.R[n, 0]  # crossed-disparity
            rds_resized_right[n, 0] = cv2.resize(
                temp, new_dim, interpolation=cv2.INTER_NEAREST
            )

            temp = self.R[n, 1]  # uncrossed-disparity
            rds_resized_right[n, 1] = cv2.resize(
                temp, new_dim, interpolation=cv2.INTER_NEAREST
            )

        self.L = rds_resized_left
        self.R = rds_resized_right

        # update size_rds_bg_pix
        self.size_rds_bg_pix = new_dim[0]


# class RF_CMM(RF_Neuron):
class RF_CMM:

    def __init__(self, RDS, f_batch, dispColMap_bootstrap, rf_centroid):
        """
        Parameters
        ----------
        f_batch : np.array
            a list of spatial frequency channels in cycles/deg
            ex: np.array([1, 2, 4, 8, 16]).astype(np.float32)

        dispPref : scalar
            The disparity preference of the BEM neuron, deg.

        dispPref_in_vox : [neurons_in_vox] np.array
            a list of disparity magnitude contained in a voxel of simulated
            disparity xcolumn map


        dispPhase_in_vox : [neurons_in_vox] np.array
            a list of phase disparity contained in a voxel of simulated disparity
            column map

            The RF disparity phase, deg.
            Subunit here is a pair of the left and right RFs.

            # phase disparity parameters

            ## important note:
            in order to make hmrds give proper disparity tuning with squared-BEM
            in position-encoding scheme,
            one possible configuration would be that left and right RF
            ((rf1_L and rf1_R) or (rf2_L and rf2_R)) should have np.pi phase
            difference.
            whereas dispPhi_batch1 and and dispPhi_batch2 should have
            np.pi/2 phase difference to keep the quadrature relationship.

            another important configuration: if (rf1_L and rf1_R) and (rf2_L and rf2_R)
            have the same phase (i.e. if the phase difference between subunit 1
            and 2 is 0), then hmrds disparity tuning will be inverted from crds.

            d_phi = 90 # phi difference between the first and second subunit (deg)
            phi_left = 0
            dispPhi_batch1_L = np.ones(rf_nBatch)*phi_left
            dispPhi_batch1_R = dispPhi_batch1_L
            dispPhi_batch2_L = dispPhi_batch1_L + d_phi
            dispPhi_batch2_R = dispPhi_batch1_R + d_phi

        rf_centroid : [2] np.array -> [x, y]
            The centroid (x, y) of the RF (the center of gaussian envelope,
            before shifted by the disparity prefernce), deg

        Returns
        -------
        None.

        """

        # super().__init__(f_batch, rf_centroid)
        self.size_rds_bg_pix = RDS.size_rds_bg_pix
        self.f_batch = f_batch
        self.rf_centroid = rf_centroid

        n_bootstrap, nVox_to_analyze, neurons_per_vox = dispColMap_bootstrap.shape

        self.n_bootstrap = n_bootstrap
        self.nVox_to_analyze = nVox_to_analyze

        # [n_bootstrap, nVox, neurons_per_vox]
        self.dispColMap_bootstrap = dispColMap_bootstrap

        # number of neurons in the simulated voxel
        self.neurons_per_vox = neurons_per_vox

        # the number of RF in a single vox
        self.n_rf_in_vox = len(f_batch) * neurons_per_vox

        # repeat each element in f_batch for neurons_per_vox times.
        # the following is the order of the repetiion:
        # the order of repeatition: [f1 f1 f1...         f2 f2 f2...         f3 f3 f3...]
        #                          neurons_per_vox   neurons_per_vox      neurons_per_vox
        self.f_batch_in_vox = np.array(
            [
                self.f_batch[f]
                for f in range(len(self.f_batch))
                for n in range(self.neurons_per_vox)
            ]
        )

        # set disparity preference, # [neurons_per_vox]
        # make a batch of dispPref
        # self.dispPref_in_vox = dispPref_in_vox

    def fwhm(self, f):
        """
        compute the FWHM of gaussian envelope at given spatial freq f.
        It is useful for estimating the square window size in cmm simulation
        without RF.

        Parameters
        ----------
        f : np.float or np.array
            spatial frequency.
        img_size_pix : TYPE
            size of the image in pixel.

        Returns
        -------
        fwhm_pix : np.int
            square window function size in pixel.

        """

        sigma = (np.sqrt(np.log(2) / 2) / (np.pi * f)) * (
            (2**1.5 + 1) / (2**1.5 - 1)
        )  # Read, nature 2007
        # sigma = 0.39/f

        # convert to pixel
        fwhm_pix = np.int(2.355 * sigma * self.size_rds_bg_pix)  # ref: wikipedia
        # fwhm_pix = np.int(2.355*sigma * 120)

        return fwhm_pix

    def create_square_window(self, disp_mu_x, f):
        """
        create a square window whose size depending on spatial freq f

        Parameters
        ----------
        f : scalar, np.float
            spatial frequency.

        Returns
        -------
        w : [size_rds_bg_pix, size_rds_bg_pix] np.array
            square window.

        """

        # disp_mu_in_pixel = 9 # the number of pixels associated with 0.2 deg
        disp_mu_in_pixel = self._compute_deg2pix(disp_mu_x)

        # compute fwhm, pixel
        fwhm_pix = self.fwhm(f)
        # fwhm_pix = fwhm(f)

        # define square window w
        w = np.zeros((self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int8)
        # w = np.zeros((size_rds_bg_pix, size_rds_bg_pix), dtype=np.int8)

        temp_start = np.int32(self.size_rds_bg_pix / 2 - fwhm_pix / 2)
        # id_row_start = np.int32(self.size_rds_bg_pix/2 - fwhm_pix/2)
        # use np.max to prevent id_row_start < 0
        id_row_start = max([0, temp_start])
        # id_row_start = np.int32(size_rds_bg_pix/2 - fwhm_pix/2)
        # w[0, 0] = id_row_start

        temp_end = temp_start + fwhm_pix
        # id_row_end = np.int32(id_row_start + fwhm_pix)
        # use np.min to prevent id_col_end > size_rds_bg_pix
        id_row_end = min([self.size_rds_bg_pix, temp_end])
        # w[0, 1] = id_row_end + 2 # plus 2 pixel to prevent zero area

        # use np.max to prevent id_col_start < 0
        temp_start = np.int32(
            self.size_rds_bg_pix / 2 - fwhm_pix / 2 - disp_mu_in_pixel
        )
        # temp_start = np.int32(size_rds_bg_pix/2 - fwhm_pix/2 - disp_mu_in_pixel)
        id_col_start = max([0, temp_start])
        # w[1, 0] = id_col_start

        # use np.min to prevent id_col_end > size_rds_bg_pix
        temp_end = temp_start + fwhm_pix
        id_col_end = min([self.size_rds_bg_pix, temp_end])
        # id_col_end = min([size_rds_bg_pix, temp_end])
        # w[1, 1] = id_col_end + 2 # plus 2 pixel to prevent zero area

        w[id_row_start:id_row_end, id_col_start:id_col_end] = np.int8(1)

        # plt.imshow(w)

        # w = np.zeros((120, 120), dtype=np.float32)
        # id_start = np.int(120/2 - fwhm_pix/2)
        # id_end = np.int(120/2 + fwhm_pix/2)
        # w[id_start:id_end, id_start:id_end] = 1

        return w

    def create_window_in_singleVox_singleFreq(self, dispPref_in_vox, f):
        """
        create a batch of square windows in a single voxel for a given
        spatial frequency referenced by the iterator f.

        Parameters
        ----------

        Returns
        -------
        w_batch : [neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
            a batch of square windows.



        Parameters
        ----------
        neurons_per_vox : scalar
            the number of neurons in a voxel.

        dispPref_in_vox : [neurons_in_vox] np.array
            a list of disparity magnitude (deg) contained in a voxel of simulated
            disparity column map

        f : scalar
            iterator for spatial frequency in f_batch.

        Returns
        -------
        window_in_vox: <[neurons_in_vox, size_rds_bg[0], size_rds_bg[1]] np.array>
                        RFs contained in a simulated voxel "vox".

        """

        f_this = self.f_batch[f]

        # get the number of RF in a voxel
        # [size_rds_bg, neurons_per_vox*size_rds_bg]
        window_in_vox = np.zeros(
            (self.neurons_per_vox, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )

        for n in range(self.neurons_per_vox):

            disp_mu_x_this = dispPref_in_vox[n]

            # get the RF centroid of each neuron in the simulated voxel.
            # it is assumed that all neurons in the voxel has the same centroid.
            # pos_rf_x = self.rf_centroid[0] # RF x coordinate
            # pos_rf_y = self.rf_centroid[1] # RF y coordinate
            # rf_centroid_this = (pos_rf_x, pos_rf_y)
            window_in_vox[n] = self.create_square_window(disp_mu_x_this, f_this)

        return window_in_vox

    def create_window_in_singleVox(self, i_bootstrap, v):
        """
        Create a batch of square window for disparity neurons contained in a single simulated
        voxel for all spatial frequency listed in self.f_batch.

        each window with spatial frequency f listed in self.f_batch is repeated
        neurons_per_vox times. Thus, there are len(f_batch) * neurons_per_vox
        windows in total.

        the order of repeatition: [f1 f1 f1...         f2 f2 f2...         f3 f3 f3...]
                                 neurons_per_vox   neurons_per_vox

        Outputs:
            - window_in_vox: <[nRF_in_vox, size_rds_bg[0], size_rds_bg[1]] np.array>
                        RFs contained in a simulated voxel "vox".

            note:
                    nRF_in_vox = len(f_batch) * neurons_per_vox

            note: vox_channels is similar to number of voxels used in cell column model
                  rfDisp_channels is similar to the number of RFs in a voxel

        """

        dispPref_in_vox = self.dispColMap_bootstrap[i_bootstrap, v]

        window_list = []
        window_list.append(
            Parallel(n_jobs=1)(
                delayed(self.create_window_in_singleVox_singleFreq)(dispPref_in_vox, f)
                for f in range(len(self.f_batch))
            )
        )

        # window_list = []
        # window_list.append(Parallel(n_jobs=1)
        #                    (delayed(window_cmm.create_window_in_singleVox_singleFreq)
        #                     (dispPref_in_vox, f)
        #                     for f in range(len(window_cmm.f_batch))))

        # [size_rds_bg, len(f_batch)*neurons_per_vox*size_rds_bg]
        window_in_vox = np.zeros(
            (self.n_rf_in_vox, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )  # RFs in a single vox

        for f in range(len(self.f_batch)):
            id_start = f * self.neurons_per_vox
            id_end = id_start + self.neurons_per_vox

            window_in_vox[id_start:id_end] = window_list[0][f]

        #    plt.imshow(np.sum(rf_vox, axis=0))

        return window_in_vox

    def create_window_in_allVox(self, dispColMap_bootstrap, i_bootstrap):
        """
        create square windows for all disparities listed in disparity column map
        (it means, a single bootstrap)

        Parameters
        ----------
        i_bootstrap : scalar
            bootstrap iterator.

        Returns
        -------
        window_vox : [nVox_to_analyze * len(f_batch) * neurons_per_vox,
                      size_rds_bg_pix, size_rds_bg_pix]

            a batch of square windows contained in nVox_to_analyze of simulated
            voxels.

        """

        window_allVox = _create_window_in_allVox(
            dispColMap_bootstrap,
            self.f_batch,
            self.nVox_to_analyze,
            self.size_rds_bg_pix,
            self.neurons_per_vox,
            i_bootstrap,
        )

        return window_allVox


# """ accelerated pg  -> sum x == 1 """
# def solve_pg(A, b, momentum=0.9, maxiter=1000):
#     """
#     https://stackoverflow.com/questions/44790116/constraint-the-sum-of-coefficients-with-scikit-learn-linear-model

#     remarks:
#             algorithm: accelerated projected gradient
#             projection: proj on probability-simplex
#                 -> naive and slow using cvxpy + ecos
#             line-search: armijo-rule along projection-arc (Bertsekas book)
#                 -> suffers from slow projection
#             stopping-criterion: naive
#             gradient-calculation: precomputes AtA
#                 -> not needed and not recommended for huge sparse data!
#     """

#     M, N = A.shape
#     x = np.zeros(N)

#     AtA = A.T.dot(A)
#     Atb = A.T.dot(b)

#     stop_count = 0

#     # projection helper
#     x_ = Variable(N)
#     v_ = Parameter(N)
#     objective_ = Minimize(0.5 * square(norm(x_ - v_, 2)))
#     constraints_ = [sum(x_) == 1]
#     problem_ = Problem(objective_, constraints_)

#     def gradient(x):
#         return AtA.dot(x) - Atb

#     def obj(x):
#         return 0.5 * np.linalg.norm(A.dot(x) - b)**2

#     it = 0
#     while True:
#         grad = gradient(x)

#         # line search
#         alpha = 1
#         beta = 0.5
#         sigma = 1e-2
#         old_obj = obj(x)
#         while True:
#             new_x = x - alpha * grad
#             new_obj = obj(new_x)
#             if old_obj - new_obj >= sigma * grad.dot(x - new_x):
#                 break
#             else:
#                 alpha *= beta

#         x_old = x[:]
#         x = x - alpha*grad

#         # projection
#         v_.value = x
#         problem_.solve()
#         x = np.array(x_.value.flat)

#         y = x + momentum * (x - x_old)

#         if np.abs(old_obj - obj(x)) < 1e-2:
#             stop_count += 1
#         else:
#             stop_count = 0

#         if stop_count == 3:
#             print('early-stopping @ it: ', it)
#             return x

#         it += 1

#         if it == maxiter:
#             return x


class Simulate_CMM(General):

    def __init__(
        self,
        RDS,
        f_batch,
        dispColMap_bootstrap,
        nVox_to_analyze,
        n_trial_epoch,
        n_trial_batch,
        rds_type,
        mtd,
    ):
        """


        Parameters
        ----------
        RDS : TYPE
            DESCRIPTION.

        f_batch : np.array
            a list of spatial frequencies, cycles/deg

        rf_bootstrap1_L : [n_booststrap, nVox, neurons_per_vox*len(f_batch),
                           size_rds_bg_pix, size_rds_bg_pix] np.array
            RF for subunit 1, left.

        rf_bootstrap1_R : [n_booststrap, nVox, neurons_per_vox*len(f_batch),
                           size_rds_bg_pix, size_rds_bg_pix] np.array
            RF for subunit 1, right.

        rf_bootstrap2_L : [n_booststrap, nVox, neurons_per_vox*len(f_batch),
                           size_rds_bg_pix, size_rds_bg_pix] np.array
            RF for subunit 2, left.

        rf_bootstrap2_R : [n_booststrap, nVox, neurons_per_vox*len(f_batch),
                           size_rds_bg_pix, size_rds_bg_pix] np.array
            RF for subunit 2, right.

        n_trial_epoch : TYPE
            the number of trial_epoch for rds.

        n_trial_batch : TYPE
            the number of trial_batch for rds.

        rds_type : TYPE
            DESCRIPTION.

        mtd : string
            "euclidean" or "sqeuclidean".
            the measure distance method used for computing RDM

        Returns
        -------
        None.

        """

        super().__init__()

        # set rds,  only has 2 disparities: crossed and uncrossed
        # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix] =
        # [n_trial, 2, size_rds_bg_pix, size_rds_bg_pix]
        self.rds_L = RDS.L  # left image,
        self.rds_R = RDS.R  # right image
        self.size_rds_bg_pix = RDS.size_rds_bg_pix

        n_rds = np.shape(RDS.L)[0]

        # set rds_u
        self.rds_u_L = RDS.u_L
        self.rds_u_R = RDS.u_R

        # set rds_type
        self.rds_type = rds_type
        self.n_trial_epoch = n_trial_epoch
        self.n_trial_batch = n_trial_batch
        self.n_rds = n_rds
        self.n_trial = n_trial_epoch * n_trial_batch

        # set rf, [n_bootstrap_batch, nVox*size_rds_bg_pix, len(f_batch)*neurons_per_vox*size_rds_bg_pix]
        # n_bootstrap_batch, nVox_to_analyze, n_rf = np.shape(rf_bootstrap_L)
        self.nVox_to_analyze = nVox_to_analyze

        # the number of neurons per voxel each spatial freq
        self.neurons_per_vox = dispColMap_bootstrap.shape[2]

        # calculate total neurons in a voxel across all spatial freq
        self.n_rf = len(f_batch) * self.neurons_per_vox

        self.f_batch = f_batch

        # get the number of pixel for fwhm of each spatial freq in f_batch
        self.compute_fwhm_pix_in_vox()  # self.fwhm_pix_in_vox, [len(f_batch)]

        # load rdm_fmri_all, [len(sbjID_all), n_ROIs, 6, 6]
        # rdm for empirical fmri data for each participant and roi.
        # the following is the way to compute rdm_fmri:
        # mtd = "sqeuclidean" # sqeuclidean gives the most make-sense corr_match weight ratio
        # rsa = RSA()
        # rsa.compute_rdm_all_sbjID(nVox_to_analyze, mtd)
        # rdm_fmri_all = rsa.rdm_fmri_all # [len(sbjID_all), len(ROIs), 6, 6]
        self.mtd = mtd
        self.rdm_fmri_all = np.load("../../../Data/CMM/rdm_fmri_all_{}.npy".format(mtd))

    def get_gpu_attr(self):
        device = cuda.get_current_device()
        attribs = [
            name.replace("CU_DEVICE_ATTRIBUTE_", "")
            for name in dir(enums)
            if name.startswith("CU_DEVICE_ATTRIBUTE_")
        ]
        for attr in attribs:
            print(attr, "=", getattr(device, attr))

    def generate_id_rds_bootstrap(self):

        id_rds_allBootstrap = _generate_id_rds_bootstrap(
            self.n_rds, self.n_trial, self.n_bootstrap_batch
        )

        return id_rds_allBootstrap

    def fwhm(self, f):
        """
        compute the FWHM of gaussian envelope at given spatial freq f.
        It is useful for estimating the square window size in cmm simulation
        without RF.

        Parameters
        ----------
        f : np.float or np.array
            spatial frequency.
        img_size_pix : TYPE
            size of the image in pixel.

        Returns
        -------
        fwhm_pix : np.int
            square window function size in pixel.

        """

        sigma = (np.sqrt(np.log(2) / 2) / (np.pi * f)) * (
            (2**1.5 + 1) / (2**1.5 - 1)
        )  # Read, nature 2007
        # sigma = 0.39/f

        # convert to pixel
        fwhm_pix = int(2.355 * sigma * self.size_rds_bg_pix)  # ref: wikipedia
        # fwhm_pix = np.int(2.355*sigma * 120)

        return fwhm_pix

    def compute_fwhm_pix_in_vox(self):
        """
        get the number of pixels required for fwhm of each spatial freq in f_batch

        Returns
        -------
        self.fwhm_pix_in_vox: [len(f_batch)] np.array.

        """

        n = len(self.f_batch)
        fwhm_pix_in_vox = np.zeros(n, dtype=np.int16)
        for i in range(n):

            f = self.f_batch[i]
            fwhm_pix = self.fwhm(f)

            fwhm_pix_in_vox[i] = fwhm_pix

        self.fwhm_pix_in_vox = fwhm_pix_in_vox

    def set_rds(self, RDS_new, rds_type_new):

        self.rds_L = RDS_new.L
        self.rds_R = RDS_new.R

        self.rds_type = rds_type_new

    def generate_rds_batch_trial(self, rds, n_trial_batch):
        """
        generate n_trials of rds. It uses module _generate_rds_batch_trial

        Parameters
        ----------
        rds : class
            rds class, consists of rds.L and rds.R and other attributes

            rds.L : [n_rds, 2, size_rds_bg_pix, size_rds_bg_pix] np.int8
                rds images for the left eye
                n_rds = 10240

                the second axis ("2") indicates crossed and uncrossed rds images

            rds.R : [n_rds, 2, size_rds_bg_pix, size_rds_bg_pix] np.int8
                rds images for the right eye
                n_rds = 10240

        n_trial_batch : scalar
            the number of trial_batch.

        Returns
        -------
        rds_LR_CU_trial : [n_trial_total * 2 * nVox_to_analyze * len(f_batch) * neurons_per_vox,
                           size_rds_bg_pix, size_rds_bg_pix]
                        np.int8

            a very big rds matrix. It stores 4 RDSs in the following order:
                rds_L_CU_trial[0, :, :] -> rds_left_crossed, trial 1
                rds_L_CU_trial[1, :, :] -> rds_left_uncrossed, trial 1
                rds_L_CU_trial[2, :, :] -> rds_left_crossed, trial 2
                rds_L_CU_trial[3, :, :] -> rds_right_crossed, trial 2

                rds_R_CU_trial[6, :, :] -> rds_right_crossed, trial 2
                rds_R_CU_trial[8, :, :] -> rds_right_uncrossed, trial 2

                ...

        """

        id_rds = np.random.choice(np.arange(self.n_rds), n_trial_batch, replace=False)
        # id_rds = np.random.choice(np.arange(10240), n_trial_batch, replace=False)

        n_rds_single_trial = (
            self.nVox_to_analyze * len(self.f_batch) * self.neurons_per_vox
        )

        rds_L_CU_trial, rds_R_CU_trial = _generate_rds_batch_trial(
            rds.L,
            rds.R,
            id_rds,
            self.size_rds_bg_pix,
            n_rds_single_trial,
            n_trial_batch,
        )

        return rds_L_CU_trial, rds_R_CU_trial

    def preprocess_cmmResp(self, cmmResp, n_trial_batch, n_trial_total):

        cmmResp_crossed, cmmResp_uncrossed = _preprocess_cmm(
            cmmResp,
            self.f_batch,
            self.nVox_to_analyze,
            self.neurons_per_vox,
            n_trial_batch,
            n_trial_total,
        )

        ################################################################################################################
        ## data preprocessing
        ################################################################################################################

        # corrResp_ards = np.load("../../../Data/CMM/corrResp_ards_dispColNoise_{}.npy"
        #                         .format(np.round(sawtooth_noise_std, 2)))
        # corrResp_crossed_ards, \
        #     corrResp_uncrossed_ards = cmm.preprocess_cmmResp(corrResp_ards,
        #                                                     n_trial_batch,
        #                                                     n_trial_total)
        # del corrResp_ards

        # matchResp_ards = np.load("../../../Data/CMM/matchResp_ards_dispColNoise_{}.npy"
        #                         .format(np.round(sawtooth_noise_std, 2)))
        # matchResp_crossed_ards, \
        #     matchResp_uncrossed_ards = cmm.preprocess_cmmResp(matchResp_ards,
        #                                                     n_trial_batch,
        #                                                     n_trial_total)
        # del matchResp_ards

        # corrResp_hmrds = np.load("../../../Data/CMM/corrResp_hmrds_dispColNoise_{}.npy"
        #                         .format(np.round(sawtooth_noise_std, 2)))
        # corrResp_crossed_hmrds, \
        #     corrResp_uncrossed_hmrds = cmm.preprocess_cmmResp(corrResp_hmrds,
        #                                                     n_trial_batch,
        #                                                     n_trial_total)
        # del corrResp_hmrds

        # matchResp_hmrds = np.load("../../../Data/CMM/matchResp_hmrds_dispColNoise_{}.npy"
        #                         .format(np.round(sawtooth_noise_std, 2)))
        # matchResp_crossed_hmrds, \
        #     matchResp_uncrossed_hmrds = cmm.preprocess_cmmResp(matchResp_hmrds,
        #                                                     n_trial_batch,
        #                                                     n_trial_total)
        # del matchResp_hmrds

        # corrResp_crds = np.load("../../../Data/CMM/corrResp_crds_dispColNoise_{}.npy"
        #                         .format(np.round(sawtooth_noise_std, 2)))
        # corrResp_crossed_crds, \
        #     corrResp_uncrossed_crds = cmm.preprocess_cmmResp(corrResp_crds,
        #                                                     n_trial_batch,
        #                                                     n_trial_total)
        # del corrResp_crds

        # matchResp_crds = np.load("../../../Data/CMM/matchResp_crds_dispColNoise_{}.npy"
        #                         .format(np.round(sawtooth_noise_std, 2)))
        # matchResp_crossed_crds, \
        #     matchResp_uncrossed_crds = cmm.preprocess_cmmResp(matchResp_crds,
        #                                                     n_trial_batch,
        #                                                     n_trial_total)
        # del matchResp_crds

        return cmmResp_crossed, cmmResp_uncrossed

    def simulate_cmm_rds_cpu(self, rds_type, id_rds_allBootstrap):
        """
        simulate squared binocular energy model for a single bootstrap.

        Parameters
        ----------
        i_bootstrap : scalar
            bootstrap iterator.

        Returns
        -------
        complexResp_all : [nVox_to_analyze, neurons_per_vox*len(f_batch),
                           crossed-uncrossed, n_trial_batch] np.array
            complex cell response based on squared BEM.

        """

        if self.rds_type != rds_type:
            rds_new = RDS(
                self.n_rds,
                self.rDot,
                self.dotDens,
                self.size_rds_bg_deg,
                self.size_rds_ct_deg,
                self.deg_per_pix,
            )

            # load rds associated with rds_type
            rds_new.load_rds(rds_type)

            # set new rds_type
            self.set_rds(rds_new, rds_type)

        corrResp_all, matchResp_all = _simulate_cmm_rds_cpu(
            self.rds_L,
            self.rds_R,
            self.rf_bootstrap_L,
            self.rf_bootstrap_R,
            self.nVox_to_analyze,
            self.n_rf,
            id_rds_allBootstrap,
            self.n_rds,
            self.n_trial_batch,
            self.n_bootstrap,
        )
        # tok = timer()
        # print("time elapsed = {} sec".format(str(tok - tik)))

        # return corrResp_all, matchResp_all, matchResp_squared_all
        return corrResp_all, matchResp_all

    def simulate_cmm_rds(self, rds_type):
        """
        simulate squared BEM for a given rds_type (ards, hmrds, or crds).

        The simulation is repeated for n_bootstrap.

        Parameters
        ----------
        rds_type : string
            a string denoting the type of rds (ards, hmrds, or crds).

        Returns
        -------
        corrResp_rds : [n_bootstrap, nVox_to_analyze, n_rf, 2, n_trial_batch] np.array
            cross-correlation response

        matchResp_rds : [n_bootstrap, nVox_to_analyze, n_rf, 2, n_trial_batch] np.array
            cross-matching response



        """

        if self.rds_type != rds_type:
            rds_new = RDS(
                self.n_rds,
                self.rDot,
                self.dotDens,
                self.size_rds_bg_deg,
                self.size_rds_ct_deg,
                self.deg_per_pix,
            )

            # load rds associated with rds_type
            rds_new.load_rds(rds_type)

            # set new rds_type
            self.set_rds(rds_new, rds_type)

        corrResp_rds = np.zeros(
            (self.n_bootstrap, self.nVox_to_analyze, self.n_rf, 2, self.n_trial_batch),
            dtype=np.float32,
        )
        matchResp_rds = np.zeros(
            (self.n_bootstrap, self.nVox_to_analyze, self.n_rf, 2, self.n_trial_batch),
            dtype=np.float32,
        )
        # matchResp_squared_rds = np.zeros((self.n_bootstrap, self.nVox_to_analyze,
        #                                   self.n_rf, 2, self.n_trial_batch),
        #                                 dtype=np.float32)

        for i in range(self.n_bootstrap):

            # corrResp_rds[i], \
            #     matchResp_rds[i], \
            #         matchResp_squared_rds[i] = self.simulate_cmm_singleBootstrap(i)

            corrResp_rds[i], matchResp_rds[i] = self.simulate_cmm_singleBootstrap(i)

        # return corrResp_rds, matchResp_rds, matchResp_squared_rds
        return corrResp_rds, matchResp_rds

    def simulate_cmm_all_rds(self):
        """
        simulate cmm for all types of rds

        Returns
        -------
        complexResp_all_rds : [len(rds_type_list), n_bootstrap, nVox_to_analyze,
                               neurons_per_vox*len(f_batch),
                               crossed-uncrossed, n_trial] np.array

            complex cell response based on the squared BEM for each rds_type.

        """

        rds_type_list = ["ards", "hmrds", "crds"]
        corrResp_all_rds = np.zeros(
            (
                len(rds_type_list),
                self.n_bootstrap,
                self.nVox_to_analyze,
                self.n_rf,
                2,
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        matchResp_all_rds = np.zeros(
            (
                len(rds_type_list),
                self.n_bootstrap,
                self.nVox_to_analyze,
                self.n_rf,
                2,
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        matchResp_squared_all_rds = np.zeros(
            (
                len(rds_type_list),
                self.n_bootstrap,
                self.nVox_to_analyze,
                self.n_rf,
                2,
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )

        for r in range(len(rds_type_list)):

            rds_type = rds_type_list[r]

            # simulate ards
            corrResp_all_rds[r], matchResp_all_rds[r], matchResp_squared_all_rds[r] = (
                self.simulate_cmm_rds(rds_type)
            )

        return corrResp_all_rds, matchResp_all_rds, matchResp_squared_all_rds

    def compute_rdm_fmri(self, nVox_to_analyze, mtd):

        # mtd = "sqeuclidean"
        # rdm_fmri_all = cmm.rdm_fmri_all
        rdm_fmri_all = np.zeros((self.n_sbjID, self.n_ROIs, 6, 6), dtype=np.float32)

        for sbj in range(self.n_sbjID):

            sbjID = self.sbjID_all[sbj]

            for roi in range(self.n_ROIs):

                # load vtc_norm, vtc data that has been shifted backward 2TR and z-scored
                # the voxels in this vtc_norm data has been sorted in descending order
                # according to t-value
                vtc_norm = pd.read_pickle(
                    "../../../../Data/VTC_normalized/vtc_shift_norm_{}_{}.pkl".format(
                        sbjID, nVox_to_analyze
                    )
                )

                ## ards_crossed
                stimID = 1
                vtc_vox = vtc_norm.loc[
                    (vtc_norm.roi == roi)
                    & (vtc_norm.stimID == stimID)
                    & (vtc_norm.vox.isin(range(nVox_to_analyze)))
                ]
                vtc_vox_x = vtc_vox.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
                )
                ards_crossed = np.array(vtc_vox_x, dtype=np.float32).mean(axis=0)

                ## ards_uncrossed
                stimID = 2
                vtc_vox = vtc_norm.loc[
                    (vtc_norm.roi == roi)
                    & (vtc_norm.stimID == stimID)
                    & (vtc_norm.vox.isin(range(nVox_to_analyze)))
                ]
                vtc_vox_x = vtc_vox.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
                )
                ards_uncrossed = np.array(vtc_vox_x, dtype=np.float32).mean(axis=0)

                ## hmrds_crossed
                stimID = 3
                vtc_vox = vtc_norm.loc[
                    (vtc_norm.roi == roi)
                    & (vtc_norm.stimID == stimID)
                    & (vtc_norm.vox.isin(range(nVox_to_analyze)))
                ]
                vtc_vox_x = vtc_vox.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
                )
                hmrds_crossed = np.array(vtc_vox_x, dtype=np.float32).mean(axis=0)

                ## hmrds_uncrossed
                stimID = 4
                vtc_vox = vtc_norm.loc[
                    (vtc_norm.roi == roi)
                    & (vtc_norm.stimID == stimID)
                    & (vtc_norm.vox.isin(range(nVox_to_analyze)))
                ]
                vtc_vox_x = vtc_vox.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
                )
                hmrds_uncrossed = np.array(vtc_vox_x, dtype=np.float32).mean(axis=0)

                ## crds_crossed
                stimID = 5
                vtc_vox = vtc_norm.loc[
                    (vtc_norm.roi == roi)
                    & (vtc_norm.stimID == stimID)
                    & (vtc_norm.vox.isin(range(nVox_to_analyze)))
                ]
                vtc_vox_x = vtc_vox.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
                )
                crds_crossed = np.array(vtc_vox_x, dtype=np.float32).mean(axis=0)

                ## crds_uncrossed
                stimID = 6
                vtc_vox = vtc_norm.loc[
                    (vtc_norm.roi == roi)
                    & (vtc_norm.stimID == stimID)
                    & (vtc_norm.vox.isin(range(nVox_to_analyze)))
                ]
                vtc_vox_x = vtc_vox.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
                )
                crds_uncrossed = np.array(vtc_vox_x, dtype=np.float32).mean(axis=0)

                # compute rdm_fmri
                temp = np.array(
                    [
                        ards_crossed,
                        ards_uncrossed,
                        hmrds_crossed,
                        hmrds_uncrossed,
                        crds_crossed,
                        crds_uncrossed,
                    ]
                )
                rdm_fmri_all[sbj, roi] = _cdist(temp, temp, mtd)  # [6, 6]

        ## save rdm_fmri_all
        np.save("../../../../Data/CMM/rdm_fmri_all_{}.npy".format(mtd), rdm_fmri_all)

        return rdm_fmri_all

    def compute_rdm_cmm(
        self,
        corrResp_crossed_ards,
        corrResp_uncrossed_ards,
        corrResp_crossed_hmrds,
        corrResp_uncrossed_hmrds,
        corrResp_crossed_crds,
        corrResp_uncrossed_crds,
        matchResp_crossed_ards,
        matchResp_uncrossed_ards,
        matchResp_crossed_hmrds,
        matchResp_uncrossed_hmrds,
        matchResp_crossed_crds,
        matchResp_uncrossed_crds,
        n_bootstrap,
        mtd,
    ):
        """
        compute representational dissimilarity matrix (rdm) based on CMM.
        the rdm is the averaged across spatial frequency.

        Parameters
        ----------
        corrResp_crossed_ards : [n_bootstrap, n_trial_total, nVox_to_analyze]
                                np.array
        corrResp_uncrossed_ards : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        corrResp_crossed_hmrds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        corrResp_uncrossed_hmrds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        corrResp_crossed_crds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        corrResp_uncrossed_crds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array

        matchResp_crossed_ards : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        matchResp_uncrossed_ards : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        matchResp_crossed_hmrds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        matchResp_uncrossed_hmrds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        matchResp_crossed_crds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array
        matchResp_uncrossed_crds : [n_bootstrap, n_trial_total, nVox_to_analyze] np.array

        cross-correlation response based on CMM for each rds_type.

        matchResp_all_rds : [len(rds_type_list), n_bootstrap, nVox_to_analyze,
                               neurons_per_vox*len(f_batch),
                               crossed-uncrossed, n_trial_batch] np.array

        cross-matching response based on CMM for each rds_type.

        Returns
        -------
        rdm_corr : [n_bootstrap, 6, 6] np.array
            rdm for cross-correlation, all bootstrap.

        rdm_match : [n_bootstrap, 6, 6] np.array
            rdm for cross-matching, all bootstrap.

        """

        # average across n_trial_batch
        corrResp_ards_crossed_avg = np.mean(corrResp_crossed_ards, axis=1)
        corrResp_ards_uncrossed_avg = np.mean(corrResp_uncrossed_ards, axis=1)
        corrResp_hmrds_crossed_avg = np.mean(corrResp_crossed_hmrds, axis=1)
        corrResp_hmrds_uncrossed_avg = np.mean(corrResp_uncrossed_hmrds, axis=1)
        corrResp_crds_crossed_avg = np.mean(corrResp_crossed_crds, axis=1)
        corrResp_crds_uncrossed_avg = np.mean(corrResp_uncrossed_crds, axis=1)

        matchResp_ards_crossed_avg = np.mean(matchResp_crossed_ards, axis=1)
        matchResp_ards_uncrossed_avg = np.mean(matchResp_uncrossed_ards, axis=1)
        matchResp_hmrds_crossed_avg = np.mean(matchResp_crossed_hmrds, axis=1)
        matchResp_hmrds_uncrossed_avg = np.mean(matchResp_uncrossed_hmrds, axis=1)
        matchResp_crds_crossed_avg = np.mean(matchResp_crossed_crds, axis=1)
        matchResp_crds_uncrossed_avg = np.mean(matchResp_uncrossed_crds, axis=1)

        rdm_corr = np.zeros((n_bootstrap, 6, 6), dtype=np.float32)
        rdm_match = np.zeros((n_bootstrap, 6, 6), dtype=np.float32)
        for i in range(n_bootstrap):

            # cross-correlation rdm
            temp = np.array(
                [
                    corrResp_ards_crossed_avg[i],
                    corrResp_ards_uncrossed_avg[i],
                    corrResp_hmrds_crossed_avg[i],
                    corrResp_hmrds_uncrossed_avg[i],
                    corrResp_crds_crossed_avg[i],
                    corrResp_crds_uncrossed_avg[i],
                ]
            )

            rdm_corr[i] = _cdist(temp, temp, mtd)  # [6, 6]
            # rdm_corr[i] = c*(2 - cdist(temp, temp, mtd))

            # cross-correlation rdm
            temp = np.array(
                [
                    matchResp_ards_crossed_avg[i],
                    matchResp_ards_uncrossed_avg[i],
                    matchResp_hmrds_crossed_avg[i],
                    matchResp_hmrds_uncrossed_avg[i],
                    matchResp_crds_crossed_avg[i],
                    matchResp_crds_uncrossed_avg[i],
                ]
            )

            rdm_match[i] = _cdist(temp, temp, mtd)
            # rdm_match[i] = c*(2 - cdist(temp, temp, mtd))

        return rdm_corr, rdm_match

    def compute_rdm_cmm_each_freq(
        self,
        corrResp_ards,
        corrResp_hmrds,
        corrResp_crds,
        matchResp_ards,
        matchResp_hmrds,
        matchResp_crds,
        mtd,
    ):
        """
        generate rdm for cmm model for each spatial frequency.

        Parameters
        ----------
        corrResp_ards : [n_bootstrap, nVox, f_batch_times_neurons_per_vox,
                        crossed_uncrossed, n_trial_batch] np.array
            simulated voxel responses to ards for cross-correlation based on cmm without RF.

        corrResp_hmrds : [n_bootstrap, nVox, f_batch_times_neurons_per_vox,
                         crossed_uncrossed, n_trial_batch] np.array
            simulated voxel responses to hmrds for cross-correlation based on cmm without RF.

        corrResp_crds : [n_bootstrap, nVox, f_batch_times_neurons_per_vox,
                        crossed_uncrossed, n_trial_batch] np.array
            simulated voxel responses to crds for cross-correlation based on cmm without RF.

        matchResp_ards : [n_bootstrap, nVox, f_batch_times_neurons_per_vox,
                        crossed_uncrossed, n_trial_batch] np.array
            simulated voxel responses to ards for cross-matching based on cmm without RF.

        matchResp_hmrds : [n_bootstrap, nVox, f_batch_times_neurons_per_vox,
                         crossed_uncrossed, n_trial_batch] np.array
            simulated voxel responses to hmrds for cross-matching based on cmm without RF.

        matchResp_crds : [n_bootstrap, nVox, f_batch_times_neurons_per_vox,
                        crossed_uncrossed, n_trial_batch] np.array
            simulated voxel responses to crds for cross-matching based on cmm without RF.

        f_batch : for example: np.array([1, 2, 4, 8, 16]).astype(np.float32) # spatial frequency
            list of spatial frequency.

        nVox_to_analyze : scalar
            the number of voxels used for the analysis

        Returns
        -------
        rdm_corr_mean : [len(f_batch), 6, 6]
            rdm_corr that has been averaged across n_bootstrap.
            obtained from compute_rdm_cmm.
        rdm_match_mean : [len(f_batch), 6, 6]
            rdm_match that has been averaged across n_bootstrap.
            obtained from compute_rdm_cmm.

        """

        # breakdown into frequency channels, corr-match, crossed-uncrossed
        # [n_bootstrap, nVox_to_analyze, len(f_batch), n_trial_batch]
        corr_ards_crossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        corr_ards_uncrossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        match_ards_crossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        match_ards_uncrossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )

        corr_hmrds_crossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        corr_hmrds_uncrossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        match_hmrds_crossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        match_hmrds_uncrossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )

        corr_crds_crossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        corr_crds_uncrossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        match_crds_crossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )
        match_crds_uncrossed = np.zeros(
            (
                self.n_bootstrap,
                self.nVox_to_analyze,
                len(self.f_batch),
                self.n_trial_batch,
            ),
            dtype=np.float32,
        )

        for f in range(len(self.f_batch)):

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            f_start = f * self.neurons_per_vox
            f_end = f_start + self.neurons_per_vox
            # f_start = f*neurons_per_vox
            # f_end = f_start + neurons_per_vox

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = corrResp_ards[:, :, f_start:f_end, 0]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_ards_crossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = corrResp_ards[:, :, f_start:f_end, 1]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_ards_uncrossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = matchResp_ards[:, :, f_start:f_end, 0]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_ards_crossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = matchResp_ards[:, :, f_start:f_end, 1]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_ards_uncrossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = corrResp_hmrds[:, :, f_start:f_end, 0]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_hmrds_crossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = corrResp_hmrds[:, :, f_start:f_end, 1]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_hmrds_uncrossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = matchResp_hmrds[:, :, f_start:f_end, 0]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_hmrds_crossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = matchResp_hmrds[:, :, f_start:f_end, 1]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_hmrds_uncrossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = corrResp_crds[:, :, f_start:f_end, 0]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_crds_crossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = corrResp_crds[:, :, f_start:f_end, 1]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_crds_uncrossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = matchResp_crds[:, :, f_start:f_end, 0]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_crds_crossed[:, :, f] = voxResp

            # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            temp = matchResp_crds[:, :, f_start:f_end, 1]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_crds_uncrossed[:, :, f] = voxResp

        # average across n_trial
        # [n_bootstrap, nVox_to_analyze, len(f_batch)]
        corr_ards_crossed_mean = np.mean(corr_ards_crossed, axis=3)
        corr_ards_uncrossed_mean = np.mean(corr_ards_uncrossed, axis=3)
        corr_hmrds_crossed_mean = np.mean(corr_hmrds_crossed, axis=3)
        corr_hmrds_uncrossed_mean = np.mean(corr_hmrds_uncrossed, axis=3)
        corr_crds_crossed_mean = np.mean(corr_crds_crossed, axis=3)
        corr_crds_uncrossed_mean = np.mean(corr_crds_uncrossed, axis=3)

        match_ards_crossed_mean = np.mean(match_ards_crossed, axis=3)
        match_ards_uncrossed_mean = np.mean(match_ards_uncrossed, axis=3)
        match_hmrds_crossed_mean = np.mean(match_hmrds_crossed, axis=3)
        match_hmrds_uncrossed_mean = np.mean(match_hmrds_uncrossed, axis=3)
        match_crds_crossed_mean = np.mean(match_crds_crossed, axis=3)
        match_crds_uncrossed_mean = np.mean(match_crds_uncrossed, axis=3)

        # create rdm
        if mtd == "correlation":
            c = 0.5
        else:
            c = 1.0

        rdm_corr = np.zeros(
            (self.n_bootstrap, len(self.f_batch), 6, 6), dtype=np.float32
        )
        rdm_match = np.zeros(
            (self.n_bootstrap, len(self.f_batch), 6, 6), dtype=np.float32
        )

        for i in range(self.n_bootstrap):
            for f in range(len(self.f_batch)):

                # corr RDM
                temp = np.array(
                    [
                        corr_ards_crossed_mean[i, :, f],
                        corr_ards_uncrossed_mean[i, :, f],
                        corr_hmrds_crossed_mean[i, :, f],
                        corr_hmrds_uncrossed_mean[i, :, f],
                        corr_crds_crossed_mean[i, :, f],
                        corr_crds_uncrossed_mean[i, :, f],
                    ]
                )

                rdm_corr[i, f] = c * cdist(temp, temp, mtd)
                # rdm_corr[i, f] = c*(2 - cdist(temp, temp, mtd))

                # match RDM
                temp = np.array(
                    [
                        match_ards_crossed_mean[i, :, f],
                        match_ards_uncrossed_mean[i, :, f],
                        match_hmrds_crossed_mean[i, :, f],
                        match_hmrds_uncrossed_mean[i, :, f],
                        match_crds_crossed_mean[i, :, f],
                        match_crds_uncrossed_mean[i, :, f],
                    ]
                )
                # r = pearsonr
                rdm_match[i, f] = c * cdist(temp, temp, mtd)
                # rdm_match[i, f] = c*(2 - cdist(temp, temp, mtd))

        # average rdm across bootstrap
        # rdm_corr_mean = np.mean(rdm_corr, axis=0)
        # rdm_match_mean = np.mean(rdm_match, axis=0)

        return rdm_corr, rdm_match

    def compute_w_cmm_singleBootstrap(self, rdm_fmri_all, x, n_bootstrap):
        """
        compute w_bem with bootstrap

        Parameters
        ----------
        rdm_corr : [n_bootstrap, 6, 6]
            rdm_bem obtained from self.compute_rdm_bem

        n_bootstrap : scalar
            number of bootsrap iteration.

        Returns
        -------
        w_cmm_singleBootstrap : [len(sbjID_all), len(self.ROIs), 2] np.array
            estimated weights of squared bem for

            w_cmm_singleBootstrap[:, :, 0] -> w_bem
            w_cmm_singleBootstrap[:, :, 1] -> dc value

        kendall_singleBootstrap : [len(self.ROIs), 2] np.array
            kendall tau statistic.

            kendall_singleBootstrap[:, 0] -> kendalltau correlation
            kendall_singleBootstrap[:, 1] -> p_val

        """

        # ## average rdm_bem across bootstrap
        # rdm_corr_mean = np.mean(rdm_corr, axis=0)
        # rdm_match_mean = np.mean(rdm_match, axis=0)

        # ## prepare regressor
        # # get above diagonal elements
        # rdm_corr_above = rdm_corr_mean[np.triu_indices(6, k=1)]
        # rdm_match_above = rdm_match_mean[np.triu_indices(6, k=1)]

        # #  mean-zeroing
        # rdm_corr_mean_zero = (rdm_corr_above - np.mean(rdm_corr_above))
        # rdm_match_mean_zero = (rdm_match_above - np.mean(rdm_match_above))

        # # normalize by dividing max value
        # rdm_corr_vec = rdm_corr_mean_zero/np.max(rdm_corr_mean_zero)
        # rdm_match_vec = rdm_match_mean_zero/np.max(rdm_match_mean_zero)

        # x = np.vstack([rdm_corr_vec,
        #                rdm_match_vec,
        #                np.ones(len(rdm_corr_vec))])

        # n_bootstrap = 1000
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_singleBootstrap = np.zeros((self.n_ROIs, 2), dtype=np.float32)
        w_cmm_singleBootstrap = np.zeros((self.n_ROIs, 3), dtype=np.float32)

        # random sampling rdm_fmri_all [len(sbjID_all), n_ROIs, 6, 6]
        id_sample = np.random.randint(self.n_sbjID, size=self.n_sbjID)
        # id_sample = np.random.randint(22, size=22)

        for roi in range(self.n_ROIs):
            # for i in range(n_bootstrap):

            # print("compute w_cmm, roi: {}, bootstrap: {}/{}"
            #       .format(self.ROIs[roi], str(i_bootstrap), str(n_bootstrap)))

            rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]

            # y_true_sbj = np.zeros((self.n_sbjID, 15), dtype=np.float32)
            # y_pred_sbj = np.zeros((self.n_sbjID, 15), dtype=np.float32)
            # for sbj in range(self.n_sbjID):

            # rdm_fmri_sbj = rdm_fmri_bootstrap[sbj]
            rdm_fmri_roi = np.mean(rdm_fmri_bootstrap, axis=0)

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_vec = rdm_fmri_above - np.mean(rdm_fmri_above)

            # normalize by dividing max value
            # rdm_fmri_vec = rdm_fmri_vec/np.max(rdm_fmri_vec)
            rdm_fmri_vec = rdm_fmri_vec / np.sqrt(np.sum(rdm_fmri_vec**2))
            # rdm_fmri_vec = rdm_fmri_vec/sem(rdm_fmri_vec)

            # # start fitting
            # w = solve_pg(x.T, rdm_fmri_vec)
            # # w, resid = nnls(x.T, rdm_fmri_vec)
            # w_cmm_singleBootstrap[roi] = w

            clf = Ridge(alpha=1, solver="lbfgs", positive=True, max_iter=1000)

            # start fitting using Lasso regression
            # clf = Lasso(alpha=1)
            clf.fit(x.T, rdm_fmri_vec)
            w = clf.coef_
            w_cmm_singleBootstrap[roi] = w

            # normalize w such that w_corr + w_match = 1
            # w_corr = w[0]
            # w_match = w[1]
            # w_corr_norm = w_corr/(w_corr + w_match + 1e-6)
            # w_match_norm = w_match/(w_corr + w_match + 1e-6)

            # w_bootstrap[i, sbj, roi, 0] = w_corr_norm
            # w_bootstrap[i, sbj, roi, 1] = w_match_norm
            # w_bootstrap[i, sbj, roi, 2] = w[2]

            # compute r2
            # y_true_sbj[sbj] = rdm_corr_vec
            # y_true_sbj[sbj] = rdm_fmri_vec
            y_pred = np.matmul(w, x)

            # r2_bootstrap[i, sbj, roi] = r2_score(rdm_fmri_vec, y_pred)
            # kendall_singleBootstrap[roi] = kendalltau(np.mean(y_true_sbj, axis=0),
            #                                           np.mean(y_pred_sbj, axis=0))
            kendall_singleBootstrap[roi] = kendalltau(rdm_fmri_vec, y_pred)

        return w_cmm_singleBootstrap, kendall_singleBootstrap

    def compute_w_cmm_bootstrap(self, rdm_corr, rdm_match, n_bootstrap):
        """
        compute w_bem with bootstrap

        Parameters
        ----------
        rdm_bem : [n_bootstrap, 6, 6]
            rdm_bem obtained from self.compute_rdm_bem

        n_bootstrap : scalar
            number of bootsrap iteration.

        Returns
        -------
        w_bem_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 2] np.array
            estimated weights of squared bem for

            w_bootstrap[:, :, :, 0] -> w_bem
            w_bootstrap[:, :, :, 1] -> dc value

        kendall_bootstrap : [n_bootstrap, len(self.ROIs), 2] np.array
            kendall tau statistic.

            kendall_bootstrap[:, :, 0] -> kendalltau correlation
            kendall_bootstrap[:, :, 1] -> p_val

        """

        # average rdm_bem across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        # prepare regressor
        # get above diagonal elements
        rdm_corr_above = rdm_corr_mean[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_mean[np.triu_indices(6, k=1)]

        #  mean-zeroing
        rdm_corr_mean_zero = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_match_mean_zero = rdm_match_above - np.mean(rdm_match_above)

        # normalize by dividing max value
        # rdm_corr_vec = rdm_corr_mean_zero/np.max(rdm_corr_mean_zero)
        # rdm_match_vec = rdm_match_mean_zero/np.max(rdm_match_mean_zero)
        rdm_corr_vec = rdm_corr_mean_zero / np.sqrt(np.sum(rdm_corr_mean_zero**2))
        rdm_match_vec = rdm_match_mean_zero / np.sqrt(np.sum(rdm_match_mean_zero**2))

        x = np.vstack([rdm_corr_vec, rdm_match_vec, np.ones(len(rdm_corr_vec))])

        w_cmm = []
        w_cmm.append(
            Parallel(n_jobs=1)(
                delayed(self.compute_w_cmm_singleBootstrap)(
                    self.rdm_fmri_all, x, n_bootstrap
                )
                for i in range(n_bootstrap)
            )
        )
        # w_cmm.append(Parallel(n_jobs=-1)
        #               (delayed(cmm.compute_w_cmm_singleBootstrap)
        #               (rdm_fmri_all, x, n_bootstrap, i)
        #               for i in range(n_bootstrap)))

        # unpack
        # n_bootstrap = 1000
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 2), dtype=np.float32)
        w_cmm_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 3), dtype=np.float32)
        # kendall_bootstrap = np.zeros((n_bootstrap, 8, 2),
        #                              dtype=np.float32)
        # w_cmm_bootstrap = np.zeros((n_bootstrap, 22, 8, 3),
        #                            dtype=np.float32)

        for i in range(n_bootstrap):
            # for i in range(n_bootstrap):
            temp = w_cmm[0][i][0]
            w_cmm_bootstrap[i] = temp

            temp = w_cmm[0][i][1]
            kendall_bootstrap[i] = temp

        return w_cmm_bootstrap, kendall_bootstrap

    def compute_w_cmm_singleSbjID_singleBootstrap(
        self, rdm_fmri_all, rdm_corr, rdm_match, sbjID, i_bootstrap
    ):
        """
        compute w_cmm for a single subject

        Parameters
        ----------
        rdm_corr : [n_bootstrap, 6, 6]
            rdm_bem obtained from self.compute_rdm_bem

        n_bootstrap : scalar
            number of bootsrap iteration.

        Returns
        -------
        w_cmm_singleBootstrap : [len(sbjID_all), len(self.ROIs), 2] np.array
            estimated weights of squared bem for

            w_cmm_singleBootstrap[:, :, 0] -> w_bem
            w_cmm_singleBootstrap[:, :, 1] -> dc value

        kendall_singleBootstrap : [len(self.ROIs), 2] np.array
            kendall tau statistic.

            kendall_singleBootstrap[:, 0] -> kendalltau correlation
            kendall_singleBootstrap[:, 1] -> p_val

        """

        # get rdm_corr and rdm_match at i_bootstrap
        rdm_corr_bootstrap = rdm_corr[i_bootstrap]
        rdm_match_bootstrap = rdm_match[i_bootstrap]

        # prepare regressor
        # get above diagonal elements
        rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

        #  mean-zeroing
        rdm_corr_mean_zero = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_match_mean_zero = rdm_match_above - np.mean(rdm_match_above)

        # normalize by dividing max value
        # rdm_corr_vec = rdm_corr_mean_zero/np.max(rdm_corr_mean_zero)
        # rdm_match_vec = rdm_match_mean_zero/np.max(rdm_match_mean_zero)
        rdm_corr_vec = rdm_corr_mean_zero / np.sqrt(np.sum(rdm_corr_mean_zero**2))
        rdm_match_vec = rdm_match_mean_zero / np.sqrt(np.sum(rdm_match_mean_zero**2))

        x = np.vstack([rdm_corr_vec, rdm_match_vec, np.ones(len(rdm_corr_vec))])

        kendall_single_bootstrap = np.zeros((self.n_ROIs, 2), dtype=np.float32)
        w_cmm_single_bootstrap = np.zeros((self.n_ROIs, 3), dtype=np.float32)

        for roi in range(self.n_ROIs):
            # for i in range(n_bootstrap):

            # print("compute w_cmm, roi: {}, bootstrap: {}/{}"
            #       .format(self.ROIs[roi], str(i_bootstrap), str(n_bootstrap)))

            rdm_fmri_roi = rdm_fmri_all[sbjID, roi]

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_vec = rdm_fmri_above - np.mean(rdm_fmri_above)

            # normalize by dividing max value
            # rdm_fmri_vec = rdm_fmri_vec/np.max(rdm_fmri_vec)
            rdm_fmri_vec = rdm_fmri_vec / np.sqrt(np.sum(rdm_fmri_vec**2))
            # rdm_fmri_vec = rdm_fmri_vec/sem(rdm_fmri_vec)

            # start fitting
            # w = solve_pg(x.T, rdm_fmri_vec)
            # w_cmm_single_bootstrap[roi] = w

            # start fitting using Ridge regression
            # forces the coefficients to be positive, use solver="lbfgs"
            # clf = Ridge(alpha=1,
            #             solver="lbfgs",
            #             positive=True,
            #             max_iter=1000)

            # start fitting using OLS
            clf = LinearRegression(positive=True)

            # start fitting using Lasso regression
            # clf = Lasso(alpha=1)
            clf.fit(x.T, rdm_fmri_vec)
            w = clf.coef_
            w_cmm_single_bootstrap[roi] = w

            y_pred = np.matmul(w, x)

            kendall_single_bootstrap[roi] = kendalltau(rdm_fmri_vec, y_pred)

        return w_cmm_single_bootstrap, kendall_single_bootstrap

    def compute_w_cmm_singleSbjID(self, rdm_corr, rdm_match, sbjID, n_bootstrap):
        """
        compute w_cmm for a single subject

        Parameters
        ----------
        rdm_bem : [n_bootstrap, 6, 6]
            rdm_bem obtained from self.compute_rdm_bem

        n_bootstrap : scalar
            number of bootsrap iteration.

        Returns
        -------
        w_bem_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 2] np.array
            estimated weights of squared bem for

            w_bootstrap[:, :, :, 0] -> w_bem
            w_bootstrap[:, :, :, 1] -> dc value

        kendall_bootstrap : [n_bootstrap, len(self.ROIs), 2] np.array
            kendall tau statistic.

            kendall_bootstrap[:, :, 0] -> kendalltau correlation
            kendall_bootstrap[:, :, 1] -> p_val

        """

        w_cmm = []
        w_cmm.append(
            Parallel(n_jobs=-1)(
                delayed(self.compute_w_cmm_singleSbjID_singleBootstrap)(
                    self.rdm_fmri_all, rdm_corr, rdm_match, sbjID, i
                )
                for i in range(n_bootstrap)
            )
        )
        # w_cmm.append(Parallel(n_jobs=-1)
        #               (delayed(cmm.compute_w_cmm_singleBootstrap)
        #               (rdm_fmri_all, x, n_bootstrap, i)
        #               for i in range(n_bootstrap)))

        # unpack
        # n_bootstrap = 1000
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 2), dtype=np.float32)
        w_cmm_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 3), dtype=np.float32)

        for i in range(n_bootstrap):
            # for i in range(n_bootstrap):
            temp = w_cmm[0][i][0]
            w_cmm_bootstrap[i] = temp

            temp = w_cmm[0][i][1]
            kendall_bootstrap[i] = temp

        return w_cmm_bootstrap, kendall_bootstrap

    def compute_w_cmm_all_sbjID(self, rdm_corr, rdm_match, n_bootstrap):
        """
        compute w_cmm for all sbjID

        Parameters
        ----------
        rdm_bem : [n_bootstrap, 6, 6]
            rdm_bem obtained from self.compute_rdm_bem

        n_bootstrap : scalar
            number of bootsrap iteration.

        Returns
        -------
        w_bem_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 2] np.array
            estimated weights of squared bem for

            w_bootstrap[:, :, :, 0] -> w_corr
            w_bootstrap[:, :, :, 1] -> w_match
            w_bootstrap[:, :, :, 2] -> dc value

        kendall_bootstrap : [n_bootstrap, len(self.ROIs), 2] np.array
            kendall tau statistic.

            kendall_bootstrap[:, :, 0] -> kendalltau correlation
            kendall_bootstrap[:, :, 1] -> p_val

        """

        w_cmm_all_sbjID = np.zeros(
            (self.n_sbjID, n_bootstrap, self.n_ROIs, 3), dtype=np.float32
        )
        kendall_all_sbjID = np.zeros(
            (self.n_sbjID, n_bootstrap, self.n_ROIs, 2), dtype=np.float32
        )

        for sbj in range(self.n_sbjID):

            print("compute w_cmm for sbjID:{}".format(sbj))
            w_cmm_sbj, kendall_sbj = self.compute_w_cmm_singleSbjID(
                rdm_corr, rdm_match, sbj, n_bootstrap
            )

            w_cmm_all_sbjID[sbj] = w_cmm_sbj
            kendall_all_sbjID[sbj] = kendall_sbj

        return w_cmm_all_sbjID, kendall_all_sbjID

    def compute_w_cmm_each_freq_singleBootstrap(
        self, rdm_fmri_all, rdm_corr_each_freq, rdm_match_each_freq, freq_id
    ):

        # average rdm across n_bootstrap
        rdm_corr_mean = np.mean(rdm_corr_each_freq, axis=0)
        rdm_mean_mean = np.mean(rdm_match_each_freq, axis=0)

        # get rdm_model for specific freq
        rdm_corr_freq = rdm_corr_mean[freq_id]
        rdm_match_freq = rdm_mean_mean[freq_id]

        # get above diagonal elements
        rdm_corr_above = rdm_corr_freq[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_freq[np.triu_indices(6, k=1)]

        #  mean-zeroing
        rdm_corr_above = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_match_above = rdm_match_above - np.mean(rdm_match_above)
        # normalize by dividing max value
        # rdm_corr_vec = rdm_corr_above/np.max(rdm_corr_above)
        # rdm_match_vec = rdm_match_above/np.max(rdm_match_above)
        rdm_corr_vec = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))
        rdm_match_vec = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

        # normalize again to ensure that both regressor rdm_corr_vec and rdm_match_vec
        # have the same maximum value.
        x = np.vstack(
            [
                rdm_corr_vec / np.max(rdm_corr_vec),
                rdm_match_vec / np.max(rdm_match_vec),
                np.ones(len(rdm_corr_vec)),
            ]
        )

        # n_bootstrap = 1000
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_bootstrap = np.zeros((self.n_ROIs, 2), dtype=np.float32)
        w_bootstrap = np.zeros((self.n_ROIs, 3), dtype=np.float32)

        # random sampling rdm_fmri
        id_sample = np.random.randint(self.n_sbjID, size=self.n_sbjID)

        for roi in range(len(self.ROIs)):

            # print("compute w_cmm at spatial freq {}, roi: {}, bootstrap: {}/{}"
            #       .format(str(self.f_batch[freq_id]), self.ROIs[roi],
            #               str(i+1), str(n_bootstrap)))

            rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]
            rdm_fmri_roi = np.mean(rdm_fmri_bootstrap, axis=0)

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_vec = rdm_fmri_above - np.mean(rdm_fmri_above)

            # normalize by dividing max valuce
            # rdm_fmri_vec = rdm_fmri_vec/np.max(rdm_fmri_vec)
            rdm_fmri_vec = rdm_fmri_vec / np.sqrt(np.sum(rdm_fmri_vec**2))
            # rdm_fmri_vec = rdm_fmri_vec/sem(rdm_fmri_vec)

            # start fitting
            w, resid = nnls(x.T, rdm_fmri_vec)
            w_bootstrap[roi] = w

            # normalize w such that w_corr + w_match = 1
            # w_corr = w[0]
            # w_match = w[1]
            # w_corr_norm = w_corr/(w_corr + w_match + 1e-6)
            # w_match_norm = w_match/(w_corr + w_match + 1e-6)

            # w_bootstrap[i, sbj, roi, 0] = w_corr_norm
            # w_bootstrap[i, sbj, roi, 1] = w_match_norm
            # w_bootstrap[i, sbj, roi, 2] = w[2]

            # compute r2
            # y_true_sbj[sbj] = rdm_fmri_vec
            y_pred = np.matmul(w, x)

            # r2_bootstrap[i, sbj, roi] = r2_score(rdm_fmri_vec, y_pred)
            # kendall_bootstrap[roi] = kendalltau(np.mean(y_true_sbj, axis=0),
            #                                     np.mean(y_pred_sbj, axis=0))
            kendall_bootstrap[roi] = kendalltau(rdm_fmri_vec, y_pred)

        return w_bootstrap, kendall_bootstrap

    def compute_w_cmm_each_freq_bootstrap(
        self,
        rdm_fmri_all,
        rdm_corr_each_freq,
        rdm_match_each_freq,
        n_bootstrap,
        freq_id,
    ):

        # n_bootstrap = 1000

        w_cmm = []
        w_cmm.append(
            Parallel(n_jobs=-1)(
                delayed(self.compute_w_cmm_each_freq_singleBootstrap)(
                    rdm_fmri_all, rdm_corr_each_freq, rdm_match_each_freq, freq_id
                )
                for i in range(n_bootstrap)
            )
        )

        # unpack
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 2), dtype=np.float32)
        w_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 3), dtype=np.float32)

        for i in range(n_bootstrap):
            temp = w_cmm[0][i][0]
            w_bootstrap[i] = temp

            temp = w_cmm[0][i][1]
            kendall_bootstrap[i] = temp

        return w_bootstrap, kendall_bootstrap

    def compute_w_cmm_corrLowFreq_matchHighFreq_singleBootstrap(self, rdm_fmri_all, x):

        kendall_singleBootstrap = np.zeros((self.n_ROIs, 2), dtype=np.float32)
        w_cmm_singleBootstrap = np.zeros((self.n_ROIs, 3), dtype=np.float32)

        # random sampling rdm_fmri_all [len(sbjID_all), n_ROIs, 6, 6]
        id_sample = np.random.randint(self.n_sbjID, size=self.n_sbjID)
        # id_sample = np.random.randint(22, size=22)

        for roi in range(self.n_ROIs):
            # print("compute w_cmm, roi: {}, bootstrap: {}/{}"
            #       .format(self.ROIs[roi], str(i_bootstrap), str(n_bootstrap)))

            rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]
            rdm_fmri_roi = np.mean(rdm_fmri_bootstrap, axis=0)

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_vec = rdm_fmri_above - np.mean(rdm_fmri_above)

            # normalize by dividing max value
            rdm_fmri_vec = rdm_fmri_vec / np.max(rdm_fmri_vec)
            # rdm_fmri_vec = rdm_fmri_vec/sem(rdm_fmri_vec)

            # start fitting
            w, resid = nnls(x.T, rdm_fmri_vec)
            w_cmm_singleBootstrap[roi] = w

            # normalize w such that w_corr + w_match = 1
            # w_corr = w[0]
            # w_match = w[1]
            # w_corr_norm = w_corr/(w_corr + w_match + 1e-6)
            # w_match_norm = w_match/(w_corr + w_match + 1e-6)

            # w_bootstrap[i, sbj, roi, 0] = w_corr_norm
            # w_bootstrap[i, sbj, roi, 1] = w_match_norm
            # w_bootstrap[i, sbj, roi, 2] = w[2]

            # compute r2
            # y_true_sbj[sbj] = rdm_corr_vec
            # y_true_sbj[sbj] = rdm_fmri_vec
            y_pred = np.matmul(w, x)

            # r2_bootstrap[i, sbj, roi] = r2_score(rdm_fmri_vec, y_pred)
            # kendall_singleBootstrap[roi] = kendalltau(np.mean(y_true_sbj, axis=0),
            #                                           np.mean(y_pred_sbj, axis=0))
            kendall_singleBootstrap[roi] = kendalltau(rdm_fmri_vec, y_pred)

        return w_cmm_singleBootstrap, kendall_singleBootstrap

    def compute_w_cmm_corrLowFreq_matchHighFreq(
        self, rdm_fmri_all, rdm_corr_each_freq, rdm_match_each_freq, n_bootstrap
    ):

        # average rdm_bem across bootstrap
        rdm_corr_mean = np.mean(rdm_corr_each_freq, axis=0)
        rdm_match_mean = np.mean(rdm_match_each_freq, axis=0)

        rdm_corrLowFreq = rdm_corr_mean[1]
        rdm_matchHighFreq = rdm_match_mean[3]

        # get above diagonal
        rdm_corrLowFreq_above = rdm_corrLowFreq[np.triu_indices(6, k=1)]
        rdm_matchHighFreq_above = rdm_matchHighFreq[np.triu_indices(6, k=1)]

        ## mean-zeroing and max-normalize
        rdm_corrLowFreq_mean_zero = rdm_corrLowFreq_above - np.mean(
            rdm_corrLowFreq_above
        )
        rdm_corrLowFreq_norm = rdm_corrLowFreq_mean_zero / np.max(
            rdm_corrLowFreq_mean_zero
        )

        rdm_matchHighFreq_mean_zero = rdm_matchHighFreq_above - np.mean(
            rdm_matchHighFreq_above
        )
        rdm_matchHighFreq_norm = rdm_matchHighFreq_mean_zero / np.max(
            rdm_matchHighFreq_mean_zero
        )

        # calculate pearson coef, 0.283 (correlation), 0.702 (euclidean)
        # rdm_r = np.corrcoef(rdm_corrLowFreq_norm, rdm_matchHighFreq_norm)[0, 1]

        x = np.vstack(
            [
                rdm_corrLowFreq_norm,
                rdm_matchHighFreq_norm,
                np.ones(len(rdm_corrLowFreq_norm)),
            ]
        )

        w_cmm = []
        w_cmm.append(
            Parallel(n_jobs=-1)(
                delayed(self.compute_w_cmm_corrLowFreq_matchHighFreq_singleBootstrap)(
                    rdm_fmri_all, x
                )
                for i in range(n_bootstrap)
            )
        )
        # w_cmm.append(Parallel(n_jobs=-1)
        #               (delayed(cmm.compute_w_cmm_singleBootstrap)
        #               (rdm_fmri_all, x, n_bootstrap, i)
        #               for i in range(n_bootstrap)))

        # unpack
        # n_bootstrap = 1000
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 2), dtype=np.float32)
        w_cmm_bootstrap = np.zeros((n_bootstrap, self.n_ROIs, 3), dtype=np.float32)
        # kendall_bootstrap = np.zeros((n_bootstrap, 8, 2),
        #                              dtype=np.float32)
        # w_cmm_bootstrap = np.zeros((n_bootstrap, 22, 8, 3),
        #                            dtype=np.float32)

        for i in range(n_bootstrap):
            # for i in range(n_bootstrap):
            temp = w_cmm[0][i][0]
            w_cmm_bootstrap[i] = temp

            temp = w_cmm[0][i][1]
            kendall_bootstrap[i] = temp

        return w_cmm_bootstrap, kendall_bootstrap

    def compute_w_cmm_group(self, rdm_corr, rdm_match):

        rdm_fmri_all = self.rdm_fmri_all
        rdm_fmri_vec_all = _create_rdm_fmri_vec(rdm_fmri_all)

        ## compute w_cmm
        w_cmm_group, loss_group = _fit_leave_one_out(
            rdm_fmri_vec_all, rdm_corr, rdm_match
        )

        return w_cmm_group, loss_group

    def compute_w_cmm_best(self, noise_dispCol_sigma_list):
        """

        compute the best w_cmm that has the least loss among noise_dispCol_sigma_list

        output:

        - w_cmm_best = [n_bootstrap, n_ROIs, n_sbjID, 3] np.float32

        - loss_min = [n_bootstrap, n_ROIs, n_sbjID] np.float32

        - id_best = [n_ROIs] dtype=np.int8

        """

        rdm_fmri_all = self.rdm_fmri_all
        n_sbjID, n_ROIs, _, _ = np.shape(rdm_fmri_all)

        rdm_fmri_vec_all = _create_rdm_fmri_vec(rdm_fmri_all)

        ## compute weight_bem, [n_sbjID, n_ROIs, 2]
        n_bootstrap = 1000
        w_cmm_group = np.zeros(
            (len(noise_dispCol_sigma_list), n_bootstrap, n_ROIs, n_sbjID, 3),
            dtype=np.float32,
        )
        loss_group = np.zeros(
            (len(noise_dispCol_sigma_list), n_bootstrap, n_ROIs, n_sbjID),
            dtype=np.float32,
        )
        for d in range(len(noise_dispCol_sigma_list)):

            # load rdm_bem and rdm_bem_squared
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_%.2f.npy"
                % (noise_dispCol_sigma_list[d])
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_%.2f.npy"
                % (noise_dispCol_sigma_list[d])
            )

            # compute w_cmm
            w_cmm_group[d], loss_group[d] = _fit_leave_one_out(
                rdm_fmri_vec_all, rdm_corr, rdm_match
            )

        ## get the best w_cmm

        # average loss_group across sbjID and bootstrap
        loss_avg = np.mean(np.mean(loss_group, axis=3), axis=1)

        w_cmm_best = np.zeros((n_bootstrap, n_ROIs, n_sbjID, 3), dtype=np.float32)
        loss_min = np.zeros((n_bootstrap, n_ROIs, n_sbjID), dtype=np.float32)
        id_best = np.zeros(n_ROIs, dtype=np.int8)

        for roi in range(n_ROIs):

            temp = loss_avg[:, roi]
            idx = np.where(temp == np.min(temp))
            id_best[roi] = idx[0][0]

            w_cmm_best[:, roi] = w_cmm_group[idx, :, roi]
            loss_min[:, roi] = loss_group[idx, :, roi]

        return w_cmm_best, loss_min, id_best


class Signal2Noise(General):

    def __init__(self, sawtooth_noise_std_list):

        super().__init__()

        self.sawtooth_noise_std_list = sawtooth_noise_std_list

    def load_voxResp_cmm(self, n_bootstrap, sawtooth_noise_std):

        voxResp_ards = np.load(
            "../../../Data/CMM/dispCol_noRF_voxResp_ards_bootstrap{}_noise{}.npy".format(
                str(n_bootstrap), str(sawtooth_noise_std)
            )
        )
        voxResp_hmrds = np.load(
            "../../../Data/CMM/dispCol_noRF_voxResp_hmrds_bootstrap{}_noise{}.npy".format(
                str(n_bootstrap), str(sawtooth_noise_std)
            )
        )
        voxResp_crds = np.load(
            "../../../Data/CMM/dispCol_noRF_voxResp_crds_bootstrap{}_noise{}.npy".format(
                str(n_bootstrap), str(sawtooth_noise_std)
            )
        )

        return voxResp_ards, voxResp_hmrds, voxResp_crds

    def _load_voxResp_fmri_sbj(self, nVox_to_analyze, sbj):
        """
        a helper function for self.load_voxResp_fmri.
        this function load fmri data of a participant.

        Parameters
        ----------
        nVox_to_analyze : scalar
            the number of voxel used for the analysis.
            ex: 250

        sbj : scalar
            the subject id.

        Returns
        -------
        voxResp_fmri_sbj : [n_ROIs,
                            nRuns, self.n_conds, nVox_to_analyze] np.array
            voxel responses of a participant.

        """

        # load P_data
        sbjID = self.sbjID_all[sbj]
        nRuns = self.nRuns_all[sbj]
        P_data = self.load_P_data(sbjID, nRuns)

        # process P_data
        P_data = self.label_P_data(P_data)

        # normalize P_data
        P_data = self.normalize_P_data(P_data)  # fixation is excluded here

        voxResp_fmri_sbj = np.zeros(
            (self.n_ROIs, nRuns, self.n_conds, nVox_to_analyze), dtype=np.float32
        )

        for roi in range(self.n_ROIs):
            # filter dataset according to sbjID, roi, and nVox and exclude fixation
            P_roi = P_data.loc[
                (P_data.roi == roi + 1)
                & (P_data.vox.isin(range(1, nVox_to_analyze + 1)))
                & (P_data.cond != 1)
            ]

            # average voxVal across rep
            P_roi_cond = (
                P_roi.groupby(["cond", "vox", "run"]).voxVal.agg(["mean"]).reset_index()
            )
            P_roi_cond = P_roi_cond.rename(columns={"mean": "avg"})

            # collect data for each condition and run
            df = P_roi_cond.pivot_table(
                index="run", columns=["cond", "vox"], values="avg"
            )

            # coordinate: [nRuns, nConds, nVox]
            voxResp_fmri_sbj[roi] = np.reshape(
                np.array(df), (nRuns, self.n_conds, nVox_to_analyze)
            )

        return voxResp_fmri_sbj

    def load_voxResp_fmri(self, nVox_to_analyze):
        """
        load voxel responses from fmri measuremnets

        Parameters
        ----------
        nVox_to_analyze : scalar
            the number of voxel used for the analysis.
            ex: 250

        Returns
        -------
        voxResp_fmri_all : [n_sbj, n_ROIs,
                            nRuns, self.n_conds, nVox_to_analyze] dict
            a dictionary containing voxResp_fmri for all participants.

        """

        voxResp_fmri = []
        voxResp_fmri.append(
            Parallel(n_jobs=self.n_sbjID)(
                delayed(self._load_voxResp_fmri_sbj)(nVox_to_analyze, sbj)
                for sbj in range(self.n_sbjID)
            )
        )

        # unpack
        voxResp_fmri_all = {}
        for sbj in range(self.n_sbjID):
            temp = voxResp_fmri[0][sbj]

            voxResp_fmri_all[sbj] = temp

        return voxResp_fmri_all

    def load_w_cmm_bootstrap(self, sawtooth_noise_std):

        w_cmm_bootstrap = np.load(
            "../../../Data/CMM/w_cmm_bootstrap_noise{}.npy".format(
                str(sawtooth_noise_std)
            )
        )

        return w_cmm_bootstrap

    def normalize_voxResp_cmm(self, voxResp_ards, voxResp_hmrds, voxResp_crds):
        """
        Normalize the simulated voxel responses voxResp_cmm in each run
        such that the combined distribution of
        voxResp_ards_crossed, voxResp_ards_uncrossed,
        voxResp_hmrds_crossed, voxResp_hmrds_uncrossed,
        voxResp_crds_crossed, and voxResp_crds_uncrossed are in interval [0, 1]:

            voxResp_mixed = [voxResp_ards_crossed, voxResp_ards_uncrossed,
                             voxResp_hmrds_crossed, voxResp_hmrds_uncrossed,
                             voxResp_crds_crossed, and voxResp_crds_uncrossed]
            num = voxResp_cmm - np.min(voxResp_mixed)/
            den = np.max(voxResp_mixed) - np.min(voxResp_mixed)
            voxResp_cmm_norm = dum/den

        Parameters
        ----------
        voxResp_ards : [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
                        np.array
            simulated voxel responses based on cmm for ards

        voxResp_hmrds : [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
                        np.array
            simulated voxel responses based on cmm for hmrds

        voxResp_crds : [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
                        np.array
            simulated voxel responses based on cmm for crds

        Returns
        -------
        voxResp_corr_norm : [n_bootstrap, nConds, nVox_to_analyze] np.array
            normalized voxresp_cmm for correlation computation.

            the nConds axis is for stimuli in the following order:
            [:, 0] -> ards_corr_crossed
            [:, 1] -> ards_corr_uncrossed
            [:, 2] -> hmrds_corr_crossed
            [:, 3] -> hmrds_corr_uncrossed
            [:, 4] -> crds_corr_crossed
            [:, 5] -> crds_corr_uncrossed


        voxResp_match_norm : [n_bootstrap, nConds, nVox_to_analyze] np.array
            normalized voxresp_cmm for matching computation

            the nConds axis is for stimuli in the following order:
            [:, 0] -> ards_match_crossed
            [:, 1] -> ards_match_uncrossed
            [:, 2] -> hmrds_match_crossed
            [:, 3] -> hmrds_match_uncrossed
            [:, 4] -> crds_match_crossed
            [:, 5] -> crds_match_uncrossed

        """

        n_bootstrap = voxResp_ards.shape[0]
        nVox_to_analyze = voxResp_ards.shape[1]

        # voxeResp_ards [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
        # get correlation and match voxel responses
        voxResp_ards_corr_crossed = np.sum(
            voxResp_ards[:, :, 0, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_ards_corr_uncrossed = np.sum(
            voxResp_ards[:, :, 0, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_ards_match_crossed = np.sum(
            voxResp_ards[:, :, 1, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_ards_match_uncrossed = np.sum(
            voxResp_ards[:, :, 1, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]

        voxResp_hmrds_corr_crossed = np.sum(
            voxResp_hmrds[:, :, 0, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_hmrds_corr_uncrossed = np.sum(
            voxResp_hmrds[:, :, 0, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_hmrds_match_crossed = np.sum(
            voxResp_hmrds[:, :, 1, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_hmrds_match_uncrossed = np.sum(
            voxResp_hmrds[:, :, 1, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]

        voxResp_crds_corr_crossed = np.sum(
            voxResp_crds[:, :, 0, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_crds_corr_uncrossed = np.sum(
            voxResp_crds[:, :, 0, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_crds_match_crossed = np.sum(
            voxResp_crds[:, :, 1, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_crds_match_uncrossed = np.sum(
            voxResp_crds[:, :, 1, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]

        # compute response average across n_trial
        ards_corr_crossed_avg = np.mean(
            voxResp_ards_corr_crossed, axis=2
        )  # [n_bootstrap, nVox]
        ards_corr_uncrossed_avg = np.mean(
            voxResp_ards_corr_uncrossed, axis=2
        )  # [n_bootstrap, nVox]
        ards_match_crossed_avg = np.mean(
            voxResp_ards_match_crossed, axis=2
        )  # [n_bootstrap, nVox]
        ards_match_uncrossed_avg = np.mean(
            voxResp_ards_match_uncrossed, axis=2
        )  # [n_bootstrap, nVox]

        hmrds_corr_crossed_avg = np.mean(
            voxResp_hmrds_corr_crossed, axis=2
        )  # [n_bootstrap, nVox]
        hmrds_corr_uncrossed_avg = np.mean(
            voxResp_hmrds_corr_uncrossed, axis=2
        )  # [n_bootstrap, nVox]
        hmrds_match_crossed_avg = np.mean(
            voxResp_hmrds_match_crossed, axis=2
        )  # [n_bootstrap, nVox]
        hmrds_match_uncrossed_avg = np.mean(
            voxResp_hmrds_match_uncrossed, axis=2
        )  # [n_bootstrap, nVox]

        crds_corr_crossed_avg = np.mean(
            voxResp_crds_corr_crossed, axis=2
        )  # [n_bootstrap, nVox]
        crds_corr_uncrossed_avg = np.mean(
            voxResp_crds_corr_uncrossed, axis=2
        )  # [n_bootstrap, nVox]
        crds_match_crossed_avg = np.mean(
            voxResp_crds_match_crossed, axis=2
        )  # [n_bootstrap, nVox]
        crds_match_uncrossed_avg = np.mean(
            voxResp_crds_match_uncrossed, axis=2
        )  # [n_bootstrap, nVox]

        # normalize the distribution of voxel responses so that all voxResp above
        # lies in interval [0, 1]
        # stimuli order:
        # [ards_corr_crossed, ards_corr_uncrossed,
        #  hmrds_corr_crossed, hmrds_corr_uncrossed,
        #  crds_corr_crossed, crds_corr_uncrossed]
        voxResp_corr_norm = np.zeros(
            (n_bootstrap, self.n_conds, nVox_to_analyze), dtype=np.float32
        )

        # stimuli order:
        # [ards_match_crossed, ards_match_uncrossed,
        #  hmrds_match_crossed, hmrds_match_uncrossed,
        #  crds_match_crossed, crds_match_uncrossed]
        voxResp_match_norm = np.zeros(
            (n_bootstrap, self.n_conds, nVox_to_analyze), dtype=np.float32
        )

        for i in range(n_bootstrap):
            # get the distribution of correlation computation
            corr_dist = np.append(
                [ards_corr_crossed_avg[i]],
                [
                    ards_corr_uncrossed_avg[i],
                    hmrds_corr_crossed_avg[i],
                    hmrds_corr_uncrossed_avg[i],
                    crds_corr_crossed_avg[i],
                    crds_corr_uncrossed_avg[i],
                ],
            )

            num = ards_corr_crossed_avg[i] - np.min(corr_dist)
            den = np.max(corr_dist) - np.min(corr_dist)
            voxResp_corr_norm[i, 0] = num / den

            num = ards_corr_uncrossed_avg[i] - np.min(corr_dist)
            voxResp_corr_norm[i, 1] = num / den

            num = hmrds_corr_crossed_avg[i] - np.min(corr_dist)
            voxResp_corr_norm[i, 2] = num / den

            num = hmrds_corr_uncrossed_avg[i] - np.min(corr_dist)
            voxResp_corr_norm[i, 3] = num / den

            num = crds_corr_crossed_avg[i] - np.min(corr_dist)
            voxResp_corr_norm[i, 4] = num / den

            num = crds_corr_uncrossed_avg[i] - np.min(corr_dist)
            voxResp_corr_norm[i, 5] = num / den

            match_dist = np.append(
                [ards_match_crossed_avg[i]],
                [
                    ards_match_uncrossed_avg[i],
                    hmrds_match_crossed_avg[i],
                    hmrds_match_uncrossed_avg[i],
                    crds_match_crossed_avg[i],
                    crds_match_uncrossed_avg[i],
                ],
            )

            num = ards_match_crossed_avg[i] - np.min(match_dist)
            den = np.max(match_dist) - np.min(match_dist)
            voxResp_match_norm[i, 0] = num / den

            num = ards_match_uncrossed_avg[i] - np.min(match_dist)
            voxResp_match_norm[i, 1] = num / den

            num = hmrds_match_crossed_avg[i] - np.min(match_dist)
            den = np.max(match_dist) - np.min(match_dist)
            voxResp_match_norm[i, 2] = num / den

            num = hmrds_match_uncrossed_avg[i] - np.min(match_dist)
            voxResp_match_norm[i, 3] = num / den

            num = crds_match_crossed_avg[i] - np.min(match_dist)
            den = np.max(match_dist) - np.min(match_dist)
            voxResp_match_norm[i, 4] = num / den

            num = crds_match_uncrossed_avg[i] - np.min(match_dist)
            voxResp_match_norm[i, 5] = num / den

        return voxResp_corr_norm, voxResp_match_norm

    def _normalize_voxResp_fmri_sbj(self, nVox_to_analyze, sbj):
        """
        normalize voxel responses from fMRI data
        such that the combined distribution of
        voxResp_ards_crossed, voxResp_ards_uncrossed,
        voxResp_hmrds_crossed, voxResp_hmrds_uncrossed,
        voxResp_crds_crossed, and voxResp_crds_uncrossed are in interval [0, 1]:

            voxResp_mixed = [voxResp_ards_crossed, voxResp_ards_uncrossed,
                             voxResp_hmrds_crossed, voxResp_hmrds_uncrossed,
                             voxResp_crds_crossed, and voxResp_crds_uncrossed]
            num = voxResp_cmm - np.min(voxResp_mixed)/
            den = np.max(voxResp_mixed) - np.min(voxResp_mixed)
            voxResp_cmm_norm = dum/den

        Parameters
        ----------
        nVox_to_analyze : scalar
            the number of voxel used for the analysis.
            ex: 250

        sbj : scalar
            the subject id.

        Returns
        -------
        voxResp_norm : [n_ROIs, nRuns, n_conds, nVox_to_analyze]
                        np.array
            the normalized voxResp_fmri for a single participant.

        """

        # load P_data
        sbjID = self.sbjID_all[sbj]
        nRuns = self.nRuns_all[sbj]
        P_data = self.load_P_data(sbjID, nRuns)

        # process P_data
        P_data = self.label_P_data(P_data)

        # normalize P_data
        P_data = self.normalize_P_data(P_data)  # fixation is excluded here

        voxResp_norm = np.zeros(
            (self.n_ROIs, nRuns, self.n_conds, nVox_to_analyze), dtype=np.float32
        )

        for roi in range(self.n_ROIs):
            # filter dataset according to sbjID, roi, and nVox and exclude fixation
            P_roi = P_data.loc[
                (P_data.roi == roi + 1)
                & (P_data.vox.isin(range(1, nVox_to_analyze + 1)))
                & (P_data.cond != 1)
            ]

            # average voxVal across rep
            P_roi_cond = (
                P_roi.groupby(["cond", "vox", "run"]).voxVal.agg(["mean"]).reset_index()
            )
            P_roi_cond = P_roi_cond.rename(columns={"mean": "avg"})

            # collect data for each condition and run
            df = P_roi_cond.pivot_table(
                index="run", columns=["cond", "vox"], values="avg"
            )

            # coordinate: [nRuns, nConds, nVox]
            df2 = np.reshape(np.array(df), (nRuns, self.n_conds, nVox_to_analyze))

            # normalize voxel responses for each run
            temp = np.max(np.max(df2, axis=2), axis=1)
            resp_max = np.tile(
                temp[:, np.newaxis, np.newaxis], (1, self.n_conds, nVox_to_analyze)
            )
            temp = np.min(np.min(df2, axis=2), axis=1)
            resp_min = np.tile(
                temp[:, np.newaxis, np.newaxis], (1, self.n_conds, nVox_to_analyze)
            )

            num = df2 - resp_min
            den = resp_max - resp_min
            voxResp_norm[roi] = num / den

        return voxResp_norm

    def normalize_voxResp_fmri(self, nVox_to_analyze):
        """
        normalize voxel responses from fMRI data for all participants.
        such that the combined distribution of
        voxResp_ards_crossed, voxResp_ards_uncrossed,
        voxResp_hmrds_crossed, voxResp_hmrds_uncrossed,
        voxResp_crds_crossed, and voxResp_crds_uncrossed are in interval [0, 1]:

        voxResp_mixed = [voxResp_ards_crossed, voxResp_ards_uncrossed,
                         voxResp_hmrds_crossed, voxResp_hmrds_uncrossed,
                         voxResp_crds_crossed, and voxResp_crds_uncrossed]
        num = voxResp_cmm - np.min(voxResp_mixed)/
        den = np.max(voxResp_mixed) - np.min(voxResp_mixed)
        voxResp_cmm_norm = dum/den

        Parameters
        ----------
        nVox_to_analyze : scalar
            the number of voxels used for the analysis.
            ex: 250

        Returns
        -------
        voxResp_unpack : [n_sbjID, n_ROIs, nRuns, n_conds, nVox_to_analyze] dict
            a dictionary containing the normalized voxel responses for all
            participants.

        """

        voxResp_fmri_norm_all = []
        voxResp_fmri_norm_all.append(
            Parallel(n_jobs=self.n_sbjID)(
                delayed(self._normalize_voxResp_fmri_sbj)(nVox_to_analyze, sbj)
                for sbj in range(self.n_sbjID)
            )
        )

        # unpack
        voxResp_unpack = {}
        for sbj in range(self.n_sbjID):
            temp = voxResp_fmri_norm_all[0][sbj]
            voxResp_unpack[sbj] = temp

        return voxResp_unpack

    def _s2n_cmm(
        self,
        voxResp_ards,
        voxResp_hmrds,
        voxResp_crds,
        w_cmm_bootstrap,
        nVox_to_analyze,
    ):
        """
        compute signal to noise ratio for each n_bootstrap,

        Parameters
        ----------
        voxResp_ards : [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
                        np.array
            simulated voxel responses based on cmm for ards

        voxResp_hmrds : [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
                        np.array
            simulated voxel responses based on cmm for hmrds

        voxResp_crds : [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
                        np.array
            simulated voxel responses based on cmm for crds

        w_cmm_bootstrap : TYPE
            DESCRIPTION.

        nVox_to_analyze : TYPE
            DESCRIPTION.

        Returns
        -------

        s2n_cmm_avg : TYPE
            DESCRIPTION.

        """

        n_bootstrap = voxResp_ards.shape[0]
        n_ROIs = np.shape(w_cmm_bootstrap)[2]

        # voxeResp_ards [n_bootstrap, nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
        # get correlation and match voxel responses
        voxResp_ards_corr_crossed = np.sum(
            voxResp_ards[:, :, 0, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_ards_corr_uncrossed = np.sum(
            voxResp_ards[:, :, 0, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_ards_match_crossed = np.sum(
            voxResp_ards[:, :, 1, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_ards_match_uncrossed = np.sum(
            voxResp_ards[:, :, 1, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]

        voxResp_hmrds_corr_crossed = np.sum(
            voxResp_hmrds[:, :, 0, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_hmrds_corr_uncrossed = np.sum(
            voxResp_hmrds[:, :, 0, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_hmrds_match_crossed = np.sum(
            voxResp_hmrds[:, :, 1, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_hmrds_match_uncrossed = np.sum(
            voxResp_hmrds[:, :, 1, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]

        voxResp_crds_corr_crossed = np.sum(
            voxResp_crds[:, :, 0, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_crds_corr_uncrossed = np.sum(
            voxResp_crds[:, :, 0, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_crds_match_crossed = np.sum(
            voxResp_crds[:, :, 1, :, :, 0], axis=2
        )  # [n_bootstrap, nVox, n_trial]
        voxResp_crds_match_uncrossed = np.sum(
            voxResp_crds[:, :, 1, :, :, 1], axis=2
        )  # [n_bootstrap, nVox, n_trial]

        # compute response average across n_trial
        ards_corr_crossed_avg = np.mean(
            voxResp_ards_corr_crossed, axis=2
        )  # [n_bootstrap, nVox]
        ards_corr_uncrossed_avg = np.mean(
            voxResp_ards_corr_uncrossed, axis=2
        )  # [n_bootstrap, nVox]
        ards_match_crossed_avg = np.mean(
            voxResp_ards_match_crossed, axis=2
        )  # [n_bootstrap, nVox]
        ards_match_uncrossed_avg = np.mean(
            voxResp_ards_match_uncrossed, axis=2
        )  # [n_bootstrap, nVox]

        hmrds_corr_crossed_avg = np.mean(
            voxResp_hmrds_corr_crossed, axis=2
        )  # [n_bootstrap, nVox]
        hmrds_corr_uncrossed_avg = np.mean(
            voxResp_hmrds_corr_uncrossed, axis=2
        )  # [n_bootstrap, nVox]
        hmrds_match_crossed_avg = np.mean(
            voxResp_hmrds_match_crossed, axis=2
        )  # [n_bootstrap, nVox]
        hmrds_match_uncrossed_avg = np.mean(
            voxResp_hmrds_match_uncrossed, axis=2
        )  # [n_bootstrap, nVox]

        crds_corr_crossed_avg = np.mean(
            voxResp_crds_corr_crossed, axis=2
        )  # [n_bootstrap, nVox]
        crds_corr_uncrossed_avg = np.mean(
            voxResp_crds_corr_uncrossed, axis=2
        )  # [n_bootstrap, nVox]
        crds_match_crossed_avg = np.mean(
            voxResp_crds_match_crossed, axis=2
        )  # [n_bootstrap, nVox]
        crds_match_uncrossed_avg = np.mean(
            voxResp_crds_match_uncrossed, axis=2
        )  # [n_bootstrap, nVox]

        # compute signal to noise for correlation computation n for each voxel
        # and bootstrap
        y = np.append(
            [ards_corr_crossed_avg],
            [
                ards_corr_uncrossed_avg,
                hmrds_corr_crossed_avg,
                hmrds_corr_uncrossed_avg,
                crds_corr_crossed_avg,
                crds_corr_uncrossed_avg,
            ],
            axis=0,
        )

        # square to avoid negative values
        y2_corr = y[:, :, 0:nVox_to_analyze] ** 2

        # s2n_corr = np.zeros((n_bootstrap, 6, nVox_to_analyze),
        #                     dtype=np.float32)
        # for i in range(n_bootstrap):
        #     temp = y2_corr[:, i]

        #     # compute standard deviation across voxels and conditions
        #     y_var = np.var(temp)

        #     # average across voxels for each condition
        #     # signal to noise
        #     s2n_corr[i] = temp/y_var

        # # average across voxels and n_bootstrap
        # s2n_corr_avg = np.mean(np.mean(s2n_corr, axis=2), axis=0)

        # compute signal to noise for matching computation n for each voxel
        # and bootstrap
        y = np.append(
            [ards_match_crossed_avg],
            [
                ards_match_uncrossed_avg,
                hmrds_match_crossed_avg,
                hmrds_match_uncrossed_avg,
                crds_match_crossed_avg,
                crds_match_uncrossed_avg,
            ],
            axis=0,
        )

        # square to avoid negative values
        y2_match = y[:, :, 0:nVox_to_analyze] ** 2

        # s2n_match = np.zeros((n_bootstrap, 6, nVox_to_analyze), dtype=np.float32)
        # for i in range(n_bootstrap):
        #     temp = y2_match[:, i]

        #     # compute standard deviation across voxels and conditions
        #     y_var = np.var(temp)

        #     # average across voxels for each condition
        #     # signal to noise
        #     s2n_match[i] = temp/y_var

        # # average across voxels and n_bootstrap
        # s2n_match_avg = np.mean(np.mean(s2n_match, axis=2), axis=0)

        # compute s2n for cmm signal
        # average w_cmm_boostrap across bootstrap
        w_cmm = np.mean(w_cmm_bootstrap, axis=0)

        s2n_cmm = np.zeros(
            (self.n_sbjID, n_bootstrap, n_ROIs, self.n_conds), dtype=np.float32
        )

        for sbj in range(self.n_sbjID):
            for i in range(n_bootstrap):
                for roi in range(n_ROIs):

                    resp_corr = w_cmm[sbj, roi, 0] * y2_corr[:, i]
                    resp_match = w_cmm[sbj, roi, 1] * y2_match[:, i]
                    resp_cmm = resp_corr + resp_match  # [nConds, nVox]

                    # average across voxels
                    y_mean = np.mean(resp_cmm, axis=1)  # [nConds]

                    # compute variance across conditions and voxels
                    y_var = np.std(resp_cmm)

                    # compute signal to noise ratio
                    s2n_cmm[sbj, i, roi] = y_mean / y_var  # [nConds]

        # average s2n_cmm across bootstrap, and then sbjID
        s2n_cmm_avg = np.mean(np.mean(s2n_cmm, axis=1), axis=0)

        return s2n_cmm_avg

    def compute_s2n_cmm(self, sawtooth_noise_std, nVox_to_analyze, n_bootstrap):
        """
        this is a wrapper function for self._s2n_cmm.
        compute signal to noise ratio for cmm.


        Parameters
        ----------
        sawtooth_noise_std : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.

        Returns
        -------
        s2n_corr_avg : TYPE
            DESCRIPTION.
        s2n_match_avg : TYPE
            DESCRIPTION.
        s2n_cmm_avg : TYPE
            DESCRIPTION.

        """

        # load simulated voxel response associated with the sawtooth_noise_std
        # sawtooth_noise_std = 0.2
        voxResp_ards, voxResp_hmrds, voxResp_crds = self.load_voxResp_cmm(
            n_bootstrap, sawtooth_noise_std
        )

        # load w_cmm_bootstrap associated with the sawtooth_noise_std
        w_cmm_bootstrap = self.load_w_cmm_bootstrap(sawtooth_noise_std)

        s2n_cmm_avg = self._s2n_cmm(
            voxResp_ards, voxResp_hmrds, voxResp_crds, w_cmm_bootstrap, nVox_to_analyze
        )

        return s2n_cmm_avg

    def _s2n_fmri_sbj(self, t_stat_all_sbjID, nVox_to_analyze, sbj):
        """
        compute signal to noise ratio for a single participant

        Parameters
        ----------
        sbj : int
            the id of a participant, starts from 0.

        Returns
        -------
        s2n_all_roi : [nROIs, nRuns, nConds] np.array
            contains signal to noise ratio of a participant..
            the condition here includes fixation

        """

        sbjID = self.sbjID_all[sbj]
        nRuns = self.nRuns_all[sbj]
        nConds = self.n_conds  # exclude fixation

        # load vtc
        vtc = self.load_vtc(sbj)

        # load stimulus timing parameters
        vtc_stimID = sio.loadmat(
            "../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        )["paramIdFull"]

        # label vtc
        vtc_labeled = self.label_vtc(vtc, vtc_stimID)

        s2n_all_roi = np.zeros((self.n_ROIs, nRuns, nConds), dtype=np.float32)

        for roi in range(self.n_ROIs):

            # get t-stat
            t_roi = t_stat_all_sbjID[sbj][self.ROIs[roi]]

            # sort voxel in t_stat in descending order
            # a  = np.random.rand(10)
            # id_sort = a.argsort()[::-1]
            # a[id_sort]
            id_sort = t_roi.argsort()[::-1]

            # normalize voxel values inside ROI
            vtc_roi = vtc_labeled.loc[vtc_labeled.roi == roi]
            # average in each run
            avg = vtc_roi.groupby(["roi", "run", "vox"])["vtc_value"].transform("mean")
            # normalize
            temp = (vtc_roi.vtc_value - avg) / avg * 100
            vtc_roi = vtc_roi.assign(vtc_norm=temp)

            # check if nVox_to_analyze < nVox_max in this ROI
            nVox_max = len(t_roi)
            if nVox_to_analyze >= nVox_max:
                nVox_to_analyze = nVox_max

            for run in range(nRuns):

                print(
                    "compute signal to noise ratio, sbjID={}, ROI={}, nVox={}, run={}".format(
                        sbjID, self.ROIs[roi], nVox_to_analyze, run + 1
                    )
                )

                vtc_run = vtc_roi.loc[vtc_roi.run == run]

                # group by [roi, run, vox, cond], average across timepoints
                vtc_group = (
                    vtc_run.groupby(["vox", "stimID", "cond"])
                    .vtc_norm.agg(["mean"])
                    .reset_index()
                )
                vtc_group = vtc_group.rename(
                    columns={"mean": "vtc_avg", "var": "vtc_var"}
                )

                # transform vtc_group.vtc_avg into matrix [nConds, nVox]
                y_avg = np.array(
                    vtc_group.pivot_table(
                        index="stimID", columns="vox", values="vtc_avg"
                    ),
                    dtype=np.float32,
                )

                # sort y_avg in descendeding order based on t_test
                # [nConds, nVox], including fixation
                y_sort = y_avg[:, id_sort]

                # select nVox_to_analyze voxels
                # [nConds, nVox], including fixation
                y_sel = y_sort[:, 0:nVox_to_analyze]

                # get fixation responses
                # [nConds, nVox], excluding fixation
                y_fix = np.tile(y_sel[0], (nConds, 1))

                # compute the response difference between stimulus and fixation
                # [nConds, nVox], excluding fixation
                y_diff = y_sel[1:] - y_fix

                # compute standard deviation of response distribution for each
                # voxel across all conditions (including fixation).
                # Thus, the response distribution of each voxel is a collection
                # of responses of that voxel all timepoint in a run.
                vtc_group = (
                    vtc_run.groupby(["vox"]).vtc_norm.agg([np.std]).reset_index()
                )
                vtc_group = vtc_group.rename(columns={"std": "vtc_std"})
                y_std = np.array(vtc_group.vtc_std)
                temp = y_std[id_sort]
                y_std = np.tile(temp[0:nVox_to_analyze], (nConds, 1))

                # compute s2n with respect to fixation for each voxel
                s2n = y_diff / y_std  # [nConds, nVox], excluding fixation

                # average across these voxels
                # [nROIs, nRuns, nConds], exclude fixation
                s2n_all_roi[roi, run] = np.mean(s2n, axis=1)

        return s2n_all_roi

    def compute_s2n_fmri_all_sbj(self, t_stat_all_sbjID, nVox_to_analyze):
        """
        compute signal to noise ratio for all particpants

        Parameters
        ----------
        t_stat_all_sbjID : list
            a list of t-statistics of each voxels for each participant.
        nVox_to_analyze : np.int
            the number of voxels used for analysis.

        Returns
        -------
        s2n_all_sbj : [n_sbjID, nROIs, nConds] np.array
            signal-to-noise ratio for all participants.

        """

        s2n_list = []

        t_start = timer()
        s2n_list.append(
            Parallel(n_jobs=10)(
                delayed(self._s2n_fmri_sbj)(t_stat_all_sbjID, nVox_to_analyze, sbj)
                for sbj in range(self.n_sbjID)
            )
        )
        t_end = timer()
        print(t_end - t_start)

        # extract s2n_list
        nConds = self.n_conds  # exclude fixation
        s2n_all_sbj = np.zeros((self.n_sbjID, self.n_ROIs, nConds), dtype=np.float32)
        for sbj in range(self.n_sbjID):
            s2n = s2n_list[0][sbj]  # [nROIs, nRuns, nConds]

            # average across run
            s2n_avg = np.mean(s2n, axis=1)  # [nROIs, nConds]

            s2n_all_sbj[sbj] = s2n_avg  # [n_sbjID, nROIs, nConds]

        return s2n_all_sbj

    def compute_s2n_as_function_of_voxel(self, t_stat_all_sbjID, nVox_list):
        """
        compute signal to noise ratio as a function of voxels.

        Parameters
        ----------
        t_stat_all_sbjID : list
            a list of t-statistics of each voxels for each participant.
        nVox_list : np.arange(25, 825, 25)
            an array containing a list of voxels used for the analysis.

        Returns
        -------
        s2n_all_vox : [len(nVox_list), n_sbjID, nROIs, nConds] np.array
            DESCRIPTION.

        """

        s2n_all_vox = np.zeros(
            (len(nVox_list), self.n_sbjID, self.n_ROIs, self.n_conds), dtype=np.float32
        )
        for v in range(len(nVox_list)):

            nVox_to_analyze = nVox_list[v]
            s2n_vox = self.compute_s2n_fmri_all_sbj(t_stat_all_sbjID, nVox_to_analyze)

            # [len(nVox_list), n_sbjID, nROIs, nConds]
            s2n_all_vox[v] = s2n_vox

        return s2n_all_vox

    def compute_s2n_cmm_with_normalizedResp(
        self, voxResp_corr_norm, voxResp_match_norm, w_cmm_bootstrap, nVox_to_analyze
    ):
        """
        compute signal to noise ratio using the normalized response

        Parameters
        ----------
        voxResp_corr_norm : [n_bootstrap, nConds, nVox_to_analyze] np.array
            normalized voxresp_cmm for correlation computation.

            the nConds axis is for stimuli in the following order:
            [:, 0] -> ards_corr_crossed
            [:, 1] -> ards_corr_uncrossed
            [:, 2] -> hmrds_corr_crossed
            [:, 3] -> hmrds_corr_uncrossed
            [:, 4] -> crds_corr_crossed
            [:, 5] -> crds_corr_uncrossed


        voxResp_match_norm : [n_bootstrap, nConds, nVox_to_analyze] np.array
            normalized voxresp_cmm for matching computation

            the nConds axis is for stimuli in the following order:
            [:, 0] -> ards_match_crossed
            [:, 1] -> ards_match_uncrossed
            [:, 2] -> hmrds_match_crossed
            [:, 3] -> hmrds_match_uncrossed
            [:, 4] -> crds_match_crossed
            [:, 5] -> crds_match_uncrossed

        w_cmm_bootstrap : TYPE
            DESCRIPTION.

        nVox_to_analyze : TYPE
            DESCRIPTION.

        Returns
        -------
        s2n_cmm_avg : [n_sbjID, n_bootstrap, n_ROIs, nConds] np.array
            DESCRIPTION.

        """

        n_bootstrap = voxResp_corr_norm.shape[0]
        n_ROIs = np.shape(w_cmm_bootstrap)[2]

        # voxResp_corr_norm [n_bootstrap, nConds, nVox_to_analyze]
        # get correlation and match voxel responses
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_corr_crossed = voxResp_corr_norm[:, 0]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_corr_uncrossed = voxResp_corr_norm[:, 1]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_match_crossed = voxResp_match_norm[:, 0]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_match_uncrossed = voxResp_match_norm[:, 1]

        # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_corr_crossed = voxResp_corr_norm[:, 2]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_corr_uncrossed = voxResp_corr_norm[:, 3]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_match_crossed = voxResp_match_norm[:, 2]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_match_uncrossed = voxResp_match_norm[:, 3]

        # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_corr_crossed = voxResp_corr_norm[:, 4]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_corr_uncrossed = voxResp_corr_norm[:, 5]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_match_crossed = voxResp_match_norm[:, 4]
        # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_match_uncrossed = voxResp_match_norm[:, 5]

        # compute signal to noise for correlation computation n for each voxel
        # and bootstrap
        y_corr = np.append(
            [voxResp_ards_corr_crossed],
            [
                voxResp_ards_corr_uncrossed,
                voxResp_hmrds_corr_crossed,
                voxResp_hmrds_corr_uncrossed,
                voxResp_crds_corr_crossed,
                voxResp_crds_corr_uncrossed,
            ],
            axis=0,
        )

        # y2_corr = y[:, :, 0:nVox_to_analyze]**2 # square to avoid negative values

        # s2n_corr = np.zeros((n_bootstrap, self.n_conds),
        #                     dtype=np.float32)
        # for i in range(n_bootstrap):
        #     temp = y_corr[:, i]

        #     # average across voxels
        #     y_mean = np.mean(temp, axis=1)

        #     # compute standard deviation across voxels and conditions
        #     y_var = np.var(temp)

        #     # average across voxels for each condition
        #     # signal to noise
        #     s2n_corr[i] = y_mean/y_var

        # # average across n_bootstrap
        # s2n_corr_avg = np.mean(s2n_corr, axis=0)

        # compute signal to noise for matching computation n for each voxel
        # and bootstrap
        y_match = np.append(
            [voxResp_ards_match_crossed],
            [
                voxResp_ards_match_uncrossed,
                voxResp_hmrds_match_crossed,
                voxResp_hmrds_match_uncrossed,
                voxResp_crds_match_crossed,
                voxResp_crds_match_uncrossed,
            ],
            axis=0,
        )

        # y2_match = y[:, :, 0:nVox_to_analyze]**2 # square to avoid negative values

        # s2n_match = np.zeros((n_bootstrap, self.n_conds), dtype=np.float32)
        # for i in range(n_bootstrap):
        #     temp = y_match[:, i]

        #     # average across voxels
        #     y_mean = np.mean(temp, axis=1)

        #     # compute standard deviation across voxels and conditions
        #     y_var = np.var(temp)

        #     # average across voxels for each condition
        #     # signal to noise
        #     s2n_match[i] = y_mean/y_var

        # # average across n_bootstrap
        # s2n_match_avg = np.mean(s2n_match, axis=0)

        # compute s2n for cmm signal
        # average w_cmm_boostrap across bootstrap
        w_cmm = np.mean(w_cmm_bootstrap, axis=0)

        s2n_cmm = np.zeros(
            (self.n_sbjID, n_bootstrap, n_ROIs, self.n_conds), dtype=np.float32
        )

        for sbj in range(self.n_sbjID):
            for i in range(n_bootstrap):
                for roi in range(n_ROIs):

                    resp_corr = w_cmm[sbj, roi, 0] * y_corr[:, i]
                    resp_match = w_cmm[sbj, roi, 1] * y_match[:, i]
                    resp_cmm = resp_corr + resp_match

                    # average across voxels
                    y_mean = np.mean(resp_cmm, axis=1)

                    # compute variance across conditions and voxels
                    y_var = np.var(resp_cmm)

                    # compute signal to noise ratio
                    s2n_cmm[sbj, i, roi] = y_mean / y_var

        # average s2n_cmm across bootstrap, and then sbjID
        s2n_cmm_avg = np.mean(np.mean(s2n_cmm, axis=1), axis=0)

        return s2n_cmm_avg

    def compute_s2n_fmri_with_normalizedResp(self, voxResp_fmri_norm_all):
        """


        Parameters
        ----------
        voxResp_fmri_norm_all : [n_sbjID, n_ROIs, nRuns, n_conds, nVox_to_analyze] dict
            a dictionary containing the normalized voxel responses for all
            participants.

        Returns
        -------
        s2n_fmri_avg : [n_sbjID, n_ROIs, n_conds] np.array
            signal to noise ratio for fmri responses

        """

        s2n_all = np.zeros((self.n_sbjID, self.n_ROIs, self.n_conds), dtype=np.float32)

        for sbj in range(self.n_sbjID):

            nRuns = self.nRuns_all[sbj]

            for roi in range(self.n_ROIs):

                s2n_run = np.zeros((nRuns, self.n_conds), dtype=np.float32)

                for run in range(nRuns):

                    # [nConds, nVox_to_analyze]
                    voxResp = voxResp_fmri_norm_all[sbj][roi, run]

                    # average across voxels
                    y_mean = np.mean(voxResp, axis=1)

                    # compute variance across conditions and voxels
                    y_var = np.var(voxResp)

                    # compute signal to noise ratio
                    s2n_run[run] = y_mean / y_var

                s2n_all[sbj, roi] = np.mean(s2n_run, axis=0)

        # average across sbjID
        s2n_fmri_avg = np.mean(s2n_all, axis=0)

        return s2n_fmri_avg


class Plot_CMM(PlotGeneral):

    def __init__(self, mtd):

        super().__init__()

        self.dpi = 600

        self.mtd = mtd
        self.rdm_fmri_all = np.load(
            "../../../Data/CMM/rdm_fmri_all_{}.npy".format(mtd)
        )  # [sbjID, roi, 6, 6]

        # plt.style.use("seaborn-colorblind")
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = "Ubuntu"
        # plt.rcParams["font.monospace"] = "Ubuntu Mono"
        # plt.rcParams["axes.labelweight"] = "bold"

        # load the lower and upper bound of noise ceiling.
        # the following is the way to compute noise ceiling:
        # rsa = RSA()
        # n_bootstrap = 1000
        # kendalltau_low, kendalltau_up = rsa.compute_noiseCeiling(rdm_fmri_all,
        #                                                          n_bootstrap)
        # kendalltau_low, [n_ROIs, n_bootstrap, n_sbjID]
        self.kendalltau_low = np.load(
            "../../../Data/CMM/noise_ceiling_kendalltau_low_{}.npy".format(mtd)
        )
        # kendalltau_up, [n_ROIs, n_bootsrap]
        self.kendalltau_up = np.load(
            "../../../Data/CMM/noise_ceiling_kendalltau_up_{}.npy".format(mtd)
        )

    def plotHeat_rdm_cmm_fit(
        self, noise_dispCol_sigma_list, w_cmm_best, id_best, save_flag
    ):
        """for visualization, the RDM are only max-normalize, without
        mean-centeringfor visualization, the RDM are only max-normalize

        Parameters
        ----------

        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_best[:, 0] -> w_corr
            w_cmm_best[:, 1] -> w_match
            w_cmm_best[:, 2] -> dc value

        mtd: [string]
            the computational method for computing the distance between two voxel
            response patterns to generate RDM.
            for example: "sqeuclidean"

        sawtooth_noise_std: [scalar]
            the standard deviation of the noise to jitter the sawtooth distribution
            in the disparity column map.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average rdm_fmri_all across sbjID
        # rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)
        rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)

        # average w_cmm_bootstrap across sbjID and bootstrap
        w_cmm = np.mean(np.mean(w_cmm_best, axis=2), axis=0)  # [nROIs, 3]

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        den = w_cmm[:, 0] + w_cmm[:, 1] + tol
        w_corr = w_cmm[:, 0] / den
        w_match = w_cmm[:, 1] / den
        # w_corr = w_cmm[:, 0]
        # w_match = w_cmm[:, 1]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 20)
        n_row = 8
        n_col = 3

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(0.5, 1.03, "RDM_fMRI and RDM_model. {}".format(self.mtd), ha="center")

        fig.text(
            0.0,
            0.925,
            "V1",
            va="center",
        )
        fig.text(0.0, 0.81, "V2", va="center")
        fig.text(0.0, 0.69, "V3", va="center")
        fig.text(-0.0, 0.575, "V3A", va="center")
        fig.text(-0.0, 0.45, "V3B", va="center")
        fig.text(-0.0, 0.325, "hV4", va="center")
        fig.text(-0.0, 0.2, "V7", va="center")
        fig.text(-0.0, 0.08, "MT", va="center")

        fig.text(0.175, 0.0, "rdm_fMRI", ha="center")
        fig.text(0.5, 0.0, "rdm_fit", ha="center")
        fig.text(0.85, 0.0, "rdm_residual", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        v_min = 0.0
        v_max = 1.0
        cmap = "jet"
        for roi in range(self.n_ROIs):

            d_roi = id_best[roi]
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            # average rdm_corr and rdm_match across bootstrap
            rdm_corr_bootstrap = np.mean(rdm_corr, axis=0)
            rdm_match_bootstrap = np.mean(rdm_match, axis=0)

            # get above diagonal element
            rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
            rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

            # mean-zeroing and max-normalize
            # rdm_corr_above -= np.mean(rdm_corr_above)
            rdm_corr_norm = rdm_corr_above / np.max(rdm_corr_above)
            # rdm_corr_norm = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))

            # rdm_match_above -= np.mean(rdm_match_above)
            rdm_match_norm = rdm_match_above / np.max(rdm_match_above)
            # rdm_match_norm = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

            # get rdm_fmri
            rdm_fmri_roi = rdm_fmri_mean[roi]

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            # rdm_fmri_above -= np.mean(rdm_fmri_above)

            # normalize by dividing max value
            rdm_fmri_vec = rdm_fmri_above / np.max(rdm_fmri_above)
            # rdm_fmri_vec = rdm_fmri_above / np.sqrt(np.sum(rdm_fmri_above**2))

            rdm_corr_roi = w_corr[roi] * rdm_corr_norm
            rdm_match_roi = w_match[roi] * rdm_match_norm
            # rdm_corr_roi = w_bootstrap_fin[roi, 0]*rdm_corr_norm
            # rdm_match_roi = w_bootstrap_fin[roi, 1]*rdm_match_norm
            rdm_fit = rdm_corr_roi + rdm_match_roi

            # calculate kendalltau
            kendall = kendalltau(rdm_fmri_vec, rdm_fit)

            # reconstruct rdm
            rdm_fmri_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_fmri_reconstruct[np.triu_indices(6, k=1)] = rdm_fmri_vec
            # copy upper to lower triangle
            i_lower = np.tril_indices(6, k=-1)
            rdm_fmri_reconstruct[i_lower] = rdm_fmri_reconstruct.T[i_lower]

            rdm_fit_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_fit_reconstruct[np.triu_indices(6, k=1)] = rdm_fit
            # copy upper to lower triangle
            rdm_fit_reconstruct[i_lower] = rdm_fit_reconstruct.T[i_lower]

            # calculate rdm_residual
            rdm_resid = np.abs(rdm_fmri_reconstruct - rdm_fit_reconstruct)

            sns.heatmap(
                rdm_fmri_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 0],
                xticklabels=False,
                yticklabels=False,
            )

            sns.heatmap(
                rdm_fit_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 1],
                xticklabels=False,
                yticklabels=False,
            )
            axes[roi, 1].set_title("kendalltau={}".format(str(np.round(kendall[0], 3))))

            sns.heatmap(
                rdm_resid,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 2],
                xticklabels=False,
                yticklabels=False,
            )

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM_Standard/PlotHeat_cmm_rdm_fit_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotHeat_rdm_MTloc(self, save_flag):
        """
        compares rdm fMRI: RDM_with_localizer vs RDM_without_localizer

        for visualization, the RDM are only max-normalize, without
        mean-centeringfor visualization, the RDM are only max-normalize



        Parameters
        ----------

        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_best[:, 0] -> w_corr
            w_cmm_best[:, 1] -> w_match
            w_cmm_best[:, 2] -> dc value

        mtd: [string]
            the computational method for computing the distance between two voxel
            response patterns to generate RDM.
            for example: "sqeuclidean"

        sawtooth_noise_std: [scalar]
            the standard deviation of the noise to jitter the sawtooth distribution
            in the disparity column map.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # get participants with MT localizer
        # get sbjID number who has MT localizer
        sbjID_mt = []
        for sbj in range(len(self.sbjID_with_MTlocalizer)):
            sbjID = self.sbjID_with_MTlocalizer[sbj]
            idx = self.sbjID_all.index(sbjID)
            sbjID_mt.append(idx)
        rdm_fmri_mtloc = self.rdm_fmri_all[sbjID_mt]  # [sbjID, roi, 6, 6]
        rdm_fmri_mtloc_mean = np.mean(rdm_fmri_mtloc, axis=0)

        # average rdm_fmri_all across sbjID
        # rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)
        rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (8, 20)
        n_row = 8
        n_col = 2

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(0.5, 1.03, "RDM_fMRI all vs. with localizer", ha="center")

        fig.text(
            0.0,
            0.925,
            "V1",
            va="center",
        )
        fig.text(0.0, 0.81, "V2", va="center")
        fig.text(0.0, 0.69, "V3", va="center")
        fig.text(-0.0, 0.575, "V3A", va="center")
        fig.text(-0.0, 0.45, "V3B", va="center")
        fig.text(-0.0, 0.325, "hV4", va="center")
        fig.text(-0.0, 0.2, "V7", va="center")
        fig.text(-0.0, 0.08, "MT", va="center")

        fig.text(0.25, 1.0, "avg all participants", ha="center")
        fig.text(0.72, 1.0, "with_localizer", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        v_min = 0.0
        v_max = 1.0
        cmap = "jet"
        for roi in range(self.n_ROIs):

            # get rdm_fmri avg across all participants
            rdm_fmri_roi = rdm_fmri_mean[roi]
            rdm_fmri_roi = rdm_fmri_roi / rdm_fmri_roi.max()

            # rdm_fmri_with_localizer
            rdm_fmri_mtloc_roi = rdm_fmri_mtloc_mean[roi]
            rdm_fmri_mtloc_roi = rdm_fmri_mtloc_roi / rdm_fmri_mtloc_roi.max()

            sns.heatmap(
                rdm_fmri_roi,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 0],
                xticklabels=False,
                yticklabels=False,
            )

            sns.heatmap(
                rdm_fmri_mtloc_roi,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 1],
                xticklabels=False,
                yticklabels=False,
            )

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM_Standard/PlotHeat_rdm_fmri_MTloc.pdf",
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotHeat_rdm_cmm(self, noise_dispCol_sigma_list, save_flag):
        """
        for visualization, the RDM are only max-normalize, without
        mean-centering

        Parameters
        ----------

        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_best[:, 0] -> w_corr
            w_cmm_best[:, 1] -> w_match
            w_cmm_best[:, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """
        v_min = 0.0
        v_max = 1.0
        cmap = "jet"

        for d in range(len(noise_dispCol_sigma_list)):

            sawtooth_noise_std = noise_dispCol_sigma_list[d]
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                    sawtooth_noise_std
                )
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                    sawtooth_noise_std
                )
            )

            # average rdm_corr and rdm_match across bootstrap
            rdm_corr_bootstrap = np.mean(rdm_corr, axis=0)
            rdm_match_bootstrap = np.mean(rdm_match, axis=0)

            # get above diagonal element
            rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
            rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

            # mean-zeroing and max-normalize
            # rdm_corr_above -= np.mean(rdm_corr_above)
            rdm_corr_norm = rdm_corr_above / np.max(rdm_corr_above)
            # rdm_corr_norm = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))

            # rdm_match_above -= np.mean(rdm_match_above)
            rdm_match_norm = rdm_match_above / np.max(rdm_match_above)
            # rdm_match_norm = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

            rdm_corr_roi = rdm_corr_norm
            rdm_match_roi = rdm_match_norm
            # rdm_fit = rdm_corr_roi + rdm_match_roi

            # reconstruct rdm_corr
            rdm_corr_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_corr_reconstruct[np.triu_indices(6, k=1)] = rdm_corr_roi
            # copy upper to lower triangle
            i_lower = np.tril_indices(6, k=-1)
            rdm_corr_reconstruct[i_lower] = rdm_corr_reconstruct.T[i_lower]

            # reconstruct rdm_match
            rdm_match_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_match_reconstruct[np.triu_indices(6, k=1)] = rdm_match_roi
            # copy upper to lower triangle
            rdm_match_reconstruct[i_lower] = rdm_match_reconstruct.T[i_lower]

            # start plotting
            sns.set_theme()
            sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

            figsize = (8, 3.75)
            n_row = 1
            n_col = 2

            fig, axes = plt.subplots(
                nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
            )

            fig.text(
                0.5,
                1.03,
                "RDM_Corr and RDM_Match {}\nsawtooth_noise_std {:.2f}".format(
                    self.mtd, sawtooth_noise_std
                ),
                ha="center",
            )

            fig.text(0.25, 0.0, "rdm_corr", ha="center")
            fig.text(0.75, 0.0, "rdm_match", ha="center")

            fig.tight_layout()

            plt.subplots_adjust(wspace=0.2, hspace=0.3)

            sns.heatmap(
                rdm_corr_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[0],
                xticklabels=False,
                yticklabels=False,
            )

            sns.heatmap(
                rdm_match_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[1],
                xticklabels=False,
                yticklabels=False,
            )

            if save_flag == 1:
                fig.savefig(
                    "../../../Plots/CMM_Standard/PlotHeat_rdm_cmm_dispColNoise_{:.2f}_{}.pdf".format(
                        sawtooth_noise_std, self.mtd
                    ),
                    dpi=self.dpi,
                    bbox_inches="tight",
                )

    def plotHeat_rdm_cmm_all_ROIs(
        self, noise_dispCol_sigma_list, w_cmm_best, id_best, save_flag
    ):
        """
        for visualization, the RDM are only max-normalize, without
        mean-centering

        Parameters
        ----------

        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_best[:, 0] -> w_corr
            w_cmm_best[:, 1] -> w_match
            w_cmm_best[:, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average w_cmm_bootstrap across sbjID and bootstrap
        w_cmm = np.mean(np.mean(w_cmm_best, axis=2), axis=0)  # [nROIs, 3]

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        den = w_cmm[:, 0] + w_cmm[:, 1] + tol
        w_corr = w_cmm[:, 0] / den
        w_match = w_cmm[:, 1] / den
        # w_corr = w_cmm[:, 0]
        # w_match = w_cmm[:, 1]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (8, 20)
        n_row = 8
        n_col = 2

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(0.5, 1.03, "RDM_Corr and RDM_Match {}".format(self.mtd), ha="center")

        fig.text(
            0.0,
            0.925,
            "V1",
            va="center",
        )
        fig.text(0.0, 0.81, "V2", va="center")
        fig.text(0.0, 0.69, "V3", va="center")
        fig.text(-0.0, 0.575, "V3A", va="center")
        fig.text(-0.0, 0.45, "V3B", va="center")
        fig.text(-0.0, 0.325, "hV4", va="center")
        fig.text(-0.0, 0.2, "V7", va="center")
        fig.text(-0.0, 0.08, "MT", va="center")

        fig.text(0.25, 0.0, "rdm_corr", ha="center")
        fig.text(0.75, 0.0, "rdm_match", ha="center")
        # fig.text(0.85, 0.0,
        #          "rdm_residual",
        #          ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        v_min = 0.0
        v_max = 1.0
        cmap = "jet"
        for roi in range(self.n_ROIs):

            d_roi = id_best[roi]
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            # average rdm_corr and rdm_match across bootstrap
            rdm_corr_bootstrap = np.mean(rdm_corr, axis=0)
            rdm_match_bootstrap = np.mean(rdm_match, axis=0)

            # get above diagonal element
            rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
            rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

            # mean-zeroing and max-normalize
            # rdm_corr_above -= np.mean(rdm_corr_above)
            rdm_corr_norm = rdm_corr_above / np.max(rdm_corr_above)
            # rdm_corr_norm = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))

            # rdm_match_above -= np.mean(rdm_match_above)
            rdm_match_norm = rdm_match_above / np.max(rdm_match_above)
            # rdm_match_norm = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

            rdm_corr_roi = w_corr[roi] * rdm_corr_norm
            rdm_match_roi = w_match[roi] * rdm_match_norm
            # rdm_corr_roi = w_bootstrap_fin[roi, 0]*rdm_corr_norm
            # rdm_match_roi = w_bootstrap_fin[roi, 1]*rdm_match_norm
            # rdm_fit = rdm_corr_roi + rdm_match_roi

            # calculate pearsonr between rdm_corr and rdm_match
            r = pearsonr(rdm_corr_roi, rdm_match_roi)

            # reconstruct rdm_corr
            rdm_corr_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_corr_reconstruct[np.triu_indices(6, k=1)] = rdm_corr_roi
            # copy upper to lower triangle
            i_lower = np.tril_indices(6, k=-1)
            rdm_corr_reconstruct[i_lower] = rdm_corr_reconstruct.T[i_lower]

            # reconstruct rdm_match
            rdm_match_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_match_reconstruct[np.triu_indices(6, k=1)] = rdm_match_roi
            # copy upper to lower triangle
            rdm_match_reconstruct[i_lower] = rdm_match_reconstruct.T[i_lower]

            sns.heatmap(
                rdm_corr_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 0],
                xticklabels=False,
                yticklabels=False,
            )

            sns.heatmap(
                rdm_match_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                ax=axes[roi, 1],
                xticklabels=False,
                yticklabels=False,
            )
            axes[roi, 1].set_title("pearsonr={}".format(str(np.round(r[0], 3))))

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM_Standard/PlotHeat_rdm_cmm_all_ROIs_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBar_w_cmm_ratio(self, w_cmm_best, save_flag):
        """

        Parameters
        ----------
        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            w_cmm_best.

            w_cmm_best[:, :, :, 0] -> w_corr
            w_cmm_best[:, :, :, 1] -> w_match
            w_cmm_best[:, :, :, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across sbjID
        w_cmm_avg = np.mean(w_cmm_best, axis=2)

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        den = w_cmm_avg[:, :, 0] + w_cmm_avg[:, :, 1] + tol
        # den = 1
        num = w_cmm_avg[:, :, 0]

        w_cmm_ratio = num / den
        # average across  bootstrap
        y = np.mean(w_cmm_ratio, axis=0)

        # use standard deviation because it uses bootstrap
        y_err = np.std(w_cmm_ratio, axis=0)
        y_sem = sem(w_cmm_ratio, axis=0)

        # compute one-way ANOVA
        dof_between = len(self.ROIs) - 1
        dof_within = len(w_cmm_ratio.flatten()) - len(self.ROIs)
        F_val, p_val = stats.f_oneway(
            w_cmm_ratio[:, 0],
            w_cmm_ratio[:, 1],
            w_cmm_ratio[:, 2],
            w_cmm_ratio[:, 3],
            w_cmm_ratio[:, 4],
            w_cmm_ratio[:, 5],
            w_cmm_ratio[:, 6],
            w_cmm_ratio[:, 7],
        )

        # post-hoc test: tukey
        res = stats.tukey_hsd(
            w_cmm_ratio[:, 0],
            w_cmm_ratio[:, 1],
            w_cmm_ratio[:, 2],
            w_cmm_ratio[:, 3],
            w_cmm_ratio[:, 4],
            w_cmm_ratio[:, 5],
            w_cmm_ratio[:, 6],
            w_cmm_ratio[:, 7],
        )
        # [stat, p_val, lower_ci, upper_ci]
        n_row = len(self.ROIs) ** 2
        tukey_stat = np.empty((n_row, 4), dtype=np.float32)
        col_comp = []
        for i, roi1 in enumerate(self.ROIs):
            id_start = i * len(self.ROIs)
            id_end = id_start + len(self.ROIs)
            tukey_stat[id_start:id_end, 0] = res.statistic[i]  # mean difference
            tukey_stat[id_start:id_end, 1] = res.pvalue[i]  # p_val
            tukey_stat[id_start:id_end, 2] = res.confidence_interval().low[
                i
            ]  # 95% CI lower bound
            tukey_stat[id_start:id_end, 3] = res.confidence_interval().high[
                i
            ]  # 95% CI upper bound

            # comparison
            for j, roi2 in enumerate(self.ROIs):
                comp = (roi1, roi2)
                col_comp.append(comp)

        temp = {
            "comparison": col_comp,
            "mean_diff": tukey_stat[:, 0],
            "p_val": tukey_stat[:, 1],
            "95%_ci_low": tukey_stat[:, 2],
            "95%_ci_up": tukey_stat[:, 3],
        }
        tukey_df = pd.DataFrame(temp)

        # compute confidence interval
        alpha = 0.05
        dof_within = len(w_cmm_ratio.flatten()) - len(self.ROIs)
        t = stats.t.ppf(1 - (alpha / 2), dof_within)
        # margin of error
        d = t * y_sem
        # intervals
        ci_upper = y + d
        ci_lower = y - d

        # create pandas dataframe for csv file
        data = pd.DataFrame(
            {
                "ROI": self.ROIs,
                "w_ratio": y,
                "w_std": y_err,
                "margin_error": d,
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "dof_between": dof_between,
                "dof_within": dof_within,
                "F_1way_anova": F_val,
                "p_val_1way_anova": p_val,
            }
        )

        # plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        pos = list(np.arange(0, 24, 3))
        error_kw = dict(lw=3, capsize=7, capthick=3)

        plt.figure(figsize=(14, 10))
        plt.bar(
            pos,
            y,
            yerr=y_err,
            width=2.5,
            fill=False,
            error_kw=error_kw,
            edgecolor="black",
            linewidth=3,
        )

        # plot line
        plt.plot([-2, 23], [0.5, 0.5], "r--", linewidth=3)

        plt.xticks(pos, self.ROIs)
        # plt.xlabel("ROI")
        plt.ylabel("ratio")
        plt.title("w_corr/(w_corr+w_match)", pad=20)

        y_low = 0.0
        y_up = 1.01
        y_step = 0.25
        plt.ylim(y_low, y_up)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotBar_w_cmm_ratio_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

            # save statistic to csv
            data.to_csv("../../../Data/CMM/w_cmm_ratio_stats.csv", index=False)
            tukey_df.to_csv("../../../Data/CMM/w_cmm_ratio_tukey.csv", index=False)

    def plotBox_w_cmm_ratio(self, w_cmm_best, save_flag):
        """

        Parameters
        ----------
        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            w_cmm_best.

            w_cmm_best[:, :, :, 0] -> w_corr
            w_cmm_best[:, :, :, 1] -> w_match
            w_cmm_best[:, :, :, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across sbjID
        w_cmm_avg = np.mean(w_cmm_best, axis=2)

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        den = w_cmm_avg[:, :, 0] + w_cmm_avg[:, :, 1] + tol
        # den = 1
        num = w_cmm_avg[:, :, 0]

        w_cmm_ratio = num / den
        # average across  bootstrap
        y = np.mean(w_cmm_ratio, axis=0)

        # use standard deviation because it uses bootstrap
        y_err = np.std(w_cmm_ratio, axis=0)
        y_sem = sem(w_cmm_ratio, axis=0)

        # compute one-way ANOVA
        dof_between = len(self.ROIs) - 1
        dof_within = len(w_cmm_ratio.flatten()) - len(self.ROIs)
        F_val, p_val = stats.f_oneway(
            w_cmm_ratio[:, 0],
            w_cmm_ratio[:, 1],
            w_cmm_ratio[:, 2],
            w_cmm_ratio[:, 3],
            w_cmm_ratio[:, 4],
            w_cmm_ratio[:, 5],
            w_cmm_ratio[:, 6],
            w_cmm_ratio[:, 7],
        )

        # post-hoc test: tukey
        res = stats.tukey_hsd(
            w_cmm_ratio[:, 0],
            w_cmm_ratio[:, 1],
            w_cmm_ratio[:, 2],
            w_cmm_ratio[:, 3],
            w_cmm_ratio[:, 4],
            w_cmm_ratio[:, 5],
            w_cmm_ratio[:, 6],
            w_cmm_ratio[:, 7],
        )
        # [stat, p_val, lower_ci, upper_ci]
        n_row = len(self.ROIs) ** 2
        tukey_stat = np.empty((n_row, 4), dtype=np.float32)
        col_comp = []
        for i, roi1 in enumerate(self.ROIs):
            id_start = i * len(self.ROIs)
            id_end = id_start + len(self.ROIs)
            tukey_stat[id_start:id_end, 0] = res.statistic[i]  # mean difference
            tukey_stat[id_start:id_end, 1] = res.pvalue[i]  # p_val
            tukey_stat[id_start:id_end, 2] = res.confidence_interval().low[
                i
            ]  # 95% CI lower bound
            tukey_stat[id_start:id_end, 3] = res.confidence_interval().high[
                i
            ]  # 95% CI upper bound

            # comparison
            for j, roi2 in enumerate(self.ROIs):
                comp = (roi1, roi2)
                col_comp.append(comp)

        temp = {
            "comparison": col_comp,
            "mean_diff": tukey_stat[:, 0],
            "p_val": tukey_stat[:, 1],
            "95%_ci_low": tukey_stat[:, 2],
            "95%_ci_up": tukey_stat[:, 3],
        }
        tukey_df = pd.DataFrame(temp)

        # compute confidence interval
        alpha = 0.05
        dof_within = len(w_cmm_ratio.flatten()) - len(self.ROIs)
        t = stats.t.ppf(1 - (alpha / 2), dof_within)
        # margin of error
        d = t * y_sem
        # intervals
        ci_upper = y + d
        ci_lower = y - d

        # create pandas dataframe for csv file
        data = pd.DataFrame(
            {
                "ROI": self.ROIs,
                "w_ratio": y,
                "w_std": y_err,
                "margin_error": d,
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "dof_between": dof_between,
                "dof_within": dof_within,
                "F_1way_anova": F_val,
                "p_val_1way_anova": p_val,
            }
        )

        # plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        pos = list(np.arange(0, 24, 3))
        plt.figure(figsize=(14, 10))

        # box plot properties
        bar_width = 2.5
        linewidth = 1.5
        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="D",
            markersize=6,
            markerfacecolor="firebrick",
            markeredgecolor="firebrick",
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)

        plt.boxplot(
            w_cmm_ratio,
            widths=bar_width,
            patch_artist=True,
            positions=pos,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # Plot line at 0.5
        plt.axhline(0.5, color="red", linestyle="--", linewidth=3)

        plt.xticks(pos, self.ROIs)
        # plt.xlabel("ROI")
        plt.ylabel("ratio")
        plt.title("w_corr/(w_corr+w_match)", pad=20)

        y_low = 0.0
        y_up = 1.01
        y_step = 0.25
        plt.ylim(y_low, y_up)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotBox_w_cmm_ratio_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

            # save statistic to csv
            data.to_csv("../../../Data/CMM/w_cmm_ratio_stats.csv", index=False)
            tukey_df.to_csv("../../../Data/CMM/w_cmm_ratio_tukey.csv", index=False)

    def plotBar_w_cmm(self, w_cmm_best, save_flag):
        """

        Parameters
        ----------
        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            w_cmm_best.

            w_cmm_best[:, :, :, 0] -> w_corr
            w_cmm_best[:, :, :, 1] -> w_match
            w_cmm_best[:, :, :, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across sbjID
        w_cmm_avg = np.mean(w_cmm_best, axis=2)

        w_corr = w_cmm_avg[:, :, 0]
        w_match = w_cmm_avg[:, :, 1]

        # average across  bootstrap
        y_corr = np.mean(w_corr, axis=0)
        y_match = np.mean(w_match, axis=0)

        # use standard deviation because it uses bootstrap
        y_corr_err = np.std(w_corr, axis=0)
        y_match_err = np.std(w_match, axis=0)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (10, 10)
        n_row = 3
        n_col = 3

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        # fig.text(-0.02, 0.5, "Weight", va="center", rotation=90)
        # fig.text(0.5, -0.03, "# voxels", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        pos = [1, 3.5]
        error_kw = dict(lw=3, capsize=7, capthick=3)

        for roi in range(self.n_ROIs):

            id_row = roi // n_row
            id_col = roi % n_col

            y = [y_corr[roi], y_match[roi]]
            y_err = [y_corr_err[roi], y_match_err[roi]]
            axes[id_row, id_col].bar(
                pos, y, yerr=y_err, width=2.5, color="gray", error_kw=error_kw
            )

            # plot line
            # axes[id_row, id_col].plot([0, 3], [0.5, 0.5], "r--", linewidth=3)

            axes[id_row, id_col].set_title(self.ROIs[roi])
            axes[id_row, id_col].set_xticks(pos)
            axes[id_row, id_col].set_xticklabels(["w_corr", "w_match"], fontsize=16)
            # plt.xlabel("ROI")
            # axes[id_row, id_col].ylabel("weight")
            # plt.title("w_corr/(w_corr+w_match)", pad=20)

            y_low = 0.0
            y_up = 0.41
            y_step = 0.1
            axes[id_row, id_col].set_ylim(y_low, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low, y_up, y_step), 2)
            )

            # remove the top and right frames
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotBar_w_cmm_{}.pdf".format(self.mtd),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotScatter_w_corr_vs_w_match(self, w_cmm_best, save_flag):
        """

        Parameters
        ----------
        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            w_cmm_best.

            w_cmm_best[:, :, :, 0] -> w_corr
            w_cmm_best[:, :, :, 1] -> w_match
            w_cmm_best[:, :, :, 2] -> dc value

        nVox_to_analyze : scalar
            the number of voxels to analyze.
            For example: 250

        sawtooth_noise_std: [scalar]
            the standard deviation of the noise to jitter the sawtooth distribution
            in the disparity column map.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across sbjID
        w_cmm_avg = np.mean(w_cmm_best, axis=2)

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        den = w_cmm_avg[:, :, 0] + w_cmm_avg[:, :, 1] + tol
        # den = 1

        w_corr_norm = w_cmm_avg[:, :, 0] / den
        w_match_norm = w_cmm_avg[:, :, 1] / den

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        x_low = 0.0
        x_up = 1.1
        x_step = 0.2
        y_low = 0.0
        y_up = 1.1
        y_step = 0.2

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        figsize = (10, 10)
        n_row = 1
        n_col = 1

        # plot normalized weight
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        # fig.text(0.5, 1.04,
        #             "w_corr VS w_match",
        #             ha="center")
        fig.text(-0.03, 0.5, "w_corr", va="center", rotation=90)
        fig.text(0.5, -0.03, "w_match", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # plot combined frequency
        for roi in range(self.n_ROIs):

            x = np.mean(w_match_norm[:, roi])
            y = np.mean(w_corr_norm[:, roi])
            x_err = np.std(w_match_norm[:, roi])
            y_err = np.std(w_corr_norm[:, roi])

            axes.errorbar(
                x,
                y,
                xerr=x_err,
                yerr=y_err,
                elinewidth=2,
                fmt=markers[roi],
                c="black",
                ms=14,
            )

        axes.legend(self.ROIs, bbox_to_anchor=(1.05, 1))

        axes.plot([x_low, x_up], [x_low, x_up], color="r", linestyle="--", linewidth=4)

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 0.025, x_up)
        axes.set_ylim(y_low - 0.025, y_up)

        # axes.text(0.19, 0.16,
        #             "Cross-\ncorrelation",
        #             color="gray",
        #             weight="bold",
        #             alpha=0.75,
        #             ha="center",
        #             rotation=45)
        # axes.text(0.29, 0.11,
        #             "Cross-\nmatching",
        #             color="gray",
        #             weight="bold",
        #             alpha=0.75,
        #             ha="center",
        #             rotation=45)

        # remove top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotScatter_w_corr_vs_w_match_normalized_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

        # plot non-normalized weight
        x_low = 0.0
        x_up = 0.41
        x_step = 0.1
        y_low = 0.0
        y_up = 0.41
        y_step = 0.1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.04,
            "w_corr VS w_match (non-normalized) {}".format(self.mtd),
            ha="center",
        )
        fig.text(-0.03, 0.5, "w_corr", va="center", rotation=90)
        fig.text(0.5, -0.03, "w_match", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # plot combined frequency
        for roi in range(self.n_ROIs):

            x = np.mean(w_cmm_avg[:, roi, 1])
            y = np.mean(w_cmm_avg[:, roi, 0])
            x_err = np.std(w_cmm_avg[:, roi, 1])
            y_err = np.std(w_cmm_avg[:, roi, 0])

            axes.errorbar(
                x,
                y,
                xerr=x_err,
                yerr=y_err,
                elinewidth=2,
                fmt=markers[roi],
                c="black",
                ms=14,
            )

        axes.legend(self.ROIs, bbox_to_anchor=(1.05, 1))

        axes.plot([x_low, x_up], [x_low, x_up], color="r", linestyle="--", linewidth=4)

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 0.025, x_up)
        axes.set_ylim(y_low - 0.025, y_up)

        # axes.text(0.08, 0.04,
        #             "Cross-\ncorrelation",
        #             color="gray",
        #             weight="bold",
        #             alpha=0.75,
        #             ha="center",
        #             rotation=45)
        # axes.text(0.18, 0.01,
        #             "Cross-\nmatching",
        #             color="gray",
        #             weight="bold",
        #             alpha=0.75,
        #             ha="center",
        #             rotation=45)

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotScatter_w_corr_vs_w_match_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotScatter_w_corr_vs_w_match_MTloc(self, w_cmm_best, save_flag):
        """
        plot scatter w_corr vs w_match in which MT comes from
        participants whose retinotopy scanning used MT localizer


        Parameters
        ----------
        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            w_cmm_best.

            w_cmm_best[:, :, :, 0] -> w_corr
            w_cmm_best[:, :, :, 1] -> w_match
            w_cmm_best[:, :, :, 2] -> dc value

        nVox_to_analyze : scalar
            the number of voxels to analyze.
            For example: 250

        sawtooth_noise_std: [scalar]
            the standard deviation of the noise to jitter the sawtooth distribution
            in the disparity column map.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across sbjID
        w_cmm_avg = np.mean(w_cmm_best, axis=2)

        # get sbjID number who has MT localizer
        sbjID_mt = []
        for sbj in range(len(self.sbjID_with_MTlocalizer)):
            sbjID = self.sbjID_with_MTlocalizer[sbj]
            idx = self.sbjID_all.index(sbjID)
            sbjID_mt.append(idx)
        w_cmm_mtloc = w_cmm_best[:, :, sbjID_mt]
        w_cmm_mtloc_avg = np.mean(w_cmm_mtloc, axis=2)

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        figsize = (10, 10)
        n_row = 1
        n_col = 1

        # plot non-normalized weight
        x_low = 0.0
        x_up = 0.41
        x_step = 0.1
        y_low = 0.0
        y_up = 0.41
        y_step = 0.1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.04,
            "w_corr VS w_match (non-normalized) {}".format(self.mtd),
            ha="center",
        )
        fig.text(-0.03, 0.5, "w_corr", va="center", rotation=90)
        fig.text(0.5, -0.03, "w_match", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # plot combined frequency
        for roi in range(self.n_ROIs):

            if roi != self.n_ROIs - 1:  # not MT

                x = np.mean(w_cmm_avg[:, roi, 1])
                y = np.mean(w_cmm_avg[:, roi, 0])
                x_err = np.std(w_cmm_avg[:, roi, 1])
                y_err = np.std(w_cmm_avg[:, roi, 0])

                axes.errorbar(
                    x,
                    y,
                    xerr=x_err,
                    yerr=y_err,
                    elinewidth=2,
                    fmt=markers[roi],
                    c="black",
                    ms=14,
                )
            else:  # get participants with MT localizer
                x = np.mean(w_cmm_mtloc_avg[:, roi, 1])
                y = np.mean(w_cmm_mtloc_avg[:, roi, 0])
                x_err = np.std(w_cmm_mtloc_avg[:, roi, 1])
                y_err = np.std(w_cmm_mtloc_avg[:, roi, 0])

                axes.errorbar(
                    x,
                    y,
                    xerr=x_err,
                    yerr=y_err,
                    elinewidth=2,
                    fmt=markers[roi],
                    c="black",
                    ms=14,
                )

        axes.legend(self.ROIs, bbox_to_anchor=(1.05, 1))

        axes.plot([x_low, x_up], [x_low, x_up], color="r", linestyle="--", linewidth=4)

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 0.025, x_up)
        axes.set_ylim(y_low - 0.025, y_up)

        # axes.text(0.08, 0.04,
        #             "Cross-\ncorrelation",
        #             color="gray",
        #             weight="bold",
        #             alpha=0.75,
        #             ha="center",
        #             rotation=45)
        # axes.text(0.18, 0.01,
        #             "Cross-\nmatching",
        #             color="gray",
        #             weight="bold",
        #             alpha=0.75,
        #             ha="center",
        #             rotation=45)

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotScatter_w_corr_vs_w_match_MTloc_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBar_goodness_of_fit(
        self,
        noise_dispCol_sigma_list,
        w_cmm_best,
        id_best,
        save_flag,
    ):
        """

        Parameters
        ----------
        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_bootstrap[:, :, :, 0] -> w_corr
            w_cmm_bootstrap[:, :, :, 1] -> w_match
            w_cmm_bootstrap[:, :, :, 2] -> dc value

        kendalltau_low : [n_ROIs, n_bootstrap, n_sbjID] np.array
            the lower bound of noise ceiling evaluated by the kendalltau
            correlation coefficient.

        kendalltau_up : [n_ROIs, n_bootstrap] np.array
            the upper bound of noise ceiling evaluated by the kendalltau
            correlation coefficient.

        nVox_to_analyze : integer
            the number of voxels to analyze.
            For example: 250

        sawtooth_noise_std: integer
            the standard deviation of the noise to jitter the sawtooth distribution
            in the disparity column map.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """
        # average rdm_fmri_all across sbjID
        # rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)
        rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)

        # average w_cmm_bootstrap across sbjID and bootstrap
        w_cmm = np.mean(np.mean(w_cmm_best, axis=2), axis=0)  # [nROIs, 3]

        # normalize weight such that w_corr + w_match = 1
        # tol = 1e-6
        # den = (w_cmm[:, 0] + w_cmm[:, 1] + tol)
        # w_corr = w_cmm[:, 0] / den
        # w_match = w_cmm[:, 1] / den
        w_corr = w_cmm[:, 0]
        w_match = w_cmm[:, 1]

        kendall_all = np.zeros((self.n_ROIs, 2), dtype=np.float32)

        for roi in range(self.n_ROIs):

            d_roi = id_best[roi]
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            # average rdm_corr and rdm_match across bootstrap
            rdm_corr_bootstrap = np.mean(rdm_corr, axis=0)
            rdm_match_bootstrap = np.mean(rdm_match, axis=0)

            # get above diagonal element
            rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
            rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

            # mean-zeroing and max-normalize
            rdm_corr_above -= np.mean(rdm_corr_above)
            rdm_corr_norm = rdm_corr_above / np.max(rdm_corr_above)
            # rdm_corr_norm = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))

            rdm_match_above -= np.mean(rdm_match_above)
            rdm_match_norm = rdm_match_above / np.max(rdm_match_above)
            # rdm_match_norm = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

            # get rdm_fmri
            rdm_fmri_roi = rdm_fmri_mean[roi]

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_above -= np.mean(rdm_fmri_above)

            # normalize by dividing max value
            rdm_fmri_vec = rdm_fmri_above / np.max(rdm_fmri_above)
            # rdm_fmri_vec = rdm_fmri_above / np.sqrt(np.sum(rdm_fmri_above**2))

            rdm_corr_roi = w_corr[roi] * rdm_corr_norm
            rdm_match_roi = w_match[roi] * rdm_match_norm
            # rdm_corr_roi = w_bootstrap_fin[roi, 0]*rdm_corr_norm
            # rdm_match_roi = w_bootstrap_fin[roi, 1]*rdm_match_norm
            rdm_fit = rdm_corr_roi + rdm_match_roi

            # calculate kendalltau
            kendall_all[roi] = kendalltau(rdm_fmri_vec, rdm_fit)

        # get the median of low and upper noise ceiling
        # kendalltau_low, [n_ROIs, n_bootstrap, n_sbjID]
        # kendalltau_up, [n_ROIs, n_bootsrap]
        bound_low = np.median(np.mean(self.kendalltau_low, axis=2), axis=1)
        bound_up = np.median(self.kendalltau_up, axis=1)

        # plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        pos = list(np.arange(0, 24, 3))
        y = kendall_all[:, 0]

        plt.figure(figsize=(14, 10))
        plt.bar(
            pos, y, width=2.5, fill=False, capsize=3, linewidth=3, edgecolor="black"
        )

        # plot lower and upper noise ceiling
        for roi in range(self.n_ROIs):
            plt.plot(
                [pos[roi] - 1, pos[roi] + 1],
                [bound_up[roi], bound_up[roi]],
                "r--",
                linewidth=3,
            )
            plt.plot(
                [pos[roi] - 1, pos[roi] + 1],
                [bound_low[roi], bound_low[roi]],
                "b--",
                linewidth=3,
            )

        plt.xticks(pos, self.ROIs)
        # plt.xlabel("ROI")
        plt.ylabel("kendalltau")
        plt.title("Goodness of fit {}".format(self.mtd), pad=20)

        y_low = 0.0
        y_up = 1.01
        y_step = 0.2
        plt.ylim(y_low, y_up)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        # convert kendall_all to csv and save
        colnames = ["kendalltau", "p-val"]
        kendall_pd = pd.DataFrame(kendall_all, columns=colnames)
        kendall_pd.loc[:, "ROI"] = self.ROIs
        if save_flag == 1:
            kendall_pd.to_csv("../../../Data/CMM/kendall.csv", index=False)
        print(kendall_pd)

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotBar_goodness_of_fit_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBar_goodness_of_fit_with_errorbar(
        self,
        noise_dispCol_sigma_list,
        w_cmm_best,
        id_best,
        save_flag,
    ):
        """

        Parameters
        ----------
        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_bootstrap[:, :, :, 0] -> w_corr
            w_cmm_bootstrap[:, :, :, 1] -> w_match
            w_cmm_bootstrap[:, :, :, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        n_bootstrap = w_cmm_best.shape[0]

        # average rdm_fmri_all across sbjID
        rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)

        # average w_cmm_bootstrap across sbjID
        w_cmm = np.mean(w_cmm_best, axis=2)  # [n_bootstrap, nROIs, 3]

        # normalize weight such that w_corr + w_match = 1
        # tol = 1e-6
        # den = (w_cmm[:, :, 0] + w_cmm[:, :, 1] + tol)
        # w_corr = w_cmm[:, :, 0] / den
        # w_match = w_cmm[:, :, 1] / den
        w_corr = w_cmm[:, :, 0]
        w_match = w_cmm[:, :, 1]

        kendall_all = np.zeros((self.n_ROIs, n_bootstrap, 2), dtype=np.float32)

        for roi in range(self.n_ROIs):

            # get rdm_fmri
            rdm_fmri_roi = rdm_fmri_mean[roi]

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_above -= np.mean(rdm_fmri_above)

            # normalize by dividing max value
            rdm_fmri_vec = rdm_fmri_above / np.max(rdm_fmri_above)
            # rdm_fmri_vec = rdm_fmri_above / np.sqrt(np.sum(rdm_fmri_above**2))

            d_roi = id_best[roi]
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            for i in range(n_bootstrap):

                # average rdm_corr and rdm_match across bootstrap
                rdm_corr_bootstrap = rdm_corr[i]
                rdm_match_bootstrap = rdm_match[i]

                # get above diagonal element
                rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
                rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

                # mean-zeroing and max-normalize
                rdm_corr_above -= np.mean(rdm_corr_above)
                rdm_corr_norm = rdm_corr_above / np.max(rdm_corr_above)
                # rdm_corr_norm = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))

                rdm_match_above -= np.mean(rdm_match_above)
                rdm_match_norm = rdm_match_above / np.max(rdm_match_above)
                # rdm_match_norm = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

                rdm_corr_roi = w_corr[i, roi] * rdm_corr_norm
                rdm_match_roi = w_match[i, roi] * rdm_match_norm
                # rdm_corr_roi = w_bootstrap_fin[roi, 0]*rdm_corr_norm
                # rdm_match_roi = w_bootstrap_fin[roi, 1]*rdm_match_norm
                rdm_fit = rdm_corr_roi + rdm_match_roi

                # calculate kendalltau
                kendall_all[roi, i] = kendalltau(rdm_fmri_vec, rdm_fit)

        # get the median of low and upper noise ceiling
        # kendalltau_low, [n_ROIs, n_bootstrap, n_sbjID]
        # kendalltau_up, [n_ROIs, n_bootsrap]
        bound_low = np.median(np.mean(self.kendalltau_low, axis=2), axis=1)
        bound_up = np.median(self.kendalltau_up, axis=1)

        # plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        # average across n_bootstrap
        kendall_all_avg = np.mean(kendall_all, axis=1)
        pos = list(np.arange(0, 24, 3))
        y = kendall_all_avg[:, 0]
        y_err = np.std(kendall_all, axis=1)[:, 0]

        error_kw = dict(lw=3, capsize=7, capthick=3)

        plt.figure(figsize=(14, 10))
        plt.bar(
            pos,
            y,
            yerr=y_err,
            width=2.5,
            fill=False,
            error_kw=error_kw,
            edgecolor="black",
            linewidth=3,
        )

        # plot lower and upper noise ceiling
        for roi in range(self.n_ROIs):
            plt.plot(
                [pos[roi] - 1, pos[roi] + 1],
                [bound_up[roi], bound_up[roi]],
                "r--",
                linewidth=3,
            )
            plt.plot(
                [pos[roi] - 1, pos[roi] + 1],
                [bound_low[roi], bound_low[roi]],
                "b--",
                linewidth=3,
            )

        plt.xticks(pos, self.ROIs)
        # plt.xlabel("ROI")
        plt.ylabel("kendalltau")
        plt.title("Goodness of fit {}".format(self.mtd), pad=20)

        y_low = 0.0
        y_up = 1.01
        y_step = 0.2
        plt.ylim(y_low, y_up)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        # convert kendall_all to csv and save
        colnames = ["kendalltau", "p-val"]
        kendall_pd = pd.DataFrame(kendall_all_avg, columns=colnames)
        kendall_pd.loc[:, "ROI"] = self.ROIs
        kendall_pd.to_csv("../../../../Data/CMM/kendall_with_errorbar.csv", index=False)
        print(kendall_pd)

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotBar_goodness_of_fit_with_errorbar_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBox_goodness_of_fit(
        self,
        noise_dispCol_sigma_list,
        w_cmm_best,
        id_best,
        save_flag,
    ):
        """

        Parameters
        ----------
        noise_dispCol_sigma_list : np.array
            a list of noise level for disparity column,
            the variability in the sawtooth profile

            ex:
            noise_dispCol_sigma_list = np.array(
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)

        w_cmm_best : [n_bootstrap, len(self.ROIs), n_sbjID, 3] np.array
            estimated correlation and match weights.

            w_cmm_bootstrap[:, :, :, 0] -> w_corr
            w_cmm_bootstrap[:, :, :, 1] -> w_match
            w_cmm_bootstrap[:, :, :, 2] -> dc value

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        n_bootstrap = w_cmm_best.shape[0]

        # average rdm_fmri_all across sbjID
        rdm_fmri_mean = np.mean(self.rdm_fmri_all, axis=0)

        # average w_cmm_bootstrap across sbjID
        w_cmm = np.mean(w_cmm_best, axis=2)  # [n_bootstrap, nROIs, 3]

        # normalize weight such that w_corr + w_match = 1
        # tol = 1e-6
        # den = (w_cmm[:, :, 0] + w_cmm[:, :, 1] + tol)
        # w_corr = w_cmm[:, :, 0] / den
        # w_match = w_cmm[:, :, 1] / den
        w_corr = w_cmm[:, :, 0]
        w_match = w_cmm[:, :, 1]

        kendall_all = np.zeros((n_bootstrap, self.n_ROIs, 2), dtype=np.float32)

        for roi in range(self.n_ROIs):

            # get rdm_fmri
            rdm_fmri_roi = rdm_fmri_mean[roi]

            # get above diagonal elements
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing
            rdm_fmri_above -= np.mean(rdm_fmri_above)

            # normalize by dividing max value
            rdm_fmri_vec = rdm_fmri_above / np.max(rdm_fmri_above)
            # rdm_fmri_vec = rdm_fmri_above / np.sqrt(np.sum(rdm_fmri_above**2))

            d_roi = id_best[roi]
            rdm_corr = np.load(
                "../../../Data/CMM/rdm_corr_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            rdm_match = np.load(
                "../../../Data/CMM/rdm_match_dispColNoise_{:.2f}.npy".format(
                    noise_dispCol_sigma_list[d_roi]
                )
            )

            for i in range(n_bootstrap):

                # average rdm_corr and rdm_match across bootstrap
                rdm_corr_bootstrap = rdm_corr[i]
                rdm_match_bootstrap = rdm_match[i]

                # get above diagonal element
                rdm_corr_above = rdm_corr_bootstrap[np.triu_indices(6, k=1)]
                rdm_match_above = rdm_match_bootstrap[np.triu_indices(6, k=1)]

                # mean-zeroing and max-normalize
                rdm_corr_above -= np.mean(rdm_corr_above)
                rdm_corr_norm = rdm_corr_above / np.max(rdm_corr_above)
                # rdm_corr_norm = rdm_corr_above / np.sqrt(np.sum(rdm_corr_above**2))

                rdm_match_above -= np.mean(rdm_match_above)
                rdm_match_norm = rdm_match_above / np.max(rdm_match_above)
                # rdm_match_norm = rdm_match_above / np.sqrt(np.sum(rdm_match_above**2))

                rdm_corr_roi = w_corr[i, roi] * rdm_corr_norm
                rdm_match_roi = w_match[i, roi] * rdm_match_norm
                # rdm_corr_roi = w_bootstrap_fin[roi, 0]*rdm_corr_norm
                # rdm_match_roi = w_bootstrap_fin[roi, 1]*rdm_match_norm
                rdm_fit = rdm_corr_roi + rdm_match_roi

                # calculate kendalltau
                kendall_all[i, roi] = kendalltau(rdm_fmri_vec, rdm_fit)

        # get the median of low and upper noise ceiling
        # kendalltau_low, [n_ROIs, n_bootstrap, n_sbjID]
        # kendalltau_up, [n_ROIs, n_bootsrap]
        bound_low = np.median(np.mean(self.kendalltau_low, axis=2), axis=1)
        bound_up = np.median(self.kendalltau_up, axis=1)

        # plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="deep")

        # average across n_bootstrap
        kendall_all_avg = np.mean(kendall_all, axis=0)
        pos = list(np.arange(0, 24, 3))
        y = kendall_all[:, :, 0]  # fetch only kendalltau values

        plt.figure(figsize=(14, 10))
        # box plot properties
        bar_width = 2.5
        linewidth = 1.5
        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="D",
            markersize=6,
            markerfacecolor="firebrick",
            markeredgecolor="firebrick",
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)

        plt.boxplot(
            y,
            widths=bar_width,
            patch_artist=True,
            positions=pos,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # plot lower and upper noise ceiling
        for roi in range(self.n_ROIs):
            plt.plot(
                [pos[roi] - 1, pos[roi] + 1],
                [bound_up[roi], bound_up[roi]],
                "r--",
                linewidth=3,
            )
            plt.plot(
                [pos[roi] - 1, pos[roi] + 1],
                [bound_low[roi], bound_low[roi]],
                "b--",
                linewidth=3,
            )

        plt.xticks(pos, self.ROIs)
        # plt.xlabel("ROI")
        plt.ylabel("kendalltau")
        plt.title("Goodness of fit {}".format(self.mtd), pad=20)

        y_low = 0.0
        y_up = 1.01
        y_step = 0.2
        plt.ylim(y_low, y_up)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        # convert kendall_all to csv and save
        colnames = ["kendalltau", "p-val"]
        kendall_pd = pd.DataFrame(kendall_all_avg, columns=colnames)
        kendall_pd.loc[:, "ROI"] = self.ROIs

        print(kendall_pd)

        # save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM_Standard/PlotBox_goodness_of_fit_{}.pdf".format(
                    self.mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

            kendall_pd.to_csv(
                "../../../Data/CMM/kendall_with_errorbar.csv", index=False
            )
