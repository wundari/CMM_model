#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:41:38 2021

    cd /NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM

    script for simulating cross-correlation and cross-matching with specific
    parameters in disparity column model for each brain region

@author: cogni
"""


import numpy as np
import tensorflow as tf

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io as sio
from scipy.stats import sem
from scipy import signal
from scipy.optimize import nnls
from scipy.stats import kendalltau
from scipy.spatial.distance import cdist

from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from sklearn.linear_model import LinearRegression

# import sys
# sys.path.append('../common/')
# import tool as tool

from timeit import default_timer as timer

# import gc

# from disparity_column_model import DisparityColumn
# from RSA import RSA

from Common.Common import General
from Common.Common import PlotGeneral
from DisparityColumn.DisparityColumn import DisparityColumn


class SquareWindow:

    def __init__(self, f_batch, size_rds_bg_pix):

        self.f_batch = f_batch
        self.size_rds_bg_pix = size_rds_bg_pix

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

    def create_square_window(self, f):
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

        # compute fwhm, pixel
        fwhm_pix = self.fwhm(f)
        # fwhm_pix = fwhm(f)

        # define square window w
        w = np.zeros((self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.float32)
        id_start = np.int(self.size_rds_bg_pix / 2 - fwhm_pix / 2)
        id_end = np.int(self.size_rds_bg_pix / 2 + fwhm_pix / 2)
        w[id_start:id_end, id_start:id_end] = 1
        # plt.imshow(w)

        # w = np.zeros((120, 120), dtype=np.float32)
        # id_start = np.int(120/2 - fwhm_pix/2)
        # id_end = np.int(120/2 + fwhm_pix/2)
        # w[id_start:id_end, id_start:id_end] = 1

        return w

    def create_window_batch(self, neurons_per_vox):
        """
        create a batch of square window.
        each window with spatial frequency f listed in self.f_batch is repeated
        neurons_per_vox times. Thus, there are len(f_batch) * neurons_per_vox
        windows in total.

        the order of repeatition: [f1 f1 f1...         f2 f2 f2...         f3 f3 f3...]
                                 neurons_per_vox   neurons_per_vox

        Parameters
        ----------
        neurons_per_vox : scalar
            the number of neurons in a voxel.

        Returns
        -------
        w_batch : [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
            a batch of square windows.

        """

        w_batch = np.array(
            [
                self.create_square_window(f)
                for f in self.f_batch
                for n in range(neurons_per_vox)
            ]
        )

        self.w_batch = w_batch
        # return w_batch


class Simulate_CMM_Unit:
    """
    An object class containing the computation of CMM simulation
    """

    def __init__(self, RDS, Window):

        # set rds,  only has 2 disparities: crossed and uncrossed
        # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix] =
        # [n_trial, 2, size_rds_bg_pix, size_rds_bg_pix]
        self.I_L = RDS.L  # left image,
        self.I_R = RDS.R  # right image

        # set rds_u
        self.I_u_L = RDS.u_L
        self.I_u_R = RDS.u_R

        # set rds_bg
        # self.I_bg = RDS.rds_bg

        # set window function or binary mask
        # self.w = Window.w_batch # [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]

        # set rds_type: ards or hmrds or crds
        self.rds_type = RDS.rds_type

        # set center disparity
        # self.disp_ct_deg = RDS.disp_ct_deg
        # self.disp_ct_pix = RDS.disp_ct_pix

        # n_trial
        self.n_batch = RDS.n_batch
        self.n_epoch = RDS.n_epoch
        self.n_trial = RDS.n_trial

    def set_rds(self, RDS_new, rds_type_new):

        # set rds,  only has 2 disparities: crossed and uncrossed
        # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix] =
        # [n_trial, 2, size_rds_bg_pix, size_rds_bg_pix]

        self.I_L = RDS_new.L
        self.I_R = RDS_new.R

        self.rds_type = rds_type_new

    def _compute_monocularResp(self, I_gpu, w_batch_gpu):
        """


        Parameters
        ----------
        I_gpu : [neurons_per_vox*len(f_batch), n_trial, crossed_uncrossed,
                 size_rds_bg_pix, size_rds_bg_pix] tf.array
            RDS image in gpu.
        w_batch_gpu : [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
                        tf.array
            a batch of square windows on gpu.

        Returns
        -------
        monoResp : [neurons_per_vox*len(f_batch), n_trial, crossed_uncrossed,
                 size_rds_bg_pix, size_rds_bg_pix] tf.array

            monocular response

        """
        # def _compute_monocularResp(I_gpu, w_batch_gpu):

        monoResp = tf.einsum("ntdij, nij -> ntdij", I_gpu, w_batch_gpu)

        return monoResp

    # def compute_monocularResp_on_rds_bg(self, I_bg_gpu, w_batch_gpu):

    #     monoResp = tf.einsum("n"

    def _repeat_and_shift_matrix(self, I, f_batch, pixShift_in_vox):
        """
        This module performs 2 operations on the rds matrix: replicate and shift.

        1. Replicate matrix means that the rds matrix is being replicated only for
        faster computation. The amount of replication is: len(f_batch)*neurons_per_vox.

        2. Shift means that the whole pixels in the rds matrix is being shited
        for the the number listed in pixShift_in_vox. The variable pixShift_in_vox
        contains the number of pixels that should be shifted in the rds matrix.
        The values in this variable is associated with the disparity preference
        in the disparity column map.

        The shift operation is
        essentially similar to shift the receptive field to get the binocular
        response. However, the binocular computation here is done by shifting
        the rds matrix, not the RF.
        Shifting the rds matrix is just to follow the definition of correlation
        and matching computation as defined in the paper by
        Doi & Fujita, frontiers in computational neuroscience 2014

        Parameters
        ----------
        I : TYPE
            DESCRIPTION.
        f_batch : TYPE
            DESCRIPTION.
        pixShift_in_vox : TYPE
            DESCRIPTION.

        Returns
        -------
        I_gpu_shift : [len(f_batch)*neurons_per_vox, n_trial, rdsDisp_channels,
                       size_rds_bg_pix, size_rds_bg_pix] tf.array
            DESCRIPTION.

        """

        # def _repeat_and_shift_matrix(I, pixShift_in_vox):

        # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
        n_trial, rdsDisp_channels, size_rds_bg_pix, _ = I.shape
        n_rf = len(f_batch) * len(pixShift_in_vox)

        # transfer to gpu
        # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
        I_gpu = tf.constant(I, dtype=tf.float32)

        ## replicate and shift image
        # [neurons_per_vox, n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
        I_gpu_shift = tf.Variable(
            tf.constant(
                0,
                shape=[
                    n_rf,
                    n_trial,
                    rdsDisp_channels,
                    size_rds_bg_pix,
                    size_rds_bg_pix,
                ],
                dtype=np.float32,
            )
        )

        count = 0
        for f in range(len(f_batch)):
            for n in range(len(pixShift_in_vox)):
                pixShift = pixShift_in_vox[n]

                tf.compat.v1.assign(
                    I_gpu_shift[count], np.roll(I_gpu, shift=pixShift, axis=3)
                )

                count += 1

        return I_gpu_shift

    def _compute_deg2pix(self, deg):
        """
        Convert degree to pixel for the fMRI experimental condition at CiNet
        viewing distanve L = 96 cm
        screen pixel = 1280 x 1024 pixel
        screen size = 35.5 cm
        pix_per_deg ~ 0.02 deg
        cm_per_pix ~ 0.035 cm
        """

        screen_height = 35.5  # screen height (cm)
        cm_per_pix = screen_height / 1024  # screen pixel resolution

        view_dist = 96  # viewing distance (cm)
        h = 2 * view_dist * np.tan((deg / 2) * np.pi / 180)

        pix = (h / cm_per_pix).astype(int)

        return pix

    def simulate_corr_ori_vox(self, RDS, Window, dispPref_in_vox):
        """

        simulate correlation computation as defined in eq. 1 in paper:
            Doi & Fujita, frontiers in computational neuroscience 2014

        The idea is that RDS with crossed- or uncrossed-disparity is shifted
        a number of pixels to the left or right depending on the disparity
        preference contained in a voxel. The pixel shifting is applied to all
        elements in the RDS matrix.

        Parameters
        ----------
        RDS : [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix] object
            DESCRIPTION.
        dispPref_in_vox : [neurons_per_vox] np.array
            DESCRIPTION.

        Returns
        -------
        corr_signal_vox : [neurons_per_vox*len(f_batch), n_trial, crossed_uncrossed] tf.tensor
            DESCRIPTION.

        """

        # [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
        w_gpu = tf.constant(Window.w_batch, dtype=np.float32)

        pixShift_in_vox = self._compute_deg2pix(dispPref_in_vox / 2)

        # [len(Window.f_batch) * len(dispPref_in_vox), n_trial, crossed_uncrossed]
        n_rf = len(Window.f_batch) * len(dispPref_in_vox)
        corr_signal_vox = tf.Variable(
            tf.constant(0, shape=[n_rf, RDS.n_trial, 2], dtype=tf.float32)
        )

        # compute the number of elements in the window function, for each freq
        den = np.reshape(
            np.repeat(np.sum(np.sum(Window.w_batch, axis=2), axis=1), RDS.n_batch * 2),
            (len(Window.f_batch) * len(dispPref_in_vox), RDS.n_batch, 2),
        )
        den_gpu = tf.constant(den, dtype=tf.float32)

        for epoch in range(RDS.n_epoch):

            id_start = epoch * RDS.n_batch
            id_end = id_start + RDS.n_batch

            ## replicate and shift RDS image for neurons_per_vox
            # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_L = RDS.L[id_start:id_end]
            # [neurons_per_vox, n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_L_gpu_shift = self._repeat_and_shift_matrix(
                I_L, Window.f_batch, pixShift_in_vox
            )

            I_R = RDS.R[id_start:id_end]
            I_R_gpu_shift = self._repeat_and_shift_matrix(
                I_R, Window.f_batch, -pixShift_in_vox
            )

            ## compute corr signal
            # [neurons_per_vox*len(f_batch), n_trial, crossed_uncrossed]
            corr_signal = tf.divide(
                tf.einsum(
                    "ntdij, ntdij -> ntd",
                    self._compute_monocularResp(I_L_gpu_shift, w_gpu),
                    self._compute_monocularResp(I_R_gpu_shift, w_gpu),
                ),
                den_gpu,
            )

            tf.compat.v1.assign(corr_signal_vox[:, id_start:id_end], corr_signal)

        # transfer data to cpu
        # corr_ori_vox = match_signal_all.numpy()

        return corr_signal_vox

    def simulate_corr_on_rds_bg(self, RDS, Window, dispPref_in_vox):
        """
        simulate correlation computation on cRDS background (surround cRDS).
        This module is for simulating relative disparity.

        The results are still messy

        Parameters
        ----------
        RDS : TYPE
            DESCRIPTION.
        Window : TYPE
            DESCRIPTION.
        dispPref_in_vox : TYPE
            DESCRIPTION.

        Returns
        -------
        corr_signal_vox : TYPE
            DESCRIPTION.

        """

        n_trial = RDS.n_trial

        # [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
        w_gpu = tf.constant(Window.w_batch, dtype=np.float32)

        # make an array of 0-deg disparity because rds_bg has 0-deg disparity
        pixShift_in_vox = np.zeros(len(dispPref_in_vox), dtype=np.float32)

        # [len(Window.f_batch) * len(dispPref_in_vox), n_trial, crossed_uncrossed]
        n_rf = len(Window.f_batch) * len(dispPref_in_vox)
        corr_signal_vox = tf.Variable(
            tf.constant(0, shape=[n_rf, n_trial, 2], dtype=tf.float32)
        )

        # compute the number of elements in the window function, for each freq
        den = np.reshape(
            np.repeat(np.sum(np.sum(Window.w_batch, axis=2), axis=1), RDS.n_batch * 2),
            (len(Window.f_batch) * len(dispPref_in_vox), RDS.n_batch, 2),
        )
        den_gpu = tf.constant(den, dtype=tf.float32)

        for epoch in range(RDS.n_epoch):

            id_start = epoch * RDS.n_batch
            id_end = id_start + RDS.n_batch

            ## replicate and shift RDS image for neurons_per_vox
            # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_bg = RDS.rds_bg[id_start:id_end]

            # shift 0-deg (or no shift because it's rds background) and replicate rds
            # [neurons_per_vox, n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_bg_gpu = self._repeat_and_shift_matrix(
                I_bg, Window.f_batch, pixShift_in_vox
            )

            ## compute corr signal
            # [neurons_per_vox*len(f_batch), n_trial, crossed_uncrossed]
            corr_signal = tf.divide(
                tf.einsum(
                    "ntdij, ntdij -> ntd",
                    self._compute_monocularResp(I_bg_gpu, w_gpu),
                    self._compute_monocularResp(I_bg_gpu, w_gpu),
                ),
                den_gpu,
            )

            tf.compat.v1.assign(corr_signal_vox[:, id_start:id_end], corr_signal)

        # transfer data to cpu
        # corr_ori_vox = match_signal_all.numpy()

        return corr_signal_vox

    def simulate_match_ori_vox(self, RDS, Window, dispPref_in_vox):
        """
        simulate correlation computation on RDS background that has 0-deg disparity
        magnitude and dot-correlation of 1 (cRDS).

        Parameters
        ----------
        RDS : [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix] object
            DESCRIPTION.
        dispPref_in_vox : [neurons_per_vox] np.array
            DESCRIPTION.

        Returns
        -------
        match_signal_vox : [neurons_per_vox, n_trial, crossed_uncrossed] tf.tensor
            DESCRIPTION.

        """

        # [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
        w_gpu = tf.constant(Window.w_batch, dtype=np.float32)

        pixShift_in_vox = self._compute_deg2pix(dispPref_in_vox / 2)

        # [len(Window.f_batch) * len(dispPref_in_vox), n_trial, crossed_uncrossed]
        n_rf = len(Window.f_batch) * len(dispPref_in_vox)
        match_signal_vox = tf.Variable(
            tf.constant(0, shape=[n_rf, RDS.n_trial, 2], dtype=tf.float32)
        )

        # compute the number of elements in the window function, for each freq
        den = np.reshape(
            np.repeat(np.sum(np.sum(Window.w_batch, axis=2), axis=1), RDS.n_batch * 2),
            (len(Window.f_batch) * len(dispPref_in_vox), RDS.n_batch, 2),
        )
        den_gpu = tf.constant(den, dtype=tf.float32)

        for epoch in range(RDS.n_epoch):

            id_start = epoch * RDS.n_batch
            id_end = id_start + RDS.n_batch

            ## replicate and shift RDS image for neurons_per_vox
            # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_L = RDS.L[id_start:id_end]
            # [n_trial, neurons_per_vox, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_L_gpu_shift = self._repeat_and_shift_matrix(
                I_L, Window.f_batch, pixShift_in_vox
            )

            I_R = RDS.R[id_start:id_end]
            I_R_gpu_shift = self._repeat_and_shift_matrix(
                I_R, Window.f_batch, -pixShift_in_vox
            )

            ## compute match signal
            # [neurons_per_vox, n_trial, crossed_uncrossed]
            match_signal = tf.divide(
                tf.nn.relu(
                    tf.einsum(
                        "ntdij, ntdij -> ntd",
                        self._compute_monocularResp(I_L_gpu_shift, w_gpu),
                        self._compute_monocularResp(I_R_gpu_shift, w_gpu),
                    )
                ),
                den_gpu,
            )

            tf.compat.v1.assign(match_signal_vox[:, id_start:id_end], match_signal)

        # transfer data to cpu
        # match_ori_vox = match_signal_all.numpy()

        return match_signal_vox

    def simulate_match_on_rds_bg(self, RDS, Window, dispPref_in_vox):
        """
        simulate matching computation on RDS background that has 0-deg disparity
        magnitude and dot-correlation of 1 (cRDS).

        simulate matching computation on cRDS background (surround cRDS).
        This module is for simulating relative disparity.

        The results are still messy

        Parameters
        ----------
        RDS : [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix] object
            DESCRIPTION.
        dispPref_in_vox : [neurons_per_vox] np.array
            DESCRIPTION.

        Returns
        -------
        match_signal_vox : [neurons_per_vox, n_trial, crossed_uncrossed] tf.tensor
            DESCRIPTION.

        """

        # [len(f_batch)*neurons_per_vox, size_rds_bg_pix, size_rds_bg_pix]
        w_gpu = tf.constant(Window.w_batch, dtype=np.float32)

        # make an array of 0-deg disparity because rds_bg has 0-deg disparity
        pixShift_in_vox = np.zeros(len(dispPref_in_vox), dtype=np.float32)

        # [len(Window.f_batch) * len(dispPref_in_vox), n_trial, crossed_uncrossed]
        n_rf = len(Window.f_batch) * len(dispPref_in_vox)
        match_signal_vox = tf.Variable(
            tf.constant(0, shape=[n_rf, RDS.n_trial, 2], dtype=tf.float32)
        )

        # compute the number of elements in the window function, for each freq
        den = np.reshape(
            np.repeat(np.sum(np.sum(Window.w_batch, axis=2), axis=1), RDS.n_batch * 2),
            (len(Window.f_batch) * len(dispPref_in_vox), RDS.n_batch, 2),
        )
        den_gpu = tf.constant(den, dtype=tf.float32)

        for epoch in range(RDS.n_epoch):

            id_start = epoch * RDS.n_batch
            id_end = id_start + RDS.n_batch

            ## replicate and shift RDS image for neurons_per_vox
            # [n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_bg = RDS.rds_bg[id_start:id_end]

            # shift 0-deg (or no shift because it's rds background) and replicate rds
            # [neurons_per_vox, n_trial, crossed_uncrossed, size_rds_bg_pix, size_rds_bg_pix]
            I_bg_gpu = self._repeat_and_shift_matrix(
                I_bg, Window.f_batch, pixShift_in_vox
            )

            ## compute match signal
            # [neurons_per_vox, n_trial, crossed_uncrossed]
            match_signal = tf.divide(
                tf.nn.relu(
                    tf.einsum(
                        "ntdij, ntdij -> ntd",
                        self._compute_monocularResp(I_bg_gpu, w_gpu),
                        self._compute_monocularResp(I_bg_gpu, w_gpu),
                    )
                ),
                den_gpu,
            )

            tf.compat.v1.assign(match_signal_vox[:, id_start:id_end], match_signal)

        # transfer data to cpu
        # match_ori_vox = match_signal_all.numpy()

        return match_signal_vox


class Simulate_CMM_on_DispCol(DisparityColumn):
    """

    An object class for simulating CMM on disparity columnar structure and
    compute RDM


    """

    def __init__(self, std_sawtooth_noise, n_bootstrap, nVox):

        super().__init__(std_sawtooth_noise, n_bootstrap, nVox)

        # self.nVox = nVox
        self.ROIs = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "MT"]

        ## load rdm_fmri_all, [len(sbjID_all), n_ROIs, 6, 6]
        # rdm for empirical fmri data for each participant and roi.
        # the following is the way to compute rdm_fmri:
        # mtd = "sqeuclidean" # sqeuclidean gives the most make-sense corr_match weight ratio
        # rsa = RSA()
        # rsa.compute_rdm_all_sbjID(nVox_to_analyze, mtd)
        # rdm_fmri_all = rsa.rdm_fmri_all # [len(sbjID_all), len(ROIs), 6, 6]
        self.rdm_fmri_all = np.load("../../../Data/CMM/rdm_fmri_all_euclidean.npy")

    def simulate_cmm_on_dispCol(
        self, RDS, Window, dispColMap_bootstrap, nVox, i_bootstrap
    ):
        """

        simulate a voxel response based on CMM computed along the disparity column

        Parameters
        ----------
        RDS : TYPE
            DESCRIPTION.
        Window : TYPE
            DESCRIPTION.
        dispColMap_bootstrap : [n_bootstrap, nVox, neurons_per_vox] np.array
            DESCRIPTION.
        nVox : TYPE
            DESCRIPTION.
        i_bootstrap : TYPE
            DESCRIPTION.

        Returns
        -------
        voxResp : [nVox, corr_match, len(f_batch)*neurons_per_vox,
                   n_trial, crossed_uncrossed] np.array,
            a simulated voxel response.

        """

        # total number of RF
        n_rf = len(Window.f_batch) * self.neurons_per_vox

        # [nVox, corr_match, n_rf, n_trial, crossed_uncrossed]
        voxResp = tf.Variable(
            tf.constant(0, shape=[nVox, 2, n_rf, RDS.n_trial, 2], dtype=tf.float32)
        )

        ## define cmm object associated with the given RDS
        cmm = Simulate_CMM_Unit(RDS, Window)
        # vox_step = 4

        for v in range(nVox):
            # for v in range(0, nVox, vox_step):

            print(
                "simulate disparity column for {}, sawtooth_noise_std:{}, bootstrap: {}/{}, vox: {}/{}".format(
                    RDS.rds_type,
                    str(self.std_sawtooth_noise),
                    str(i_bootstrap + 1),
                    str(self.n_bootstrap),
                    str(v + 1),
                    str(self.nVox),
                )
            )

            # get dispPref for all neurons in a voxel
            # dispPref_in_vox = self.dispColMap_bootstrap[i_bootstrap, v]
            dispPref_in_vox = dispColMap_bootstrap[i_bootstrap, v]  # [neurons_per_vox]

            # vox_start = v
            # vox_end = vox_start + vox_step
            # dispPref_in_vox = self.dispColMap_bootstrap[i_bootstrap,
            #                                             vox_start:vox_end].reshape(-1)

            ## define square window associated with dispPref_in_vox
            # window = SquareWindow(f_batch, RDS.size_rds_bg_pix_rds_bg_pix)
            Window.create_window_batch(len(dispPref_in_vox))

            ## simulate cmm corr, [neurons_per_vox*len(f_batch), n_trial, cross_uncrossed]
            vox_resp_corr = cmm.simulate_corr_ori_vox(RDS, Window, dispPref_in_vox)

            ## simulate cmm match, [neurons_per_vox*len(f_batch), n_trial, cross_uncrossed]
            vox_resp_match = cmm.simulate_match_ori_vox(RDS, Window, dispPref_in_vox)

            # transfer to voxResp
            tf.compat.v1.assign(voxResp[v, 0], vox_resp_corr)
            tf.compat.v1.assign(voxResp[v, 1], vox_resp_match)

            # # transfer to voxResp
            # tf.compat.v1.assign(voxResp[vox_start, 0], vox_resp_corr[0:n_rf]) #[rf_nBatch, n_trial, rdsDisp_channels]
            # tf.compat.v1.assign(voxResp[vox_start+1, 0], vox_resp_corr[n_rf:2*n_rf])
            # tf.compat.v1.assign(voxResp[vox_start+2, 0], vox_resp_corr[2*n_rf:3*n_rf])
            # tf.compat.v1.assign(voxResp[vox_start+3, 0], vox_resp_corr[3*n_rf:4*n_rf])

            # tf.compat.v1.assign(voxResp[vox_start, 1], vox_resp_match[0:n_rf])
            # tf.compat.v1.assign(voxResp[vox_start+1, 1], vox_resp_match[n_rf:2*n_rf])
            # tf.compat.v1.assign(voxResp[vox_start+2, 1], vox_resp_match[2*n_rf:3*n_rf])
            # tf.compat.v1.assign(voxResp[vox_start+3, 1], vox_resp_match[3*n_rf:4*n_rf])

        ## transfer data to cpu
        # self.voxResp = voxResp.numpy()

        return voxResp.numpy()

    def _normalize_vox(self, voxResp):
        # def _normalize_vox(voxResp):
        """
        normalize voxel responses

        Parameters
        ----------
        voxResp : [n_bootstrap, nVox, n_trial] np.array
            1D signal.

        Returns
        -------
        y_norm : [n_trial] np.array
            normalized response.

        """

        n_trial = np.shape(voxResp)[2]

        # average across trials
        avg = np.mean(voxResp, axis=2)

        # normalize
        avg_expand = np.repeat(avg[:, :, np.newaxis], n_trial, axis=2)

        y_norm = (voxResp - avg_expand) / avg_expand

        return y_norm

    def compute_rdm_cmm_each_freq(
        self, voxResp_ards, voxResp_hmrds, voxResp_crds, f_batch, nVox_to_analyze, mtd
    ):
        """
        generate rdm for cmm model for each spatial frequency.

        Parameters
        ----------
        voxResp_ards : [n_bootstrap, nVox, corr_match, f_batch_times_neurons_per_vox,
                        n_trial, crossed_uncrossed] np.array
            simulated voxel responses for ards based on cmm without RF.

        voxResp_hmrds : [n_bootstrap, nVox, corr_match, f_batch_times_neurons_per_vox,
                         n_trial, crossed_uncrossed] np.array
            simulated voxel responses for ards based on cmm without RF.

        voxResp_crds : [n_bootstrap, nVox, corr_match, f_batch_times_neurons_per_vox,
                        n_trial, crossed_uncrossed] np.array
            simulated voxel responses for ards based on cmm without RF.

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
        voxResp_ards2 = voxResp_ards[:, 0:nVox_to_analyze]
        voxResp_hmrds2 = voxResp_hmrds[:, 0:nVox_to_analyze]
        voxResp_crds2 = voxResp_crds[:, 0:nVox_to_analyze]

        n_bootstrap, nVox, _, f_batch_times_neurons_per_vox, n_trial, _ = (
            voxResp_ards.shape
        )
        neurons_per_vox = np.int(f_batch_times_neurons_per_vox / len(f_batch))

        ## breakdown into frequency channels, corr-match, crossed-uncrossed
        # [n_bootstrap, nVox_to_analyze, len(f_batch), n_trial]
        corr_ards_crossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        corr_ards_uncrossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        match_ards_crossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        match_ards_uncrossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )

        corr_hmrds_crossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        corr_hmrds_uncrossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        match_hmrds_crossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        match_hmrds_uncrossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )

        corr_crds_crossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        corr_crds_uncrossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        match_crds_crossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )
        match_crds_uncrossed = np.zeros(
            (n_bootstrap, nVox_to_analyze, len(f_batch), n_trial), dtype=np.float32
        )

        for f in range(len(f_batch)):
            #

            f_start = f * neurons_per_vox
            f_end = f_start + neurons_per_vox
            # f_start = f*neurons_per_vox
            # f_end = f_start + neurons_per_vox

            temp = voxResp_ards2[
                :, :, 0, f_start:f_end, :, 0
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_ards_crossed[:, :, f] = voxResp

            temp = voxResp_ards2[
                :, :, 0, f_start:f_end, :, 1
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_ards_uncrossed[:, :, f] = voxResp

            temp = voxResp_ards2[
                :, :, 1, f_start:f_end, :, 0
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_ards_crossed[:, :, f] = voxResp

            temp = voxResp_ards2[
                :, :, 1, f_start:f_end, :, 1
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_ards_uncrossed[:, :, f] = voxResp

            temp = voxResp_hmrds2[
                :, :, 0, f_start:f_end, :, 0
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_hmrds_crossed[:, :, f] = voxResp

            temp = voxResp_hmrds2[
                :, :, 0, f_start:f_end, :, 1
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_hmrds_uncrossed[:, :, f] = voxResp

            temp = voxResp_hmrds2[
                :, :, 1, f_start:f_end, :, 0
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_hmrds_crossed[:, :, f] = voxResp

            temp = voxResp_hmrds2[
                :, :, 1, f_start:f_end, :, 1
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_hmrds_uncrossed[:, :, f] = voxResp

            temp = voxResp_crds2[
                :, :, 0, f_start:f_end, :, 0
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_crds_crossed[:, :, f] = voxResp

            temp = voxResp_crds2[
                :, :, 0, f_start:f_end, :, 1
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            corr_crds_uncrossed[:, :, f] = voxResp

            temp = voxResp_crds2[
                :, :, 1, f_start:f_end, :, 0
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_crds_crossed[:, :, f] = voxResp

            temp = voxResp_crds2[
                :, :, 1, f_start:f_end, :, 1
            ]  # [n_bootstrap, nVox, neurons_per_vox, n_trials]
            # sum neural responses in each voxel
            voxResp = np.sum(temp, axis=2)
            # normalize
            # y_norm = self._normalize_vox(voxResp)
            match_crds_uncrossed[:, :, f] = voxResp

        ## average across n_trial
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

        ## create rdm
        if mtd == "correlation":
            c = 0.5
        else:
            c = 1.0

        rdm_corr = np.zeros((n_bootstrap, len(f_batch), 6, 6), dtype=np.float32)
        rdm_match = np.zeros((n_bootstrap, len(f_batch), 6, 6), dtype=np.float32)

        for i in range(n_bootstrap):
            for f in range(len(f_batch)):

                ## corr RDM
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
                # rdm_corr[i, f] = 1 - cdist(temp, temp, mtd)

                ## match RDM
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
                # rdm_match[i, f] = 1 - cdist(temp, temp, mtd)

        ## average rdm across bootstrap
        # rdm_corr_mean = np.mean(rdm_corr, axis=0)
        # rdm_match_mean = np.mean(rdm_match, axis=0)

        return rdm_corr, rdm_match

    def compute_rdm_cmm_avg_across_freq(
        self, voxResp_ards, voxResp_hmrds, voxResp_crds, nVox_to_analyze, mtd
    ):
        """
        generate rdm for cmm model by taking the sum (avg) of voxel responses
        across all spatial frequency

        Parameters
        ----------
        voxResp_ards : [n_bootstrap, nVox, corr_match,
                        f_batch_times_neurons_per_vox,
                        n_trial, crossed_uncrossed] np.array
            simulated voxel responses for ards based on cmm without RF.

        voxResp_hmrds : [n_bootstrap, nVox, corr_match,
                         f_batch_times_neurons_per_vox,
                         n_trial, crossed_uncrossed] np.array
            simulated voxel responses for ards based on cmm without RF.

        voxResp_crds : [n_bootstrap, nVox, corr_match,
                        f_batch_times_neurons_per_vox,
                        n_trial, crossed_uncrossed] np.array
            simulated voxel responses for ards based on cmm without RF.

        nVox_to_analyze : scalar
            the number of voxels used for the analysis

        Returns
        -------
        rdm_corr_mean : [6, 6]
            rdm_corr that has been averaged across n_bootstrap.
            obtained from compute_rdm_cmm.
        rdm_match_mean : [6, 6]
            rdm_match that has been averaged across n_bootstrap.
            obtained from compute_rdm_cmm.

        """
        voxResp_ards_sum = np.sum(voxResp_ards[:, 0:nVox_to_analyze], axis=3)
        voxResp_hmrds_sum = np.sum(voxResp_hmrds[:, 0:nVox_to_analyze], axis=3)
        voxResp_crds_sum = np.sum(voxResp_crds[:, 0:nVox_to_analyze], axis=3)

        n_bootstrap = voxResp_ards.shape[0]

        corr_ards_crossed = voxResp_ards_sum[
            :, :, 0, :, 0
        ]  # [n_bootstrap, nVox, n_trials]
        corr_ards_uncrossed = voxResp_ards_sum[
            :, :, 0, :, 1
        ]  # [n_bootstrap, nVox, n_trials]
        match_ards_crossed = voxResp_ards_sum[
            :, :, 1, :, 0
        ]  # [n_bootstrap, nVox, n_trials]
        match_ards_uncrossed = voxResp_ards_sum[
            :, :, 1, :, 1
        ]  # [n_bootstrap, nVox, n_trials]
        corr_hmrds_crossed = voxResp_hmrds_sum[
            :, :, 0, :, 0
        ]  # [n_bootstrap, nVox, n_trials]
        corr_hmrds_uncrossed = voxResp_hmrds_sum[
            :, :, 0, :, 1
        ]  # [n_bootstrap, nVox, n_trials]
        match_hmrds_crossed = voxResp_hmrds_sum[
            :, :, 1, :, 0
        ]  # [n_bootstrap, nVox, n_trials]
        match_hmrds_uncrossed = voxResp_hmrds_sum[
            :, :, 1, :, 1
        ]  # [n_bootstrap, nVox, n_trials]
        corr_crds_crossed = voxResp_crds_sum[
            :, :, 0, :, 0
        ]  # [n_bootstrap, nVox, n_trials]
        corr_crds_uncrossed = voxResp_crds_sum[
            :, :, 0, :, 1
        ]  # [n_bootstrap, nVox, n_trials]
        match_crds_crossed = voxResp_crds_sum[
            :, :, 1, :, 0
        ]  # [n_bootstrap, nVox, n_trials]
        match_crds_uncrossed = voxResp_crds_sum[
            :, :, 1, :, 1
        ]  # [n_bootstrap, nVox, n_trials]

        ## average across n_trial
        corr_ards_crossed_mean = np.mean(
            corr_ards_crossed, axis=2
        )  # [n_bootstrap, nVox_to_analyze]
        corr_ards_uncrossed_mean = np.mean(corr_ards_uncrossed, axis=2)
        corr_hmrds_crossed_mean = np.mean(corr_hmrds_crossed, axis=2)
        corr_hmrds_uncrossed_mean = np.mean(corr_hmrds_uncrossed, axis=2)
        corr_crds_crossed_mean = np.mean(corr_crds_crossed, axis=2)
        corr_crds_uncrossed_mean = np.mean(corr_crds_uncrossed, axis=2)

        match_ards_crossed_mean = np.mean(match_ards_crossed, axis=2)
        match_ards_uncrossed_mean = np.mean(match_ards_uncrossed, axis=2)
        match_hmrds_crossed_mean = np.mean(match_hmrds_crossed, axis=2)
        match_hmrds_uncrossed_mean = np.mean(match_hmrds_uncrossed, axis=2)
        match_crds_crossed_mean = np.mean(match_crds_crossed, axis=2)
        match_crds_uncrossed_mean = np.mean(match_crds_uncrossed, axis=2)

        ## create rdm
        if mtd == "correlation":
            c = 0.5
        else:
            c = 1.0

        rdm_corr = np.zeros((n_bootstrap, 6, 6), dtype=np.float32)
        rdm_match = np.zeros((n_bootstrap, 6, 6), dtype=np.float32)

        for i in range(n_bootstrap):
            ## corr RDM
            temp = np.array(
                [
                    corr_ards_crossed_mean[i],
                    corr_ards_uncrossed_mean[i],
                    corr_hmrds_crossed_mean[i],
                    corr_hmrds_uncrossed_mean[i],
                    corr_crds_crossed_mean[i],
                    corr_crds_uncrossed_mean[i],
                ]
            )

            rdm_corr[i] = c * cdist(temp, temp, mtd)
            # rdm_corr[i] = 1 - cdist(temp, temp, mtd)

            ## match RDM
            temp = np.array(
                [
                    match_ards_crossed_mean[i],
                    match_ards_uncrossed_mean[i],
                    match_hmrds_crossed_mean[i],
                    match_hmrds_uncrossed_mean[i],
                    match_crds_crossed_mean[i],
                    match_crds_uncrossed_mean[i],
                ]
            )
            # r = pearsonr
            rdm_match[i] = c * cdist(temp, temp, mtd)
            # rdm_match[i] = 1 - cdist(temp, temp, mtd)

        ## average rdm across bootstrap
        # rdm_corr_mean = np.mean(rdm_corr, axis=0)
        # rdm_match_mean = np.mean(rdm_match, axis=0)

        return rdm_corr, rdm_match

    def compute_weight_cmm(self, rdm_fmri_all, rdm_corr, rdm_match, n_row_for_rdm=6):
        """
        compute the contribution (weight) of correlation and match computations
        in representationsal space.

        the weight of correlation and match is estimated by solving linear equation:
            rdm_fmri = w_corr*rdm_corr + w_match*rdm_match

        the solution is based on

        Parameters
        ----------
        rdm_fmri_all : [sbj, roi, 0:n_row_for_rdm, 0:n_row_for_rdm] np.array
            rdm matrix from empirical fmri data.

        rdm_corr : [n_bootstrap, 6, 6]
            rdm_corr that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        rdm_match : [n_bootstrap, 6, 6]
            rdm_match that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        n_row_for_rdm : scalar
            to get matrix entries above diagonal elements.

        Returns
        -------
        w_cmm : [len(sbjID_all), len(self.ROIs), 3]
            DESCRIPTION.

        """

        # average rdm_corr and rdm_match across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        ## prepare regressor
        # get above diagonal elements
        temp = rdm_corr_mean[0:n_row_for_rdm, 0:n_row_for_rdm]
        x_corr = temp[np.triu_indices(n_row_for_rdm, k=1)]
        temp = rdm_match_mean[0:n_row_for_rdm, 0:n_row_for_rdm]
        x_match = temp[np.triu_indices(n_row_for_rdm, k=1)]

        # normalize by mean-zeroing and dividing max value
        x_corr = x_corr - np.mean(x_corr)
        x_match = x_match - np.mean(x_match)

        x_corr_norm = x_corr / np.max(x_corr)
        x_match_norm = x_match / np.max(x_match)

        x = np.vstack([x_corr_norm, x_match_norm, np.ones(len(x_corr))])

        w_cmm = np.zeros((len(rdm_fmri_all), len(self.ROIs), 3))
        # r2_cmm = np.zeros((len(rdm_fmri_all), len(self.ROIs), 1))
        vif_all = np.zeros((len(rdm_fmri_all), len(self.ROIs), 2))
        # y_true_all = np.zeros((len(rdm_fmri_all),len(self.ROIs), 15))
        # y_pred_all = np.zeros((len(rdm_fmri_all),len(self.ROIs), 15))

        for roi in range(len(self.ROIs)):

            for sbj in range(len(rdm_fmri_all)):
                # temp = rdm_fmri_norm[sbj, 0:n_row_for_rdm, 0:n_row_for_rdm]
                temp = rdm_fmri_all[sbj, roi, 0:n_row_for_rdm, 0:n_row_for_rdm]
                y_sbj = temp[np.triu_indices(n_row_for_rdm, k=1)]

                # zero-mean
                y_sbj = y_sbj - np.mean(y_sbj)

                # max-normalize
                y_sbj = y_sbj / np.max(y_sbj)
                # y_sbj = y_sbj/sem(y_sbj)
                # y_sbj = y_sbj/np.linalg.norm(y_sbj)

                # y_sbj = y[sbj]
                # w = np.matmul(y_sbj, x_inv)

                # using non-negative least square
                w, resid = nnls(x.T, y_sbj)
                w_cmm[sbj, roi] = w

                # compute r2
                # y_pred = np.matmul(w, x)
                # y_true_all[sbj, roi] = y_sbj
                # y_pred_all[sbj, roi] = y_pred
                # r2_cmm[sbj, roi] = r2_score(y_sbj, y_pred)

                # calculate vif
                vif_df = np.column_stack(([x_corr_norm, x_match_norm]))
                vif = [
                    variance_inflation_factor(vif_df, i) for i in range(vif_df.shape[1])
                ]
                vif_all[sbj, roi] = vif

        return w_cmm, vif_all

    def compute_weight_cmm_specific_freq(
        self, rdm_fmri_all, rdm_corr_mean, rdm_match_mean, n_bootstrap, f_batch, freq_id
    ):

        n_sbjID = len(rdm_fmri_all)

        # get rdm_model for specific freq
        rdm_corr_freq = rdm_corr_mean[freq_id]
        rdm_match_freq = rdm_match_mean[freq_id]

        # get above diagonal elements
        rdm_corr_above = rdm_corr_freq[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_freq[np.triu_indices(6, k=1)]

        #  mean-zeroing
        rdm_corr_above = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_match_above = rdm_match_above - np.mean(rdm_match_above)
        # normalize by dividing max value
        rdm_corr_vec = rdm_corr_above / np.max(rdm_corr_above)
        rdm_match_vec = rdm_match_above / np.max(rdm_match_above)

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
        kendall_bootstrap = np.zeros((n_bootstrap, len(self.ROIs), 2), dtype=np.float32)
        w_bootstrap = np.zeros(
            (n_bootstrap, n_sbjID, len(self.ROIs), 3), dtype=np.float32
        )

        for roi in range(len(self.ROIs)):
            for i in range(n_bootstrap):

                print(
                    "compute w_cmm at spatial freq {}, roi: {}, bootstrap: {}/{}".format(
                        str(f_batch[freq_id]),
                        self.ROIs[roi],
                        str(i + 1),
                        str(n_bootstrap),
                    )
                )

                # random sampling rdm_fmri
                id_sample = np.random.randint(n_sbjID, size=n_sbjID)
                rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]

                y_true_sbj = np.zeros((n_sbjID, 15), dtype=np.float32)
                y_pred_sbj = np.zeros((n_sbjID, 15), dtype=np.float32)
                for sbj in range(n_sbjID):

                    rdm_fmri_sbj = rdm_fmri_bootstrap[sbj]

                    # get above diagonal elements
                    rdm_fmri_above = rdm_fmri_sbj[np.triu_indices(6, k=1)]

                    # mean-zeroing
                    rdm_fmri_vec = rdm_fmri_above - np.mean(rdm_fmri_above)

                    # normalize by dividing max valuce
                    rdm_fmri_vec = rdm_fmri_vec / np.max(rdm_fmri_vec)
                    # rdm_fmri_vec = rdm_fmri_vec/sem(rdm_fmri_vec)

                    # start fitting
                    w, resid = nnls(x.T, rdm_fmri_vec)

                    w_bootstrap[i, sbj, roi] = w

                    # normalize w such that w_corr + w_match = 1
                    # w_corr = w[0]
                    # w_match = w[1]
                    # w_corr_norm = w_corr/(w_corr + w_match + 1e-6)
                    # w_match_norm = w_match/(w_corr + w_match + 1e-6)

                    # w_bootstrap[i, sbj, roi, 0] = w_corr_norm
                    # w_bootstrap[i, sbj, roi, 1] = w_match_norm
                    # w_bootstrap[i, sbj, roi, 2] = w[2]

                    # compute r2
                    y_true_sbj[sbj] = rdm_fmri_vec
                    y_pred_sbj[sbj] = np.matmul(w, x)

                # r2_bootstrap[i, sbj, roi] = r2_score(rdm_fmri_vec, y_pred)
                kendall_bootstrap[i, roi] = kendalltau(
                    np.mean(y_true_sbj, axis=0), np.mean(y_pred_sbj, axis=0)
                )

        return w_bootstrap, kendall_bootstrap

    def compute_weight_cmm_bootstrap(
        self, rdm_fmri_all, rdm_corr, rdm_match, n_bootstrap
    ):
        """
        bootstrap the calculation of w_cmm.

        Parameters
        ----------
        rdm_fmri_all : [sbj, roi, 0:n_row_for_rdm, 0:n_row_for_rdm] np.array
            rdm matrix from empirical fmri data.

        rdm_corr : [n_bootstrap, 6, 6]
            rdm_corr that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        rdm_match : [n_bootstrap, 6, 6]
            rdm_match that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        n_bootstrap : scalar
            number of bootsrap iteration.

        Returns
        -------
        w_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 3] np.array
            estimated correlation and match weights.

            w_bootstrap[:, :, :, 0] -> w_corr
            w_bootstrap[:, :, :, 1] -> w_match
            w_bootstrap[:, :, :, 2] -> dc value

        kendall_bootstrap : [n_bootstrap, len(self.ROIs), 2] np.array
            kendall tau statistic.

            kendall_bootstrap[:, :, 0] -> kendalltau correlation
            kendall_bootstrap[:, :, 1] -> p_val

        """

        #

        n_sbjID = len(rdm_fmri_all)

        # average rdm_corr and rdm_match across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        # get above diagonal elements
        rdm_corr_above = rdm_corr_mean[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_mean[np.triu_indices(6, k=1)]

        #  mean-zeroing
        rdm_corr_above = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_match_above = rdm_match_above - np.mean(rdm_match_above)
        # normalize by dividing max value
        rdm_corr_vec = rdm_corr_above / np.max(rdm_corr_above)
        rdm_match_vec = rdm_match_above / np.max(rdm_match_above)

        x = np.vstack([rdm_corr_vec, rdm_match_vec, np.ones(len(rdm_corr_vec))])

        # n_bootstrap = 1000
        # r2_bootstrap = np.zeros((n_bootstrap, n_sbjID, len(self.ROIs)), dtype=np.float32)
        kendall_bootstrap = np.zeros((n_bootstrap, len(self.ROIs), 2), dtype=np.float32)
        r2_adjusted_bootstrap = np.zeros(
            (n_bootstrap, len(self.ROIs)), dtype=np.float32
        )
        w_bootstrap = np.zeros(
            (n_bootstrap, n_sbjID, len(self.ROIs), 3), dtype=np.float32
        )

        for roi in range(len(self.ROIs)):
            for i in range(n_bootstrap):

                print(
                    "compute w_cmm, roi: {}, bootstrap: {}/{}".format(
                        self.ROIs[roi], str(i + 1), str(n_bootstrap)
                    )
                )

                # random sampling rdm_fmri
                id_sample = np.random.randint(n_sbjID, size=n_sbjID)
                rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]

                y_true_sbj = np.zeros((n_sbjID, 15), dtype=np.float32)
                y_pred_sbj = np.zeros((n_sbjID, 15), dtype=np.float32)
                for sbj in range(n_sbjID):

                    rdm_fmri_sbj = rdm_fmri_bootstrap[sbj]

                    # get above diagonal elements
                    rdm_fmri_above = rdm_fmri_sbj[np.triu_indices(6, k=1)]

                    # mean-zeroing
                    rdm_fmri_vec = rdm_fmri_above - np.mean(rdm_fmri_above)

                    # # normalize by dividing max value
                    rdm_fmri_vec = rdm_fmri_vec / np.max(rdm_fmri_vec)
                    # rdm_fmri_vec = rdm_fmri_vec/sem(rdm_fmri_vec)

                    # start fitting
                    w, resid = nnls(x.T, rdm_fmri_vec)

                    w_bootstrap[i, sbj, roi] = w

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
                    y_true_sbj[sbj] = rdm_fmri_vec
                    y_pred_sbj[sbj] = np.matmul(w, x)

                # r2_bootstrap[i, sbj, roi] = r2_score(rdm_fmri_vec, y_pred)
                kendall_bootstrap[i, roi] = kendalltau(
                    np.mean(y_true_sbj, axis=0), np.mean(y_pred_sbj, axis=0)
                )

                # r2 adjusted
                r2 = r2_score(np.mean(y_true_sbj, axis=0), np.mean(y_pred_sbj, axis=0))
                p = 2
                r2_adjusted_bootstrap[i, roi] = 1 - (
                    (1 - r2) * (n_sbjID - 1) / (n_sbjID - p - 1)
                )

        return w_bootstrap, kendall_bootstrap, r2_adjusted_bootstrap


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

        ## compute response average across n_trial
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

        ## normalize the distribution of voxel responses so that all voxResp above
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

        ## compute response average across n_trial
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

        ## compute signal to noise for correlation computation n for each voxel
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

        y2_corr = y[:, :, 0:nVox_to_analyze] ** 2  # square to avoid negative values

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

        ## compute signal to noise for matching computation n for each voxel
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

        y2_match = y[:, :, 0:nVox_to_analyze] ** 2  # square to avoid negative values

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

        ## compute s2n for cmm signal
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

            ## normalize voxel values inside ROI
            vtc_roi = vtc_labeled.loc[vtc_labeled.roi == roi]
            # average in each run
            avg = vtc_roi.groupby(["roi", "run", "vox"])["vtc_value"].transform("mean")
            # normalize
            temp = (vtc_roi.vtc_value - avg) / avg * 100
            vtc_roi = vtc_roi.assign(vtc_norm=temp)

            ## check if nVox_to_analyze < nVox_max in this ROI
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
                y_sort = y_avg[:, id_sort]  # [nConds, nVox], including fixation

                # select nVox_to_analyze voxels
                y_sel = y_sort[
                    :, 0:nVox_to_analyze
                ]  # [nConds, nVox], including fixation

                # get fixation responses
                y_fix = np.tile(
                    y_sel[0], (nConds, 1)
                )  # [nConds, nVox], excluding fixation

                # compute the response difference between stimulus and fixation
                y_diff = y_sel[1:] - y_fix  # [nConds, nVox], excluding fixation

                ## compute standard deviation of response distribution for each
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
                s2n_all_roi[roi, run] = np.mean(
                    s2n, axis=1
                )  # [nROIs, nRuns, nConds], exclude fixation

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

            s2n_all_vox[v] = s2n_vox  # [len(nVox_list), n_sbjID, nROIs, nConds]

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
        voxResp_ards_corr_crossed = voxResp_corr_norm[
            :, 0
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_corr_uncrossed = voxResp_corr_norm[
            :, 1
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_match_crossed = voxResp_match_norm[
            :, 0
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_ards_match_uncrossed = voxResp_match_norm[
            :, 1
        ]  # [n_bootstrap, nVox_to_analyzed]

        voxResp_hmrds_corr_crossed = voxResp_corr_norm[
            :, 2
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_corr_uncrossed = voxResp_corr_norm[
            :, 3
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_match_crossed = voxResp_match_norm[
            :, 2
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_hmrds_match_uncrossed = voxResp_match_norm[
            :, 3
        ]  # [n_bootstrap, nVox_to_analyzed]

        voxResp_crds_corr_crossed = voxResp_corr_norm[
            :, 4
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_corr_uncrossed = voxResp_corr_norm[
            :, 5
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_match_crossed = voxResp_match_norm[
            :, 4
        ]  # [n_bootstrap, nVox_to_analyzed]
        voxResp_crds_match_uncrossed = voxResp_match_norm[
            :, 5
        ]  # [n_bootstrap, nVox_to_analyzed]

        ## compute signal to noise for correlation computation n for each voxel
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

        ## compute signal to noise for matching computation n for each voxel
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

        ## compute s2n for cmm signal
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

                    voxResp = voxResp_fmri_norm_all[sbj][
                        roi, run
                    ]  # [nConds, nVox_to_analyze]

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

    def __init__(self):

        super().__init__()

        self.dpi = 600

        plt.style.use("seaborn-colorblind")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

    def plotHeat_rdm_cmm(
        self, rdm_corr, rdm_match, f_batch, mtd, sawtooth_noise_std, save_flag
    ):
        """
        plot heatmap rdm_corr and rdm_match for each spatial frequency.

        each entry of rdm is divided by its max value such that it is normalized
        to 1. The range is between 0 and 1.

        Parameters
        ----------
        rdm_corr : [n_bootstrap, len(f_batch), 6, 6]
            rdm_corr for each spatial freq.
            obtained from compute_rdm_cmm_each_freq.

        rdm_match : [n_bootstrap, len(f_batch), 6, 6]
            rdm_match for each spatial freq
            obtained from compute_rdm_cmm_each_freq.

        f_batch : for example: np.array([1, 2, 4, 8, 16]).astype(np.float32) # spatial frequency
            list of spatial frequency.

        Returns
        -------
        None.

        """

        ## average rdm_corr and rdm_match across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        ## plot heatmap
        conds = [
            "aRDS_cross",
            "aRDS_uncross",
            "hmRDS_cross",
            "hmRDS_uncross",
            "cRDS_cross",
            "cRDS_uncross",
        ]

        sns.set()
        sns.set(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (8, 3.5)
        n_row = 1
        n_col = 2

        cmap = "jet"
        for f in range(len(f_batch)):

            ## calculate pearson-correlation between rdm_corr and rdm_match
            temp_corr = rdm_corr_mean[f]
            # get above diagonal
            rdm_corr_above = temp_corr[np.triu_indices(6, k=1)]

            ## mean-zeroing and max-normalize
            rdm_corr_mean_zero = rdm_corr_above - np.mean(rdm_corr_above)
            rdm_corr_norm = rdm_corr_mean_zero / np.max(rdm_corr_mean_zero)

            temp_match = rdm_match_mean[f]
            # get above diagonal
            rdm_match_above = temp_match[np.triu_indices(6, k=1)]

            ## mean-zeroing and max-normalize
            rdm_match_mean_zero = rdm_match_above - np.mean(rdm_match_above)
            rdm_match_norm = rdm_match_mean_zero / np.max(rdm_match_mean_zero)

            # calculate pearson coef
            rdm_r = np.corrcoef(rdm_corr_norm, rdm_match_norm)[0, 1]
            # rdm_r = cdist(temp_corr, temp_match)

            ## reconstruct rdm
            rdm_corr_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_corr_reconstruct[np.triu_indices(6, k=1)] = rdm_corr_norm / 2 + 0.5
            # copy upper to lower triangle
            i_lower = np.tril_indices(6, k=-1)
            rdm_corr_reconstruct[i_lower] = rdm_corr_reconstruct.T[i_lower]

            rdm_match_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_match_reconstruct[np.triu_indices(6, k=1)] = rdm_match_norm / 2 + 0.5
            # copy upper to lower triangle
            rdm_match_reconstruct[i_lower] = rdm_match_reconstruct.T[i_lower]

            fig, axes = plt.subplots(
                nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
            )

            fig.text(
                0.5,
                1.1,
                "RDM_Correlation VS RDM_Match f={}, {}".format(str(f_batch[f]), mtd),
                ha="center",
            )
            fig.text(-0.2, 0.5, "Conditions", va="center", rotation=90)
            fig.text(0.5, -0.6, "Conditions", ha="center")
            fig.text(
                0.5, -0.5, "pearson_r = {}".format(str(np.round(rdm_r, 2))), ha="center"
            )

            fig.tight_layout()

            plt.subplots_adjust(wspace=0.2, hspace=0.3)

            # estimate v_min and v_max for cbar
            v_min = np.round(np.min(rdm_corr_reconstruct), 2)
            v_max = np.round(np.max(rdm_corr_reconstruct), 2)
            # v_max = 1.0

            sns.heatmap(
                rdm_corr_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                xticklabels=conds,
                yticklabels=conds,
                ax=axes[0],
            )
            axes[0].set_title("Correlation computation", pad=20)

            sns.heatmap(
                rdm_match_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                xticklabels=conds,
                yticklabels=conds,
                ax=axes[1],
            )
            axes[1].set_title("Match computation", pad=20)

            if save_flag == 1:
                fig.savefig(
                    "../../../Plots/CMM/RDM/Noise{}/PlotHeat_cmm_noRF_rdm_f_{}_{}.pdf".format(
                        str(sawtooth_noise_std), str(f_batch[f]), mtd
                    ),
                    dpi=self.dpi,
                    bbox_inches="tight",
                )

    def plotHeat_rdm_cmm_without_normalization(
        self, rdm_corr, rdm_match, f_batch, mtd, sawtooth_noise_std, save_flag
    ):
        """
        plot heatmap rdm_corr and rdm_match without normalization
        for each spatial frequency.

        Without normalization here means that the rdm is max-normalize to rdm_corr,
        meaning that each rdm entry is divided by max(rdm_corr). The range is
        between 0 and 1.


        Parameters
        ----------
        rdm_corr : [n_bootstrap, len(f_batch), 6, 6]
            rdm_corr for each spatial freq.
            obtained from compute_rdm_cmm_each_freq.

        rdm_match : [n_bootstrap, len(f_batch), 6, 6]
            rdm_match for each spatial freq
            obtained from compute_rdm_cmm_each_freq.

        f_batch : for example: np.array([1, 2, 4, 8, 16]).astype(np.float32) # spatial frequency
            list of spatial frequency.

        Returns
        -------
        None.

        """

        ## average across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        ## plot heatmap
        conds = [
            "aRDS_cross",
            "aRDS_uncross",
            "hmRDS_cross",
            "hmRDS_uncross",
            "cRDS_cross",
            "cRDS_uncross",
        ]

        sns.set()
        sns.set(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (8, 3.5)
        n_row = 1
        n_col = 2

        cmap = "jet"
        for f in range(len(f_batch)):

            ## calculate pearson-correlation between rdm_corr and rdm_match
            temp_corr = rdm_corr_mean[f]
            # get above diagonal
            rdm_corr_above = temp_corr[np.triu_indices(6, k=1)]
            # mean-zeroing and max-normalize
            rdm_corr_mean_zero = rdm_corr_above - np.mean(rdm_corr_above)
            rdm_corr_norm = rdm_corr_mean_zero / np.max(rdm_corr_mean_zero)

            temp_match = rdm_match_mean[f]
            # get above diagonal
            rdm_match_above = temp_match[np.triu_indices(6, k=1)]
            # mean-zeroing and max-normalize
            rdm_match_mean_zero = rdm_match_above - np.mean(rdm_match_above)
            rdm_match_norm = rdm_match_mean_zero / np.max(rdm_corr_mean_zero)

            # calculate pearson coef
            rdm_r = np.corrcoef(rdm_corr_norm, rdm_match_norm)[0, 1]
            # rdm_r = cdist(temp_corr, temp_match)

            ## reconstruct rdm
            rdm_corr_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_corr_reconstruct[np.triu_indices(6, k=1)] = rdm_corr_norm / 2 + 0.5
            # copy upper to lower triangle
            i_lower = np.tril_indices(6, k=-1)
            rdm_corr_reconstruct[i_lower] = rdm_corr_reconstruct.T[i_lower]

            rdm_match_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_match_reconstruct[np.triu_indices(6, k=1)] = rdm_match_norm / 2 + 0.5
            # copy upper to lower triangle
            rdm_match_reconstruct[i_lower] = rdm_match_reconstruct.T[i_lower]

            fig, axes = plt.subplots(
                nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
            )

            fig.text(
                0.5,
                1.1,
                "RDM_Correlation VS RDM_Match f={}, {}".format(str(f_batch[f]), mtd),
                ha="center",
            )
            fig.text(-0.2, 0.5, "Conditions", va="center", rotation=90)
            fig.text(0.5, -0.6, "Conditions", ha="center")
            fig.text(
                0.5, -0.5, "pearson_r = {}".format(str(np.round(rdm_r, 2))), ha="center"
            )

            fig.tight_layout()

            plt.subplots_adjust(wspace=0.2, hspace=0.3)

            # estimate v_min and v_max for cbar
            v_min = np.round(np.min(rdm_corr_reconstruct), 2)
            v_max = np.round(np.max(rdm_corr_reconstruct), 2)
            # v_max = 1.0

            sns.heatmap(
                rdm_corr_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                xticklabels=conds,
                yticklabels=conds,
                ax=axes[0],
            )
            axes[0].set_title("Correlation computation", pad=20)

            sns.heatmap(
                rdm_match_reconstruct,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                xticklabels=conds,
                yticklabels=conds,
                ax=axes[1],
            )
            axes[1].set_title("Match computation", pad=20)

            if save_flag == 1:
                fig.savefig(
                    "../../../Plots/CMM/RDM/Noise{}/PlotHeat_cmm_noRF_rdm_f_{}_{}_without_normalization.pdf".format(
                        str(sawtooth_noise_std), str(f_batch[f]), mtd
                    ),
                    dpi=self.dpi,
                    bbox_inches="tight",
                )

    def plotHeat_rdm_cmm_avg(
        self, rdm_corr, rdm_match, mtd, sawtooth_noise_std, save_flag
    ):
        """
        plot heatmap rdm_corr and rdm_match that have been averaged across
        spatial frequency

        Parameters
        ----------
        rdm_corr : [n_bootstrap, 6, 6]
            rdm_corr that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        rdm_match : [n_bootstrap, 6, 6]
            rdm_match that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

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

        # average across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        # get above diagonal elements
        rdm_corr_above = rdm_corr_mean[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_mean[np.triu_indices(6, k=1)]

        ## mean-zeroing and max normalize
        rdm_corr_mean_zero = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_corr_norm = rdm_corr_mean_zero / np.max(rdm_corr_mean_zero)
        rdm_match_mean_zero = rdm_match_above - np.mean(rdm_match_above)
        rdm_match_norm = rdm_match_mean_zero / np.max(rdm_match_mean_zero)

        ## calculate pearson-correlation between rdm_corr and rdm_match
        rdm_r = np.corrcoef(rdm_corr_above, rdm_match_above)[0, 1]
        # rdm_r = cdist(rdm_corr_mean, rdm_match_mean)

        ## reconstruct rdm
        rdm_corr_reconstruct = np.zeros((6, 6), dtype=np.float32)
        rdm_corr_reconstruct[np.triu_indices(6, k=1)] = rdm_corr_norm / 2 + 0.5
        # copy upper to lower triangle
        i_lower = np.tril_indices(6, k=-1)
        rdm_corr_reconstruct[i_lower] = rdm_corr_reconstruct.T[i_lower]

        rdm_match_reconstruct = np.zeros((6, 6), dtype=np.float32)
        rdm_match_reconstruct[np.triu_indices(6, k=1)] = rdm_match_norm / 2 + 0.5
        # copy upper to lower triangle
        rdm_match_reconstruct[i_lower] = rdm_match_reconstruct.T[i_lower]

        conds = [
            "aRDS_cross",
            "aRDS_uncross",
            "hmRDS_cross",
            "hmRDS_uncross",
            "cRDS_cross",
            "cRDS_uncross",
        ]

        sns.set()
        sns.set(context="paper", style="white", font_scale=2, palette="deep")

        # estimate v_min and v_max for cbar
        v_min = np.round(np.min(rdm_corr_reconstruct), 2)
        v_max = np.round(np.max(rdm_corr_reconstruct), 2)
        # v_max = 1.0

        figsize = (8, 3.5)
        n_row = 1
        n_col = 2

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.1,
            "RDM_Correlation VS RDM_Match, avg across freq,\nmtd:{}, noise{}:".format(
                mtd, str(sawtooth_noise_std)
            ),
            ha="center",
        )
        fig.text(-0.2, 0.5, "Conditions", va="center", rotation=90)
        fig.text(0.5, -0.6, "Conditions", ha="center")
        fig.text(
            0.5, -0.5, "pearson_r = {}".format(str(np.round(rdm_r, 2))), ha="center"
        )

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        cmap = "jet"
        sns.heatmap(
            rdm_corr_reconstruct,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=conds,
            yticklabels=conds,
            ax=axes[0],
        )
        axes[0].set_title("Correlation computation", pad=20)

        sns.heatmap(
            rdm_match_reconstruct,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=conds,
            yticklabels=conds,
            ax=axes[1],
        )
        axes[1].set_title("Match computation", pad=20)

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/RDM/Noise{}/PlotHeat_cmm_noRF_rdm_avg_{}.pdf".format(
                    str(sawtooth_noise_std), mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotHeat_rdm_cmm_avg_without_normalization(
        self, rdm_corr, rdm_match, mtd, sawtooth_noise_std, save_flag
    ):
        """
        plot heatmap rdm_corr and rdm_match that have been averaged across
        spatial frequency

        Parameters
        ----------
        rdm_corr : [n_bootstrap, 6, 6]
            rdm_corr that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        rdm_match : [n_bootstrap, 6, 6]
            rdm_match that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

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

        # average across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        # if len(rdm_corr_mean.shape)>2:
        #     # plot the average of rdm_cmm across spatial frequency channel
        #     rdm_corr_mean_freq = np.mean(rdm_corr_mean, axis=0)
        #     rdm_match_mean_freq = np.mean(rdm_match_mean, axis=0)

        # else:
        rdm_corr_mean_freq = rdm_corr_mean.copy()
        rdm_match_mean_freq = rdm_match_mean.copy()

        # get above diagonal elements
        rdm_corr_above = rdm_corr_mean_freq[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_mean_freq[np.triu_indices(6, k=1)]

        ## mean-zeroing and max normalize
        # den = np.max([rdm_corr_above, rdm_match_above])
        rdm_corr_mean_zero = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_corr_norm = rdm_corr_mean_zero / np.max(rdm_corr_mean_zero)
        rdm_match_mean_zero = rdm_match_above - np.mean(rdm_match_above)
        rdm_match_norm = rdm_match_mean_zero / np.max(rdm_corr_mean_zero)

        ## calculate pearson-correlation between rdm_corr and rdm_match
        rdm_r = np.corrcoef(rdm_corr_above, rdm_match_above)[0, 1]
        # rdm_r = cdist(rdm_corr_mean_freq, rdm_match_mean_freq)

        ## reconstruc rdm
        rdm_corr_reconstruct = np.zeros((6, 6), dtype=np.float32)
        rdm_corr_reconstruct[np.triu_indices(6, k=1)] = rdm_corr_norm / 2 + 0.5
        # copy upper to lower triangle
        i_lower = np.tril_indices(6, k=-1)
        rdm_corr_reconstruct[i_lower] = rdm_corr_reconstruct.T[i_lower]

        rdm_match_reconstruct = np.zeros((6, 6), dtype=np.float32)
        rdm_match_reconstruct[np.triu_indices(6, k=1)] = rdm_match_norm / 2 + 0.5
        # copy upper to lower triangle
        rdm_match_reconstruct[i_lower] = rdm_match_reconstruct.T[i_lower]

        conds = [
            "aRDS_cross",
            "aRDS_uncross",
            "hmRDS_cross",
            "hmRDS_uncross",
            "cRDS_cross",
            "cRDS_uncross",
        ]

        sns.set()
        sns.set(context="paper", style="white", font_scale=2, palette="deep")

        # estimate v_min and v_max for cbar
        # v_min = 0.0
        v_min = np.round(np.min(rdm_corr_reconstruct), 2)
        v_max = np.round(np.max(rdm_corr_reconstruct), 2)
        # v_max = 1.0

        figsize = (8, 3.5)
        n_row = 1
        n_col = 2

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.1,
            "RDM_Correlation VS RDM_Match (noRF), avg across freq, {}".format(mtd),
            ha="center",
        )
        fig.text(-0.2, 0.5, "Conditions", va="center", rotation=90)
        fig.text(0.5, -0.6, "Conditions", ha="center")
        fig.text(
            0.5, -0.5, "pearson_r = {}".format(str(np.round(rdm_r, 2))), ha="center"
        )

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        cmap = "jet"

        # c = 1
        sns.heatmap(
            rdm_corr_reconstruct,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=conds,
            yticklabels=conds,
            ax=axes[0],
        )
        axes[0].set_title("Correlation computation", pad=20)

        sns.heatmap(
            rdm_match_reconstruct,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=conds,
            yticklabels=conds,
            ax=axes[1],
        )
        axes[1].set_title("Match computation", pad=20)

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/RDM/Noise{}/PlotHeat_cmm_noRF_rdm_avg_{}_without_normalization.pdf".format(
                    str(sawtooth_noise_std), mtd
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotHeat_rdm_fit(
        self,
        rdm_fmri_all,
        rdm_corr,
        rdm_match,
        w_cmm_bootstrap,
        mtd,
        sawtooth_noise_std,
        save_flag,
    ):
        """


        Parameters
        ----------
        rdm_fmri_all : TYPE
            DESCRIPTION.

        rdm_corr : [n_bootstrap, 6, 6]
            rdm_corr that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        rdm_match : [n_bootstrap, 6, 6]
            rdm_match that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        w_cmm_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 3] np.array
            estimated correlation and match weights.

            w_cmm_bootstrap[:, :, :, 0] -> w_corr
            w_cmm_bootstrap[:, :, :, 1] -> w_match
            w_cmm_bootstrap[:, :, :, 2] -> dc value

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

        # average w_cmm_bootstrap across sbjID
        w_avg = np.mean(w_cmm_bootstrap, axis=1)  # [n_bootstrap, nROIs, 3]

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        w_corr_norm = w_avg[:, :, 0] / (w_avg[:, :, 0] + w_avg[:, :, 1] + tol)
        w_match_norm = w_avg[:, :, 1] / (w_avg[:, :, 0] + w_avg[:, :, 1] + tol)

        # average across n_bootstrap
        w_corr_norm = np.mean(w_corr_norm, axis=0)
        w_match_norm = np.mean(w_match_norm, axis=0)

        # average rdm_fmri across sbjID
        rdm_fmri_mean = np.mean(rdm_fmri_all, axis=0)

        # average rdm_corr and rdm_match across bootstrap
        rdm_corr_mean = np.mean(rdm_corr, axis=0)
        rdm_match_mean = np.mean(rdm_match, axis=0)

        # get above diagonal
        rdm_corr_above = rdm_corr_mean[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_mean[np.triu_indices(6, k=1)]

        # mean-zeroing and max-normalize
        rdm_corr_norm = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_corr_norm = rdm_corr_norm / np.max(rdm_corr_norm)
        rdm_match_norm = rdm_match_above - np.mean(rdm_match_above)
        rdm_match_norm = rdm_match_norm / np.max(rdm_match_norm)

        # start plotting
        sns.set()
        sns.set(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 20)
        n_row = 8
        n_col = 3

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(0.5, 1.03, "RDM_fMRI and RDM_model", ha="center")

        fig.text(-0.05, 0.5, "ROI", va="center", rotation=90)
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

            rdm_fmri_roi = rdm_fmri_mean[roi]

            # get above diagonal
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing and max-normalize
            rdm_fmri_norm = rdm_fmri_above - np.mean(rdm_fmri_above)
            rdm_fmri_norm = rdm_fmri_norm / np.max(rdm_fmri_norm)

            # make rdm_model
            rdm_corr_roi = w_corr_norm[roi] * rdm_corr_norm
            rdm_match_roi = w_match_norm[roi] * rdm_match_norm
            rdm_fit = rdm_corr_roi + rdm_match_roi

            # calculate kendalltau
            kendall = kendalltau(rdm_fmri_norm, rdm_fit)

            ## reconstruct rdm
            rdm_fmri_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_fmri_reconstruct[np.triu_indices(6, k=1)] = rdm_fmri_norm / 2 + 0.5
            # copy upper to lower triangle
            i_lower = np.tril_indices(6, k=-1)
            rdm_fmri_reconstruct[i_lower] = rdm_fmri_reconstruct.T[i_lower]

            rdm_fit_reconstruct = np.zeros((6, 6), dtype=np.float32)
            rdm_fit_reconstruct[np.triu_indices(6, k=1)] = rdm_fit / 2 + 0.5
            # copy upper to lower triangle
            rdm_fit_reconstruct[i_lower] = rdm_fit_reconstruct.T[i_lower]

            ## calculate rdm_residual
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
                "../../../Plots/CMM/RDM/Noise{}/PlotHeat_cmm_noRF_rdm_fit_{}_noise{}.pdf".format(
                    str(sawtooth_noise_std), mtd, str(sawtooth_noise_std)
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotScatter_w_corr_vs_w_match(
        self, w_cmm_bootstrap, nVox_to_analyze, sawtooth_noise_std, save_flag
    ):
        """


        Parameters
        ----------
        w_cmm_bootstrap : [n_bootstrap, len(sbjID), nROIs, 3] np.array
            w_cmm_bootstrap.

            w_cmm[:, :, :, 0] -> w_corr
            w_cmm[:, :, :, 1] -> w_match
            w_cmm[:, :, :, 2] -> dc value

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

        ## average w_cmm_bootstrap across sbjID
        w_cmm = np.mean(w_cmm_bootstrap, axis=1)

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        w_corr_norm = w_cmm[:, :, 0] / (w_cmm[:, :, 0] + w_cmm[:, :, 1] + tol)
        w_match_norm = w_cmm[:, :, 1] / (w_cmm[:, :, 0] + w_cmm[:, :, 1] + tol)

        sns.set()
        sns.set(context="paper", style="white", font_scale=4, palette="deep")

        # x_low = 0.05
        # x_up = 0.26
        # x_step = 0.05
        # y_low = 0.05
        # y_up = 0.31
        # y_step = 0.05

        x_low = 0.1
        x_up = 0.83
        x_step = 0.1
        y_low = 0.1
        y_up = 0.83
        y_step = 0.1

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        figsize = (10, 10)
        n_row = 1
        n_col = 1

        ## plot normalized weight
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.04,
            "w_corr VS w_match (sum-normalized to 1)\n #Voxels={}".format(
                str(nVox_to_analyze)
            ),
            ha="center",
        )
        fig.text(-0.03, 0.5, "w_corr", va="center", rotation=90)
        fig.text(0.5, -0.03, "w_match", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot combined frequency
        for roi in range(len(self.ROIs)):

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

        axes.plot([x_low, x_up], [x_low, x_up], "r--", linewidth=4)

        axes.set_title("Combined freq", pad=25)

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        axes.text(
            0.3,
            0.19,
            "Cross-correlation",
            color="gray",
            weight="bold",
            alpha=0.75,
            ha="center",
            rotation=45,
        )
        axes.text(
            0.3,
            0.11,
            "Cross-matching",
            color="gray",
            weight="bold",
            alpha=0.75,
            ha="center",
            rotation=45,
        )

        # remove top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        ## save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM/Weight_RDM_fitting/Noise{}/PlotScatter_w_corr_vs_w_match_bootstrap_nVox_{}_noise{}_weightNormalized.pdf".format(
                    str(sawtooth_noise_std),
                    str(nVox_to_analyze),
                    str(sawtooth_noise_std),
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

        ## plot non-normalized weight
        x_low = 0.0
        x_up = 0.35
        x_step = 0.1
        y_low = 0.0
        y_up = 0.35
        y_step = 0.1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.04,
            "w_corr VS w_match (non-normalized)\n #Voxels={}".format(
                str(nVox_to_analyze)
            ),
            ha="center",
        )
        fig.text(-0.03, 0.5, "w_corr", va="center", rotation=90)
        fig.text(0.5, -0.03, "w_match", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot combined frequency
        for roi in range(len(self.ROIs)):

            x = np.mean(w_cmm[:, roi, 1])
            y = np.mean(w_cmm[:, roi, 0])
            x_err = np.std(w_cmm[:, roi, 1])
            y_err = np.std(w_cmm[:, roi, 0])

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

        axes.plot([x_low, x_up], [x_low, x_up], "r--", linewidth=4)

        axes.set_title("Combined freq", pad=20)

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        axes.text(
            0.1,
            0.05,
            "Cross-correlation",
            color="gray",
            weight="bold",
            alpha=0.75,
            ha="center",
            rotation=45,
        )
        axes.text(
            0.1,
            0.01,
            "Cross-matching",
            color="gray",
            weight="bold",
            alpha=0.75,
            ha="center",
            rotation=45,
        )

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM/Weight_RDM_fitting/Noise{}/PlotScatter_w_corr_vs_w_match_bootstrap_nVox_{}_noise{}.pdf".format(
                    str(sawtooth_noise_std),
                    str(nVox_to_analyze),
                    str(sawtooth_noise_std),
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBar_w_cmm_ratio(
        self, w_cmm_bootstrap, nVox_to_analyze, sawtooth_noise_std, save_flag
    ):
        """

        Parameters
        ----------
        w_cmm_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 3] np.array
            w_cmm_bootstrap.

            w_cmm_bootstrap[:, :, :, 0] -> w_corr
            w_cmm_bootstrap[:, :, :, 1] -> w_match
            w_cmm_bootstrap[:, :, :, 2] -> dc value

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

        ## average w_cmm_bootstrap across sbjID
        w_cmm_bootstrap_avg = np.mean(w_cmm_bootstrap, axis=1)

        # normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        den = w_cmm_bootstrap_avg[:, :, 0] + w_cmm_bootstrap_avg[:, :, 1] + tol
        num = w_cmm_bootstrap_avg[:, :, 0]

        w_cmm_ratio = num / den
        # w_cmm_ratio = num
        y = np.mean(w_cmm_ratio, axis=0)
        y_err = np.std(
            w_cmm_ratio, axis=0
        )  # use standard deviation because it uses bootstrap

        sns.set()
        sns.set(context="paper", style="white", font_scale=4, palette="deep")

        pos = list(np.arange(0, 24, 3))
        error_kw = dict(lw=3, capsize=7, capthick=3)

        plt.figure(figsize=(14, 10))
        plt.bar(pos, y, yerr=y_err, width=2.5, color="gray", error_kw=error_kw)

        # plot line
        plt.plot([-2, 23], [0.5, 0.5], "r--", linewidth=3)

        plt.xticks(pos, self.ROIs)
        # plt.xlabel("ROI")
        plt.ylabel("ratio")
        plt.title("w_corr/(w_corr+w_match)", pad=20)

        y_low = 0.0
        y_up = 0.76
        y_step = 0.25
        plt.ylim(y_low, y_up, y_step)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        ## save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM/Weight_RDM_fitting/Noise{}/PlotBar_w_cmm_ratio_nVox_{}_noise{}.pdf".format(
                    str(sawtooth_noise_std),
                    str(nVox_to_analyze),
                    str(sawtooth_noise_std),
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBar_goodness_of_fit(
        self,
        rdm_fmri_all,
        rdm_corr,
        rdm_match,
        w_cmm_bootstrap,
        kendalltau_low,
        kendalltau_up,
        nVox_to_analyze,
        sawtooth_noise_std,
        save_flag,
    ):
        """


        Parameters
        ----------
        rdm_fmri_all : [n_sbjID, n_ROIs, 6, 6] np.array
            DESCRIPTION.

        rdm_corr : [n_bootstrap, 6, 6]
            rdm_corr that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        rdm_match : [n_bootstrap, 6, 6]
            rdm_match that has been averaged across spatial frequency.
            obtained from compute_rdm_cmm_avg_across_freq.

        w_cmm_bootstrap : [n_bootstrap, len(sbjID_all), len(self.ROIs), 3] np.array
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

        n_sbj, n_roi, _, _ = rdm_fmri_all.shape

        # average w_cmm_bootstrap across sbjID
        w_avg = np.mean(w_cmm_bootstrap, axis=1)  # [n_bootstrap, nROIs, 3]

        ## normalize weight such that w_corr + w_match = 1
        tol = 1e-6
        w_corr_norm = w_avg[:, :, 0] / (w_avg[:, :, 0] + w_avg[:, :, 1] + tol)
        w_match_norm = w_avg[:, :, 1] / (w_avg[:, :, 0] + w_avg[:, :, 1] + tol)

        # average across n_bootstrap
        w_corr_norm = np.mean(w_corr_norm, axis=0)
        w_match_norm = np.mean(w_match_norm, axis=0)

        ## average rdm_fmri_all across sbjID
        rdm_fmri_mean = np.mean(rdm_fmri_all, axis=0)

        # average rdm_corr and rdm_match across bootstrap
        rdm_corr_avg = np.mean(rdm_corr, axis=0)
        rdm_match_avg = np.mean(rdm_match, axis=0)

        # get above diagonal element
        rdm_corr_above = rdm_corr_avg[np.triu_indices(6, k=1)]
        rdm_match_above = rdm_match_avg[np.triu_indices(6, k=1)]

        # mean-zeroing and max-normalize
        rdm_corr_norm = rdm_corr_above - np.mean(rdm_corr_above)
        rdm_corr_norm = rdm_corr_norm / np.max(rdm_corr_norm)
        rdm_match_norm = rdm_match_above - np.mean(rdm_match_above)
        rdm_match_norm = rdm_match_norm / np.max(rdm_match_norm)

        # average w_bootstrap across sbjID and n_bootstrap
        # w_bootstrap_fin = np.mean(np.mean(w_bootstrap, axis=1),
        #                            axis=0)

        kendall_all_roi = np.zeros(n_roi)
        for roi in range(n_roi):

            rdm_fmri_roi = rdm_fmri_mean[roi]

            # get above diagonal element
            rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

            # mean-zeroing and max-normalize
            rdm_fmri_norm = rdm_fmri_above - np.mean(rdm_fmri_above)
            rdm_fmri_norm = rdm_fmri_norm / np.max(rdm_fmri_norm)

            rdm_corr_roi = w_corr_norm[roi] * rdm_corr_norm
            rdm_match_roi = w_match_norm[roi] * rdm_match_norm
            # rdm_corr_roi = w_bootstrap_fin[roi, 0]*rdm_corr_norm
            # rdm_match_roi = w_bootstrap_fin[roi, 1]*rdm_match_norm
            rdm_fit = rdm_corr_roi + rdm_match_roi

            # calculate kendalltau
            kendall_all_roi[roi] = kendalltau(rdm_fmri_norm, rdm_fit)[0]

        ## get the median of low and upper noise ceiling
        bound_low = np.median(np.mean(kendalltau_low, axis=2), axis=1)
        bound_up = np.median(kendalltau_up, axis=1)

        # plot
        sns.set()
        sns.set(context="paper", style="white", font_scale=4, palette="deep")

        pos = list(np.arange(0, 24, 3))

        plt.figure(figsize=(14, 10))
        plt.bar(pos, kendall_all_roi, width=2.5, color="gray", capsize=3)

        # plot lower and upper noise ceiling
        for roi in range(n_roi):
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
        plt.title("Goodness of fit", pad=20)

        y_low = 0.0
        y_up = 1.01
        y_step = 0.2
        plt.ylim(y_low, y_up, y_step)
        plt.yticks(
            np.arange(y_low, y_up, y_step), np.round(np.arange(y_low, y_up, y_step), 2)
        )

        # remove the top and right frames
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("left")

        ## save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM/Weight_RDM_fitting/Noise{}/PlotBar_goodness_of_fit_nVox_{}_noise{}.pdf".format(
                    str(sawtooth_noise_std),
                    str(nVox_to_analyze),
                    str(sawtooth_noise_std),
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotBar_s2n_fmri(self, s2n_all_sbj, nVox_to_analyze, save_flag):
        """
        plot bar signal to noise

        Parameters
        ----------
        s2n_all_sbj : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across participants
        s2n_avg = np.mean(s2n_all_sbj, axis=0)

        s2n_avg_ards_crossed = s2n_avg[:, 0]
        s2n_avg_ards_uncrossed = s2n_avg[:, 1]
        s2n_avg_hmrds_crossed = s2n_avg[:, 2]
        s2n_avg_hmrds_uncrossed = s2n_avg[:, 3]
        s2n_avg_crds_crossed = s2n_avg[:, 4]
        s2n_avg_crds_uncrossed = s2n_avg[:, 5]

        # standard error across participants
        s2n_sem = sem(s2n_all_sbj, axis=0)

        s2n_sem_ards_crossed = s2n_sem[:, 0]
        s2n_sem_ards_uncrossed = s2n_sem[:, 1]
        s2n_sem_hmrds_crossed = s2n_sem[:, 2]
        s2n_sem_hmrds_uncrossed = s2n_sem[:, 3]
        s2n_sem_crds_crossed = s2n_sem[:, 4]
        s2n_sem_crds_uncrossed = s2n_sem[:, 5]

        plt.style.use("seaborn-colorblind")
        sns.set()
        sns.set(context="paper", style="white", font_scale=2, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        # start plotting
        figsize = (10, 9)
        n_row = 3
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=False, sharey=False
        )

        # fig.text(0.5,1.02,
        #          "GLM beta values",
        #          ha="center")
        fig.text(
            -0.02,
            0.5,
            "Signal-to-noise ratio, # Voxels={}\nTask VS. Fixation".format(
                nVox_to_analyze
            ),
            va="center",
            ha="center",
            rotation=90,
        )
        # fig.text(0.5, -0.03,
        #          "ROI",
        #          va="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.5)

        id_early = [0, 1, 2, 3]  # the last element is dummy
        id_dorsal = [3, 4, 6, 7]
        id_ventral = [5, 5, 5, 5]  # the last 3 elements are dummy
        roi_early = ["V1", "V2", "V3", ""]
        roi_dorsal = ["V3A", "V3B", "V7", "MT"]
        roi_ventral = ["hV4", "", "", ""]

        bar_width = 6
        step = 42
        pos_init = step * 8
        pos_crds_crossed = np.arange(0, pos_init, step)
        d = bar_width
        pos_crds_uncrossed = np.arange(d, pos_init + d, step)

        d = 2 * bar_width
        pos_hmrds_crossed = np.arange(d, pos_init + d, step)
        d = 3 * bar_width
        pos_hmrds_uncrossed = np.arange(d, pos_init + d, step)

        d = 4 * bar_width
        pos_ards_crossed = np.arange(d, pos_init + d, step)
        d = 5 * bar_width
        pos_ards_uncrossed = np.arange(d, pos_init + d, step)

        color_bar = ["green", "orange", "magenta"]  # crds, hmrds, ards
        alpha_crossed = 0.4
        alpha_uncrossed = 1.0

        x_low = -6
        x_up = 160
        y_low = 0.0
        y_up = 2.1
        y_step = 0.5

        ## early cortex
        # plot crds_crossed early cortex
        temp = s2n_avg_crds_crossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_crds_crossed[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_crds_crossed[0 : len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot crds_uncrossed
        temp = s2n_avg_crds_uncrossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_crds_uncrossed[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_crds_uncrossed[0 : len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot hmrds_crossed
        temp = s2n_avg_hmrds_crossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_hmrds_crossed[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_hmrds_crossed[0 : len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot hmrds_uncrossed
        temp = s2n_avg_hmrds_uncrossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_hmrds_uncrossed[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_hmrds_uncrossed[0 : len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot ards_crossed
        temp = s2n_avg_ards_crossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_ards_crossed[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_ards_crossed[0 : len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot ards_uncrossed
        temp = s2n_avg_ards_uncrossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_ards_uncrossed[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_ards_uncrossed[0 : len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # remove top and right frame
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes[0].xaxis.set_ticks_position("bottom")
        axes[0].yaxis.set_ticks_position("left")

        axes[0].set_title("Early areas")

        axes[0].set_xlim(x_low, x_up)
        axes[0].set_ylim(y_low, y_up)
        axes[0].set_yticks(np.arange(y_low, y_up, y_step))
        axes[0].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        pos_roi = (
            pos_crds_crossed[0 : len(id_early)]
            + pos_crds_uncrossed[0 : len(id_early)]
            + pos_hmrds_crossed[0 : len(id_early)]
            + pos_hmrds_crossed[0 : len(id_early)]
            + pos_ards_crossed[0 : len(id_early)]
            + pos_ards_uncrossed[0 : len(id_early)]
        ) / 6
        axes[0].set_xticks(pos_roi)
        axes[0].set_xticklabels(roi_early)

        ## dorsal cortex
        # plot crds_crossed early cortex
        axes[1].bar(
            pos_crds_crossed[0 : len(id_dorsal)],
            s2n_avg_crds_crossed[id_dorsal],
            yerr=s2n_sem_crds_crossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot crds_uncrossed
        axes[1].bar(
            pos_crds_uncrossed[0 : len(id_dorsal)],
            s2n_avg_crds_uncrossed[id_dorsal],
            yerr=s2n_sem_crds_uncrossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot hmrds_crossed
        axes[1].bar(
            pos_hmrds_crossed[0 : len(id_dorsal)],
            s2n_avg_hmrds_crossed[id_dorsal],
            yerr=s2n_sem_hmrds_crossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot hmrds_uncrossed
        axes[1].bar(
            pos_hmrds_uncrossed[0 : len(id_dorsal)],
            s2n_avg_hmrds_uncrossed[id_dorsal],
            yerr=s2n_sem_hmrds_uncrossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot ards_crossed
        axes[1].bar(
            pos_ards_crossed[0 : len(id_dorsal)],
            s2n_avg_ards_crossed[id_dorsal],
            yerr=s2n_sem_ards_crossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot ards_uncrossed
        axes[1].bar(
            pos_ards_uncrossed[0 : len(id_dorsal)],
            s2n_avg_ards_uncrossed[id_dorsal],
            yerr=s2n_sem_ards_uncrossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # remove top and right frame
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes[1].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")

        axes[1].set_xlim(x_low, x_up)
        axes[1].set_ylim(y_low, y_up)
        axes[1].set_yticks(np.arange(y_low, y_up, y_step))
        axes[1].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        pos_roi = (
            pos_crds_crossed[0 : len(id_dorsal)]
            + pos_crds_uncrossed[0 : len(id_dorsal)]
            + pos_hmrds_crossed[0 : len(id_dorsal)]
            + pos_hmrds_crossed[0 : len(id_dorsal)]
            + pos_ards_crossed[0 : len(id_dorsal)]
            + pos_ards_uncrossed[0 : len(id_dorsal)]
        ) / 6
        axes[1].set_xticks(pos_roi)
        axes[1].set_xticklabels(roi_dorsal)

        axes[1].set_title("Dorsal areas")

        ## ventral cortex
        # plot crds_crossed early cortex
        temp = s2n_avg_crds_crossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_crds_crossed[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_crds_crossed[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot crds_uncrossed
        temp = s2n_avg_crds_uncrossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_crds_uncrossed[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_crds_uncrossed[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot hmrds_crossed
        temp = s2n_avg_hmrds_crossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_hmrds_crossed[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_hmrds_crossed[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot hmrds_uncrossed
        temp = s2n_avg_hmrds_uncrossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_hmrds_uncrossed[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_hmrds_uncrossed[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot ards_crossed
        temp = s2n_avg_ards_crossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_ards_crossed[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_ards_crossed[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot ards_uncrossed
        temp = s2n_avg_ards_uncrossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_ards_uncrossed[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_ards_uncrossed[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # remove top and right frame
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes[2].xaxis.set_ticks_position("bottom")
        axes[2].yaxis.set_ticks_position("left")

        axes[2].set_title("Ventral areas")

        axes[2].set_xlim(x_low, x_up)
        axes[2].set_ylim(y_low, y_up)
        axes[2].set_yticks(np.arange(y_low, y_up, y_step))
        axes[2].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        pos_roi = (
            pos_crds_crossed[0 : len(id_ventral)]
            + pos_crds_uncrossed[0 : len(id_ventral)]
            + pos_hmrds_crossed[0 : len(id_ventral)]
            + pos_hmrds_crossed[0 : len(id_ventral)]
            + pos_ards_crossed[0 : len(id_ventral)]
            + pos_ards_uncrossed[0 : len(id_ventral)]
        ) / 6
        axes[2].set_xticks(pos_roi)
        axes[2].set_xticklabels(roi_ventral)

        # plt.ylabel("GLM beta", labelpad=15)

        axes[2].legend(
            [
                "cRDS_crossed",
                "cRDS_uncrossed",
                "hmRDS_crossed",
                "hmrds_uncrossed",
                "aRDS_crossed",
                "aRDS_uncrossed",
            ],
            fontsize=12,
            bbox_to_anchor=(0.7, 0.05),
        )

        ## save plot
        if save_flag == 1:
            plt.savefig(
                "../../../Plots/CMM/S2N/PlotBar_s2n_nVox_{}.pdf".format(
                    nVox_to_analyze
                ),
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def plotLine_s2n_across_voxel(
        self, s2n_all_vox, nVox_list, nVoxMax_to_analyze, save_flag
    ):
        """
        plot line signal to noise as a function of voxel

        Parameters
        ----------
        s2n_all_vox : TYPE
            DESCRIPTION.
        nVox_list : TYPE
            DESCRIPTION.
        nVoxMax_to_analyze : TYPE
            DESCRIPTION.

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        s2n_avg = np.mean(s2n_all_vox, axis=1)  # [nVox, nROIs, nConds]
        s2n_sem = sem(s2n_all_vox, axis=1)  # [nVox, nROIs, nConds]

        plt.style.use("seaborn-colorblind")
        sns.set()
        sns.set(context="paper", style="white", font_scale=3, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        figsize = (17, 17)
        n_row = 3
        n_col = 3
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(0.5, 1.03, "Signal-to-noise ratio as a function of voxel", ha="center")
        fig.text(-0.02, 0.5, "Signal-to-noise ratio", va="center", rotation=90)
        fig.text(0.5, -0.02, "# voxels", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        y_low = 0.4
        y_up = 0.81
        y_step = 0.1
        x_low = 0
        x_up = nVox_list[-1] + 26
        x_step = 50

        # find the index for the max number of voxels used for the analysis
        voxMax_id = np.where(nVox_list == nVoxMax_to_analyze)[0][0] + 1
        x = nVox_list[0:voxMax_id]

        x_low = 0
        x_up = 325
        x_step = 100
        y_low = 0.0
        y_up = 2.1
        y_step = 0.5

        for roi in range(self.n_ROIs):
            id_row = np.int(roi / n_row)
            id_col = roi % n_col

            s2n_avg_ards_crossed = s2n_avg[0:voxMax_id, roi, 0]
            s2n_avg_ards_uncrossed = s2n_avg[0:voxMax_id, roi, 1]
            s2n_avg_hmrds_crossed = s2n_avg[0:voxMax_id, roi, 2]
            s2n_avg_hmrds_uncrossed = s2n_avg[0:voxMax_id, roi, 3]
            s2n_avg_crds_crossed = s2n_avg[0:voxMax_id, roi, 4]
            s2n_avg_crds_uncrossed = s2n_avg[0:voxMax_id, roi, 5]

            s2n_sem_ards_crossed = s2n_sem[0:voxMax_id, roi, 0]
            s2n_sem_ards_uncrossed = s2n_sem[0:voxMax_id, roi, 1]
            s2n_sem_hmrds_crossed = s2n_sem[0:voxMax_id, roi, 2]
            s2n_sem_hmrds_uncrossed = s2n_sem[0:voxMax_id, roi, 3]
            s2n_sem_crds_crossed = s2n_sem[0:voxMax_id, roi, 4]
            s2n_sem_crds_uncrossed = s2n_sem[0:voxMax_id, roi, 5]

            axes[id_row, id_col].errorbar(
                x,
                s2n_avg_ards_crossed,
                yerr=s2n_sem_ards_crossed,
                linestyle="-",
                color="magenta",
                linewidth=4,
            )
            axes[id_row, id_col].errorbar(
                x,
                s2n_avg_ards_uncrossed,
                yerr=s2n_sem_ards_uncrossed,
                linestyle="--",
                color="magenta",
                linewidth=4,
            )

            axes[id_row, id_col].errorbar(
                x,
                s2n_avg_hmrds_crossed,
                yerr=s2n_sem_hmrds_crossed,
                linestyle="-",
                color="orange",
                linewidth=4,
            )
            axes[id_row, id_col].errorbar(
                x,
                s2n_avg_hmrds_uncrossed,
                yerr=s2n_sem_hmrds_uncrossed,
                linestyle="--",
                color="orange",
                linewidth=4,
            )

            axes[id_row, id_col].errorbar(
                x,
                s2n_avg_crds_crossed,
                yerr=s2n_sem_crds_crossed,
                linestyle="-",
                color="green",
                linewidth=4,
            )
            axes[id_row, id_col].errorbar(
                x,
                s2n_avg_crds_uncrossed,
                yerr=s2n_sem_crds_uncrossed,
                linestyle="--",
                color="green",
                linewidth=4,
            )

            axes[id_row, id_col].set_title(self.ROIs[roi])

            # remove the top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(x_low, x_up, x_step), 2)
            )

            axes[id_row, id_col].set_ylim(y_low, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low, y_up, y_step), 2)
            )

        # plot dummy for legend
        x_dummy = [0, 50]
        y_dummy = [0.25, 0.25]
        axes[id_row, id_col + 1].plot(
            x_dummy, y_dummy, linestyle="-", color="green", linewidth=4
        )
        axes[id_row, id_col + 1].plot(
            x_dummy, y_dummy, linestyle="--", color="green", linewidth=4
        )
        axes[id_row, id_col + 1].plot(
            x_dummy, y_dummy, linestyle="-", color="orange", linewidth=4
        )
        axes[id_row, id_col + 1].plot(
            x_dummy, y_dummy, linestyle="--", color="orange", linewidth=4
        )
        axes[id_row, id_col + 1].plot(
            x_dummy, y_dummy, linestyle="-", color="magenta", linewidth=4
        )
        axes[id_row, id_col + 1].plot(
            x_dummy, y_dummy, linestyle="--", color="magenta", linewidth=4
        )

        axes[id_row, id_col + 1].legend(
            [
                "cRDS_crossed",
                "cRDS_uncrossed",
                "hmRDS_crossed",
                "hmRDS_uncrossed",
                "aRDS_crossed",
                "aRDS_uncrossed",
            ],
            fontsize=24,
        )

        axes[id_row, id_col + 1].set_ylim(0, 0.1)
        axes[id_row, id_col + 1].get_xaxis().set_visible(False)
        axes[id_row, id_col + 1].get_yaxis().set_visible(False)

        # remove frame
        axes[id_row, id_col + 1].spines["top"].set_visible(False)
        axes[id_row, id_col + 1].spines["right"].set_visible(False)
        axes[id_row, id_col + 1].spines["bottom"].set_visible(False)
        axes[id_row, id_col + 1].spines["left"].set_visible(False)

        ## save plot
        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/S2N/PlotLine_s2n.pdf",
                dpi=self.dpi,
                bbox_inches="tight",
            )
