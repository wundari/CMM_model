"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/GLM/GLM_v2.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/GLM
Created Date: 2022-06-27 10:18:28
Author: Bayu G. Wundari
-----
Last Modified: 2022-06-29 12:03:35
Modified By: Bayu G. Wundari

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
script for computing GLM from vtc_data obtained from Matlab.

The vtc data are stored in:
    /media/cogni/EVO4TB/fMRI_Analysis/bw18_005/Analysis/Data/vtc_extract_non_smooth
    
    or
    
    /media/cogni/EVO4TB/fMRI_Analysis/bw18_005/Analysis/Data/vtc_extract_smooth
    for vtc data that has been filtered

    However, these two folders vtc_extract_smoooth and vtc_extract_non_smoooth have 
    been copied to 
    /home/wundari/NVME/fmri_data_processing/bw18_005_2/Data/
    
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from timeit import default_timer as timer

import scipy.io as sio
from scipy.stats import gamma
from scipy.stats import sem
from scipy.linalg import pinv

import matplotlib.pyplot as plt
import seaborn as sns

import sys

# sys.path.append("../Common")
from Common.Common import General

# General = Common.General()


class GLM(General):
    def __init__(self):

        super().__init__()

    def create_hrf2gamma(self, t_axis, c_undershoot, t_peak, t_undershoot):
        """
        create hemodynamic function based on two gamma function

        Parameters
        ----------
        t_axis : np.array
            the time axis of the hrf.
            for example: t_axis = np.range(16)

        c_undershoot : scalar, optional
            the scaling constant for the undershoot gamma.
            The default is 0.35.

        t_peak : scalar, optional
            the timepoint denoting the peak of the hrf.
            The default is 3.

        t_undershoot : scalar, optional
            the timepoint denoting the undershoot of the hrf.
            The default is 7.

        Returns
        -------
        hrf : np.array
            the resulting hrf.

        """

        # gamma peak
        gamma1 = gamma.pdf(t_axis, t_peak)

        # gamma undershoot
        gamma2 = c_undershoot * gamma.pdf(t_axis, t_undershoot)

        # hrf
        hrf = gamma1 - gamma2

        # normalize
        hrf = hrf / np.sum(hrf)

        return hrf

    def create_designMatrix(self, vtc_stimID, sbjID):
        """
        create a desing matrix accroding to the experimental protocol and
        convolve it with hrf

        Parameters
        ----------
        vtc_stimID : TYPE
            DESCRIPTION.
        sbj : TYPE
            DESCRIPTION.

        Returns
        -------
        designMatrix_conv : [n_timepoints, n_conds+n_nuissance_regressors] np.array
            convolved design matrix.

        """

        nRuns = np.shape(vtc_stimID)[1]

        # generate hrf based on two gamma function
        t_axis = np.arange(16)
        c_undershoot = 0.5
        t_peak = 4
        t_undershoot = 6
        hrf = self.create_hrf2gamma(t_axis, c_undershoot, t_peak, t_undershoot)
        # hrf = glm.create_hrf2gamma(t_axis, c_undershoot, t_peak, t_undershoot)

        # calculate the number of rows t and columns v for the whole run
        # number of timepoints per scan session
        n_timepoints_all_run = nRuns * self.n_timepoints_per_run
        # n_timepoints_all_run = nRuns * glm.n_timepoints_per_run

        # allocate designMatrix
        designMatrix = np.zeros((n_timepoints_all_run, self.n_conds), dtype=np.float32)
        # designMatrix = np.zeros((n_timepoints_all_run, glm.n_conds), dtype=np.float32)

        # allocate designMatrix_conv
        n_col = self.n_conds + nRuns * self.n_nuissance_reg
        # n_col = glm.n_conds + nRuns * glm.n_nuissance_reg
        designMatrix_conv = np.zeros((n_timepoints_all_run, n_col), dtype=np.float32)

        nuissance_reg = np.zeros((n_timepoints_all_run, nRuns * self.n_nuissance_reg))
        # nuissance_reg = np.zeros((n_timepoints_all_run, nRuns * glm.n_nuissance_reg))
        for run in range(nRuns):

            for block in range(
                1, self.n_blocks_per_run + 1
            ):  # skip the fixation in the first block
                # for block in range(
                #     1, glm.n_blocks_per_run + 1
                # ):  # skip the fixation in the first block
                id_start = (
                    3
                    + run * self.n_timepoints_per_run
                    + (block - 1) * self.n_timepoints_per_block
                )
                # id_start = (
                #     3
                #     + run * glm.n_timepoints_per_run
                #     + (block - 1) * glm.n_timepoints_per_block
                # )

                id_end = id_start + self.n_timepoints_per_block
                # id_end = id_start + glm.n_timepoints_per_block

                # fetch condition id, associated with the column of designMatrix
                cond = vtc_stimID[block, run] - 1
                designMatrix[id_start:id_end, cond] = 1

            ## create nuissance regressors
            # load rtc file
            # sbjID = self.sbjID_all[sbj]
            rtc = self.load_rtc_file(sbjID, run)
            # rtc = glm.load_rtc_file(sbjID, run)

            id_start = run * self.n_timepoints_per_run
            id_end = id_start + self.n_timepoints_per_run
            # id_start = run * glm.n_timepoints_per_run
            # id_end = id_start + glm.n_timepoints_per_run

            # convolve design matrix with hrf
            for m in range(self.n_conds):
                # for m in range(glm.n_conds):
                temp = np.convolve(designMatrix[id_start:id_end, m], hrf)[
                    : self.n_timepoints_per_run
                ]
                # temp = np.convolve(designMatrix[id_start:id_end, m], hrf)[
                #     : glm.n_timepoints_per_run
                # ]
                designMatrix_conv[id_start:id_end, m] = temp

            col_start = run * self.n_nuissance_reg
            col_end = col_start + self.n_nuissance_reg - 1
            # col_start = run * glm.n_nuissance_reg
            # col_end = col_start + glm.n_nuissance_reg - 1
            nuissance_reg[id_start:id_end, col_start:col_end] = rtc[:, 6:]

            # rtc = glm.load_rtc_file(sbjID, run)

            # id_start = run*self.n_timepoints_per_run
            # id_end = id_start + self.n_timepoints_per_run
            # col_start = run*self.n_nuissance_reg
            # col_end = col_start + self.n_nuissance_reg - 1
            # nuissance_reg[id_start:id_end, col_start:col_end] = rtc[:, 6:]

            # add a constant column after nuissance regressor
            nuissance_reg[id_start:id_end, col_end] = np.ones(self.n_timepoints_per_run)
            # nuissance_reg[id_start:id_end, col_end] = np.ones(glm.n_timepoints_per_run)

        # ## convolve designMatrix with hemodynamic function
        # # generate hrf based on two gamma function
        # t_axis = np.arange(16)
        # c_undershoot = 0.5
        # t_peak = 4
        # t_undershoot = 6
        # hrf = self.create_hrf_2gamma(t_axis, c_undershoot,
        #                              t_peak, t_undershoot)

        # # convolve the designMatrix with the hrf
        # n_col = self.n_conds + nRuns*self.n_nuissance_reg
        # designMatrix_conv = np.zeros((n_timepoints_all_run, n_col),
        #                              dtype=np.float32)
        # # designMatrix_conv = np.zeros((2262, n_conds), dtype=np.float32)
        # for m in range(self.n_conds):
        #     temp = np.convolve(designMatrix[:, m], hrf)[:n_timepoints_all_run]
        #     # designMatrix_conv[:, m] = temp/np.max(temp)
        #     designMatrix_conv[:, m] = temp

        ## append nuissance regressors
        designMatrix_conv[:, self.n_conds :] = nuissance_reg
        # designMatrix_conv[:, glm.n_conds :] = nuissance_reg

        # m = 1
        # plt.plot(designMatrix[0:203, m]), plt.plot(designMatrix_conv[0:203, m])
        # plt.plot(np.sum(designMatrix[0:203], axis=1)), plt.plot(np.sum(designMatrix_conv[0:203], axis=1))

        return designMatrix_conv

    def _compute_vox_avg(self, vtc_labeled, sbj):
        """
        Compute voxel average for each roi, run, condition, vox.
        The condition includes fixation

        Parameters
        ----------
        vtc_labeled : TYPE
            DESCRIPTION.
        sbj : TYPE
            DESCRIPTION.

        Returns
        -------
        vtc_avg: dict
                a dictionary containing the voxel average for each roi, run, condition, vox
                {V1: [nRuns, 7, nVox],
                 V2: [nRuns, 7, nVox], ...}.

                the condition includes fixation, thus nConds = 7

        """
        nRuns = self.nRuns_all[sbj]

        ## group across roi, run, vox, and condition
        print("Grouping dataframe")
        vtc_group = (
            vtc_labeled.groupby(["roi", "run", "vox", "stimID", "cond"])
            .vtc_value.agg(["mean", sem])
            .reset_index()
        )
        vtc_group = vtc_group.rename(columns={"mean": "vtc_avg", "sem": "vtc_sem"})

        vtc_avg = {}
        for roi in range(self.n_ROIs):

            vtc_roi = vtc_group.loc[vtc_group.roi == roi]
            nVox = np.int(vtc_roi.vox.max() + 1)  # voxel idx starts from 0
            y_roi = np.zeros(
                (nRuns, 7, nVox), dtype=np.float32
            )  # [nRuns, 7, nVox], include fixation
            for run in range(nRuns):

                print(
                    "compute voxel average, sbjID={}, roi={}, run={}".format(
                        self.sbjID_all[sbj], self.ROIs[roi], run + 1
                    )
                )

                # get voxel values for each run, cond
                vtc_group_roi = vtc_group.loc[
                    (vtc_group.roi == roi) & (vtc_group.run == run)
                ]

                # create matrix [cond, vox]
                y_cond = np.array(
                    vtc_group_roi.pivot_table(
                        index="stimID", columns="vox", values="vtc_avg"
                    ),
                    dtype=np.float32,
                )

                # average across condition in each run
                # y_temp = np.mean(y_cond, axis=0)
                # y_avg_across_cond = np.tile(y_temp, [7, 1])

                # # normalize voxel
                # y_norm = (y_cond - y_avg_across_cond)/y_avg_across_cond * 100

                y_roi[run] = y_cond  # [nRuns, 7, nVox], include fixation

            vtc_avg[self.ROIs[roi]] = y_roi

        return vtc_avg

    def compute_signal_change_single_sbj(self, t_stat_all_sbjID, nVox_to_analyze, sbj):
        """
        compute percent signal change for a single participant.
        The percent signal change is defined as:

            percent-change = (y - y_fix)/y_fix * 100

        Parameters
        ----------
        t_stat_all_sbjID : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.
        sbj : TYPE
            DESCRIPTION.

        Returns
        -------
        y_diff_all_roi : TYPE
            DESCRIPTION.

        """

        sbjID = self.sbjID_all[sbj]
        nRuns = self.nRuns_all[sbj]
        nConds = self.n_conds  # exclude fixation

        # # load vtc
        # vtc = self.load_vtc(sbj)

        # # load stimulus timing parameters
        # vtc_stimID = sio.loadmat("../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}"
        #                          .format(sbjID))["paramIdFull"]

        # # label vtc
        # vtc_labeled = self.label_vtc(vtc, vtc_stimID)

        ## load vtc_labeled
        vtc_labeled = self.load_vtc_labeled(sbjID)

        y_diff_all_roi = np.zeros((self.n_ROIs, nRuns, nConds), dtype=np.float32)

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
            # avg = (vtc_roi.groupby(["roi", "run", "vox"])["vtc_value"]
            #                 .transform("mean"))

            # # normalize
            # temp = (vtc_roi.vtc_value - avg)/avg * 100
            # vtc_roi = vtc_roi.assign(vtc_norm=temp)

            ## check if nVox_to_analyze < nVox_max in this ROI
            nVox_max = len(t_roi)
            if nVox_to_analyze >= nVox_max:
                nVox_to_analyze = nVox_max

            for run in range(nRuns):

                print(
                    "compute percent signal change, sbjID={}, ROI={}, nVox={}, run={}".format(
                        sbjID, self.ROIs[roi], nVox_to_analyze, run + 1
                    )
                )

                vtc_run = vtc_roi.loc[vtc_roi.run == run]

                # group by [roi, run, vox, cond], average across timepoints
                vtc_group = (
                    vtc_run.groupby(["vox", "stimID", "cond"])
                    .vtc_value.agg(["mean", np.var])
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
                )  # [nConds, nVox], including fixation

                # compute the response difference between stimulus and fixation
                y_diff = (
                    (y_sel[1:] - y_fix) / y_fix * 100
                )  # [nConds, nVox], excluding fixation

                # average across these voxels
                y_diff_all_roi[roi, run] = np.mean(
                    y_diff, axis=1
                )  # [nROIs, nRuns, nConds], exclude fixation

        return y_diff_all_roi

    def compute_signal_change_all_sbj(self, t_stat_all_sbjID, nVox_to_analyze):
        """
        compute signal change for all particpants.
        The percent signal change is defined as:

            percent-change = (y - y_fix)/y_fix * 100

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

        signal_change_list = []

        t_start = timer()
        signal_change_list.append(
            Parallel(n_jobs=10)(
                delayed(self.compute_signal_change_single_sbj)(
                    t_stat_all_sbjID, nVox_to_analyze, sbj
                )
                for sbj in range(self.n_sbjID)
            )
        )
        t_end = timer()
        print(t_end - t_start)

        # extract s2n_list
        nConds = self.n_conds  # exclude fixation
        signal_change_all_sbj = np.zeros(
            (self.n_sbjID, self.n_ROIs, nConds), dtype=np.float32
        )
        for sbj in range(self.n_sbjID):
            signal_change = signal_change_list[0][sbj]  # [nROIs, nRuns, nConds]

            # average across run
            signal_avg = np.mean(signal_change, axis=1)  # [nROIs, nConds]

            signal_change_all_sbj[sbj] = signal_avg  # [n_sbjID, nROIs, nConds]

        return signal_change_all_sbj

    def compute_glm(self, vtc_labeled, designMatrix_conv, mtd_normalization):
        """
        glm analysis for all roi.

        Parameters
        ----------
        vtc_labeled : TYPE
            DESCRIPTION.
        designMatrix_conv : TYPE
            DESCRIPTION.
        nROIs : TYPE
            DESCRIPTION.

        Returns
        -------
        beta_all_roi : dict
            a dictionary containing beta glm for each ROI.
            Each element in the dictionary is np.array

            [n_regressors, nVox_max] np.array

        beta_avg_roi : [roi, n_regressors_main_conditions)] np.array
            beta value after averaged across voxel.
            it only contains the regressors for the main conditions.


        """

        # compute pseudo inverse
        # X_plus = np.linalg.pinv(designMatrix_conv) # pseudoinverse
        X_plus = pinv(designMatrix_conv)  # pseudoinverse

        # beta = X_plus.dot(y)
        # x = designMatrix_conv.copy()
        # X_plus = np.linalg.inv(x.T.dot(x)).dot(x.T)

        # nROIs = len(self.ROIs)

        # p = np.shape(designMatrix_conv)[1] # number of regressors
        # beta_all_roi = np.zeros((self.n_ROIs, p, nVox_max),
        #                         dtype=np.float32)
        beta_avg_roi = np.zeros(
            (self.n_ROIs, 6), dtype=np.float32
        )  # beta for avg across voxel in roi
        beta_all_roi = {}

        for roi in range(self.n_ROIs):

            # filter dataframe according to roi
            vtc_roi = vtc_labeled.loc[(vtc_labeled.roi == roi)]

            # get the max nVox in the given roi
            # nVox_roi = np.int(vtc_roi.vox.max())

            ## compute beta for each voxel
            # mtd_normalization = 1 # % bold change
            y = self.normalize_vtc_roi(vtc_roi, mtd_normalization)  # [timepoint, vox]
            # y = glm.normalize_vtc_roi(vtc_roi, mtd_normalization) # [timepoint, vox]
            # y = vtc_norm[self.ROIs[roi]] # [timepoint_all_run, nVox]

            beta = X_plus.dot(y)
            beta_all_roi[self.ROIs[roi]] = beta

            ## compute beta_roi, after avg signal across voxels
            # average across voxel
            y_mean = np.mean(y, axis=1)  # [n_timepoints]
            # plt.plot(y[0:203])
            beta_mean = X_plus.dot(y_mean)
            beta_avg_roi[roi, :] = beta_mean[0:6]

            # reconstruct model
            # y_glm = X.dot(beta)

            # i_start = 0
            # i_end = 203
            # plt.plot(y[i_start:i_end, 0]), plt.plot(y_glm[i_start:i_end, 0])

        return beta_all_roi, beta_avg_roi

    def compute_glm_single_sbjID(self, sbj, mtd_normalization):
        """
        compute beta glm for a given sbj

        Parameters
        ----------
        sbj : scalar
            an index for participant.
        mtd_normalization : scalar
            a scalar .

        Returns
        -------
        beta_sbj : dict
            a dictionary.
        beta_avg_sbj : TYPE
            DESCRIPTION.

        """

        sbjID = self.sbjID_all[sbj]

        print("glm, sbjID:{}".format(sbjID))
        # load stimulus timing parameters
        vtc_stimID = sio.loadmat(
            "../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        )["paramIdFull"]

        # # load vtc file
        # vtc = self.load_vtc(sbj) # ["run", "roi", "timepoint", "vox", "vtc_value"].
        # # vtc = glm.load_vtc(sbj)

        # # label vtc data
        # vtc_labeled = self.label_vtc(vtc, vtc_stimID)
        # vtc_labeled = glm.label_vtc(vtc, vtc_stimID)

        ## load vtc_labeled, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        vtc_labeled = self.load_vtc_labeled(sbjID)
        # vtc_labeled = glm.load_vtc_labeled(sbjID)

        # construct design matrix, [nRuns*203, n_regressors*
        # designMatrix_conv = self.create_designMatrix(vtc_stimID, sbjID)
        designMatrix_conv = self.rtc2designMatrix(sbj)
        # designMatrix_conv2 = glm.rtc2designMatrix(sbj)

        # compute beta
        beta_sbj, beta_avg_sbj = self.compute_glm(
            vtc_labeled, designMatrix_conv, mtd_normalization
        )
        # beta_sbj, beta_avg_sbj = glm.compute_glm(vtc_labeled,
        #                                           designMatrix_conv,
        #                                           mtd_normalization)

        return beta_sbj, beta_avg_sbj

    def compute_glm_all_sbjID(self, mtd_normalization):

        # n_sbjID = len(self.sbjID_all)
        # nROIs = len(self.ROIs)

        beta_all = []
        t_start = timer()
        beta_all.append(
            Parallel(n_jobs=8)(
                delayed(self.compute_glm_single_sbjID)(sbj, mtd_normalization)
                for sbj in range(self.n_sbjID)
            )
        )
        t_end = timer()
        print(t_end - t_start)  # 204 sec, n_jobs=10, backend="loky"

        # unpack
        beta_all_sbjID = []
        beta_avg_all_sbjID = np.zeros((self.n_sbjID, self.n_ROIs, 6))
        for sbj in range(self.n_sbjID):
            beta_all_sbjID.append(beta_all[0][sbj][0])

            beta_avg_all_sbjID[sbj] = beta_all[0][sbj][1]

        # nRuns_max = np.max(self.nRuns_all)
        # p = self.n_regressors + nRuns_max*self.n_nuissance_reg # number of regressors
        # beta_all_sbjID = np.zeros((self.n_sbjID, self.n_ROIs, p, nVox_max),
        #                     dtype=np.float32)
        # beta_avg_roi_all_sbjID = np.zeros((self.n_sbjID, self.n_ROIs, p),
        #             dtype=np.float32)
        # for sbj in range(self.n_sbjID):
        #     nRuns = self.nRuns_all[sbj]
        #     p = self.n_regressors + nRuns*self.n_nuissance_reg # number of regressors for this sbjID
        #     beta_all_sbjID[sbj, :, 0:p] = beta_all[0][sbj][0]

        #     beta_avg_roi_all_sbjID[sbj, :, 0:p] = beta_all[0][sbj][1]

        # return beta_all_sbjID, beta_avg_roi_all_sbjID

        # beta_all_sbjID = beta_all[]

        return beta_all_sbjID, beta_avg_all_sbjID

    def diagnose_glm(self, beta_all_sbjID, sbj, roi, vox, run, mtd_normalization):

        sbjID = self.sbjID_all[sbj]

        vtc_stimID = sio.loadmat(
            "../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        )["paramIdFull"]

        # design matrix [n_timepoints, n_conds+n_nuissance_regressors]
        designMatrix_conv = self.create_designMatrix(vtc_stimID, sbjID)
        # designMatrix_conv = glm.create_designMatrix(vtc_stimID, sbjID)
        p = np.shape(designMatrix_conv)[1]  # number of regressors
        # p = 6

        # get beta [n_regressors, nVox]
        beta = beta_all_sbjID[sbj][self.ROIs[roi]][0:p]

        # compute y_glm
        # y_glm = designMatrix_conv.dot(beta)
        y_glm = designMatrix_conv[:, 0:p].dot(beta)

        # load vtc file
        vtc = self.load_vtc(sbj)

        # label vtc data
        vtc_labeled = self.label_vtc(vtc, vtc_stimID)

        # filter dataframe according to roi and nVox
        vtc_roi = vtc_labeled.loc[(vtc_labeled.roi == roi)]

        # mtd_normalization = 2 # % bold change
        y = self.normalize_vtc_roi(vtc_roi, mtd_normalization)  # [n_timepoints, nVox]
        # y = glm.normalize_vtc_roi(vtc_roi, mtd_normalization)

        # calculate residual
        e = y - y_glm

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        plt.figure(figsize=(10, 10))
        # vox = 0
        i_start = run * self.n_timepoints_per_run
        i_end = i_start + self.n_timepoints_per_run

        plt.plot(y[i_start:i_end, vox])  # plot y_true
        plt.plot(y_glm[i_start:i_end, vox])  # plot y_glm
        plt.plot(e[i_start:i_end, vox])
        plt.legend(["y_true", "y_glm", "error"])

        plt.xlabel("Timepoint")
        if mtd_normalization == -1:
            plt.ylabel("No normalization")
        elif mtd_normalization == 0:
            plt.ylabel("z-score")
        elif mtd_normalization == 1:
            plt.ylabel("% BOLD-change")
        elif mtd_normalization == 2:
            plt.ylabel("% signal-change")

        plt.title(
            "GLM diagnostic\nsbjID: {}, ROI: {}, vox: {}".format(
                self.sbjID_all[sbj], self.ROIs[roi], str(vox)
            )
        )

        # plot residual
        # check residual covariance
        # compute f stat, see brainvoyager
        # https://support.brainvoyager.com/brainvoyager/functional-analysis-statistics/42-glm-modelling-single-study/216-users-guide-the-general-linear-model-glm

    def rtc2designMatrix(self, sbj):

        # load stimulus timing parameters
        sbjID = self.sbjID_all[sbj]
        # vtc_stimID = sio.loadmat("../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}"
        #                           .format(sbjID))["paramIdFull"]

        nRuns = self.nRuns_all[sbj]
        n_timepoints_all_run = nRuns * self.n_timepoints_per_run  # number of rows
        n_col = self.n_conds + self.n_nuissance_reg * nRuns  # number of columns
        designMatrix_from_rtc = np.zeros(
            (n_timepoints_all_run, n_col), dtype=np.float32
        )

        # convert rtc into design matrix
        nRuns = self.nRuns_all[sbj]
        for run in range(nRuns):

            id_start = run * self.n_timepoints_per_run
            id_end = id_start + self.n_timepoints_per_run

            # load rtc file
            rtc = self.load_rtc_file(sbjID, run)
            # rtc = glm.load_rtc_file(sbjID, run)

            # append a column vector of one to rtc
            rtc = np.append(rtc, np.ones((self.n_timepoints_per_run, 1)), axis=1)

            ## get the first 6 columns that contain the parameters for all
            # experimental conditions
            temp = rtc[:, 0:6]
            # normalize [0, 1]
            # temp = (temp - np.min(temp))/(np.max(temp) - np.min(temp))
            designMatrix_from_rtc[id_start:id_end, 0:6] = temp

            # get the nuissance regressors
            id_col_start = self.n_conds + run * self.n_nuissance_reg
            id_col_end = id_col_start + self.n_nuissance_reg

            temp = rtc[:, 6:]
            # normalize [-1, 1]
            # temp = (2 * (temp - np.min(temp)))/(np.max(temp) - np.min(temp)) - 1
            designMatrix_from_rtc[id_start:id_end, id_col_start:id_col_end] = temp

        return designMatrix_from_rtc

    # def diagnose_designMatrix(self, designMatrix_conv,
    #                           sbjID, run):

    #     ## load rtc file
    #     # sbjID = self.sbjID_all[sbj]
    #     rtc = glm.load_rtc_file(sbjID, run)

    #     ## get manual designMatrix_conv
    #     # load stimulus timing parameters
    #     vtc_stimID = sio.loadmat("../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}"
    #                              .format(sbjID))["paramIdFull"]
    #     designMatrix_conv = glm.create_designMatrix(vtc_stimID, sbjID)

    #     designMatrix_conv2 = glm.create_designMatrix(vtc_stimID, sbjID)

    # diff =

    def compute_t_stat_single_sbjID(self, beta_all_sbjID, sbj, mtd_normalization):

        sbjID = self.sbjID_all[sbj]

        vtc_stimID = sio.loadmat(
            "../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        )["paramIdFull"]
        designMatrix_conv = self.create_designMatrix(vtc_stimID, sbjID)
        # designMatrix_conv = glm.create_designMatrix(vtc_stimID, sbjID)
        # designMatrix_conv = self.rtc2designMatrix(sbj)
        p = np.shape(designMatrix_conv)[1]  # number of regressors
        # p = 6 # number of regressors, only the main conditions

        # # load vtc file
        # vtc = self.load_vtc(sbj) # ["run", "roi", "timepoint", "vox", "vtc_value"].
        # # vtc = glm.load_vtc(sbj)

        # # label vtc data
        # vtc_labeled = self.label_vtc(vtc, vtc_stimID)

        ## load vtc_labeled, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        vtc_labeled = self.load_vtc_labeled(sbjID)

        # remove fixation condition
        # vtc_labeled = vtc_labeled.drop(vtc_labeled[vtc_labeled.stimID==0].index)

        # nVox_max = 1000
        # t_stat_sbj = np.zeros((self.n_ROIs, nVox_max), dtype=np.float32)
        t_stat_sbj = {}
        for roi in range(self.n_ROIs):

            print("compute t-stat , sbjID: {}, roi: {}".format(sbjID, self.ROIs[roi]))

            # filter dataframe according to roi and nVox
            vtc_roi = vtc_labeled.loc[(vtc_labeled.roi == roi)]

            # mtd_normalization = 1 # % bold change
            # y = glm.normalize_vtc_roi(vtc_roi, mtd_normalization)
            y = self.normalize_vtc_roi(
                vtc_roi, mtd_normalization
            )  # [n_timepoints, vox]
            nVox = np.shape(y)[1]

            # compute y_glm
            # beta = beta_all_sbjID[sbj][glm.ROIs[roi]][0:p, 0:nVox]
            beta = beta_all_sbjID[sbj][self.ROIs[roi]][0:p, 0:nVox]
            y_glm = designMatrix_conv.dot(beta)[:, 0:nVox]

            # calculate residual
            e = y - y_glm
            # resid = (y - y_glm)**2
            # r2 = np.var(y_glm, axis=0)/(np.var(y_glm, axis=0) + np.var(e, axis=0))

            # compute explained variance
            # from sklearn.metrics import explained_variance_score
            # v = 1
            # r2 = explained_variance_score(y[:, v], y_glm[:, v])

            # compute f-stat
            # n = np.shape(y)[0] # the number of timepoint
            # p = np.shape(beta)[0] # the number of regressors
            # f = r2*(n - p)/((1 - r2) * (p - 1))

            ## compute t-stat
            # the t-stat here is computed to see if there is any significant
            # effect given by the average of all experimental conditions.
            # Thus, the contrast vector is translated as fix vs. all_condition.
            c = np.zeros(p)
            c[0:6] = 1 / 6  # to see there is any effect given by the average of
            # all experimental conditions

            var_e = np.var(e, axis=0)
            num = np.dot(c, beta)
            den = var_e.dot(
                c.T.dot(np.linalg.inv(designMatrix_conv.T.dot(designMatrix_conv))).dot(
                    c
                )
            )
            # t = num/den
            t = num / np.sqrt(den)
            t_stat_sbj[self.ROIs[roi]] = t

        return t_stat_sbj

    def compute_t_stat_all_sbjID(self, beta_all_sbjID, mtd_normalization):

        t_stat_all = []
        t_start = timer()
        t_stat_all.append(
            Parallel(n_jobs=self.n_sbjID)(
                delayed(self.compute_t_stat_single_sbjID)(
                    beta_all_sbjID, sbj, mtd_normalization
                )
                for sbj in range(self.n_sbjID)
            )
        )
        t_end = timer()
        print(t_end - t_start)  # 220 sec, n_jobs=len(sbjID_all), backend="loky"

        # unpack
        t_stat_all_sbjID = []
        for sbj in range(self.n_sbjID):
            t_stat_all_sbjID.append(t_stat_all[0][sbj])

        return t_stat_all_sbjID

    def sort_beta_single_sbjID(
        self, beta_all_sbjID, t_stat_all_sbjID, nVox_to_analyze, sbj
    ):

        beta_sort_sbj = np.zeros((self.n_ROIs, 6, nVox_to_analyze), dtype=np.float32)
        # t_stat_sort_sbj = np.zeros((self.n_ROIs, nVox), dtype=np.float32)
        for roi in range(self.n_ROIs):

            # get beta
            beta = beta_all_sbjID[sbj][self.ROIs[roi]]

            # get t_stat
            t_stat = t_stat_all_sbjID[sbj][self.ROIs[roi]]

            # sort voxel in t_stat descending order
            # a  = np.random.rand(10)
            # id_sort = a.argsort()[::-1]
            # a[id_sort]
            id_sort = t_stat.argsort()[::-1]

            # get beta associated with descended t_stat
            beta_sort = beta[0:6, id_sort]
            beta_sort_sbj[roi] = beta_sort[:, 0:nVox_to_analyze]

        return beta_sort_sbj

    def sort_beta_all_sbjID(self, beta_all_sbjID, t_stat_all_sbjID, nVox_to_analyze):

        beta_sort_all = []
        t_start = timer()
        beta_sort_all.append(
            Parallel(n_jobs=self.n_sbjID)(
                delayed(self.sort_beta_single_sbjID)(
                    beta_all_sbjID, t_stat_all_sbjID, nVox_to_analyze, sbj
                )
                for sbj in range(self.n_sbjID)
            )
        )
        t_end = timer()
        print(t_end - t_start)  # 2.3 sec, n_jobs=len(sbjID_all), backend="loky"

        # unpack
        beta_sort_all_sbjID = np.zeros(
            (self.n_sbjID, self.n_ROIs, 6, nVox_to_analyze), dtype=np.float32
        )
        for sbj in range(self.n_sbjID):
            beta_sort_all_sbjID[sbj] = beta_sort_all[0][sbj]

        return beta_sort_all_sbjID

    def plot_designMatrix(self, designMatrix_conv):

        plt.style.use("seaborn-colorblind")
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        sns.heatmap(designMatrix_conv[0:406, 0:26], vmin=-1, vmax=1, cmap="gray")

    def plotBar_beta(self, beta_all_sbjID):

        beta_avg = np.zeros((self.n_ROIs, self.n_sbjID, 6), dtype=np.float32)

        for roi in range(self.n_ROIs):

            roi_str = self.ROIs[roi]

            for sbj in range(self.n_sbjID):

                beta_roi = beta_all_sbjID[sbj][roi_str]

                # average beta across voxel
                temp = np.mean(beta_roi, axis=1)[0:6]
                beta_avg[roi, sbj] = temp

        # average across sbjID
        beta_avg_ards_crossed = np.mean(beta_avg[:, :, 0], axis=1)
        beta_avg_ards_uncrossed = np.mean(beta_avg[:, :, 1], axis=1)
        beta_avg_hmrds_crossed = np.mean(beta_avg[:, :, 2], axis=1)
        beta_avg_hmrds_uncrossed = np.mean(beta_avg[:, :, 3], axis=1)
        beta_avg_crds_crossed = np.mean(beta_avg[:, :, 4], axis=1)
        beta_avg_crds_uncrossed = np.mean(beta_avg[:, :, 5], axis=1)

        # compute sem
        beta_sem_ards_crossed = sem(beta_avg[:, :, 0], axis=1)
        beta_sem_ards_uncrossed = sem(beta_avg[:, :, 1], axis=1)
        beta_sem_hmrds_crossed = sem(beta_avg[:, :, 2], axis=1)
        beta_sem_hmrds_uncrossed = sem(beta_avg[:, :, 3], axis=1)
        beta_sem_crds_crossed = sem(beta_avg[:, :, 4], axis=1)
        beta_sem_crds_uncrossed = sem(beta_avg[:, :, 5], axis=1)

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        # start plotting
        figsize = (10, 8)
        n_row = 3
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=False, sharey=False
        )

        # fig.text(0.5,1.02,
        #          "GLM beta values",
        #          ha="center")
        fig.text(-0.03, 0.5, "GLM beta", va="center", rotation=90)
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
        y_up = 1.26
        y_step = 0.25

        ## early cortex
        # plot crds_crossed early cortex
        temp = beta_avg_crds_crossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = beta_sem_crds_crossed[id_early]
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
        temp = beta_avg_crds_uncrossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = beta_sem_crds_uncrossed[id_early]
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
        temp = beta_avg_hmrds_crossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = beta_sem_hmrds_crossed[id_early]
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
        temp = beta_avg_hmrds_uncrossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = beta_sem_hmrds_uncrossed[id_early]
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
        temp = beta_avg_ards_crossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = beta_sem_ards_crossed[id_early]
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
        temp = beta_avg_ards_uncrossed[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = beta_sem_ards_uncrossed[id_early]
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
            beta_avg_crds_crossed[id_dorsal],
            yerr=beta_sem_crds_crossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot crds_uncrossed
        axes[1].bar(
            pos_crds_uncrossed[0 : len(id_dorsal)],
            beta_avg_crds_uncrossed[id_dorsal],
            yerr=beta_sem_crds_uncrossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot hmrds_crossed
        axes[1].bar(
            pos_hmrds_crossed[0 : len(id_dorsal)],
            beta_avg_hmrds_crossed[id_dorsal],
            yerr=beta_sem_hmrds_crossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot hmrds_uncrossed
        axes[1].bar(
            pos_hmrds_uncrossed[0 : len(id_dorsal)],
            beta_avg_hmrds_uncrossed[id_dorsal],
            yerr=beta_sem_hmrds_uncrossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            alpha=alpha_uncrossed,
            capsize=3,
        )

        # plot ards_crossed
        axes[1].bar(
            pos_ards_crossed[0 : len(id_dorsal)],
            beta_avg_ards_crossed[id_dorsal],
            yerr=beta_sem_ards_crossed[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            alpha=alpha_crossed,
            capsize=3,
        )

        # plot ards_uncrossed
        axes[1].bar(
            pos_ards_uncrossed[0 : len(id_dorsal)],
            beta_avg_ards_uncrossed[id_dorsal],
            yerr=beta_sem_ards_uncrossed[id_dorsal],
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
        temp = beta_avg_crds_crossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = beta_sem_crds_crossed[id_ventral]
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
        temp = beta_avg_crds_uncrossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = beta_sem_crds_uncrossed[id_ventral]
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
        temp = beta_avg_hmrds_crossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = beta_sem_hmrds_crossed[id_ventral]
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
        temp = beta_avg_hmrds_uncrossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = beta_sem_hmrds_uncrossed[id_ventral]
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
        temp = beta_avg_ards_crossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = beta_sem_ards_crossed[id_ventral]
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
        temp = beta_avg_ards_uncrossed[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = beta_sem_ards_uncrossed[id_ventral]
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

        # plt.savefig("../../../Plots/CMM/GLM/PlotBar_glm.pdf",
        #             dpi=500,
        #             bbox_inches="tight")
