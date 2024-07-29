"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/Common/Common.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/Common
Created Date: 2022-06-20 22:57:29
Author: Bayu G. Wundari
-----
Last Modified: 2022-06-29 12:03:10
Modified By: Bayu G. Wundari
-----
Copyright (c) 2022 National Institute of Information and Communications Technology (NICT)

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io as sio


class General:
    def __init__(self):

        # self.sbjID_all = ['AI','BGW','DK',
        #                    'GH','HB','HD',
        #                    'HN','HO','JS',
        #                    'KT','KO','KY',
        #                    'MM','MI','NN',
        #                    'RS','SF','SHM',
        #                    'TO','TD',
        #                    'TAM','TM','YU']

        # self.nRuns_all = [10,11,10,
        #                     6,11,10,
        #                     8,10,8,
        #                     8,10,11,
        #                     4,9,11,
        #                     8,9,8,
        #                     6,10,
        #                     10,9,10] # number of runs for each sbjID

        self.sbjID_all = [
            "AI",
            "BGW",
            "GH",
            "HB",
            "HD",
            "HN",
            "HO",
            "JS",
            "KT",
            "KO",
            "KY",
            "MM",
            "MI",
            "NN",
            "RS",
            "SF",
            "SHM",
            "TO",
            "TD",
            "TAM",
            "TM",
            "YU",
        ]
        self.n_sbjID = len(self.sbjID_all)

        self.nRuns_all = [
            10,
            11,
            6,
            11,
            10,
            8,
            10,
            8,
            8,
            10,
            11,
            4,
            9,
            11,
            8,
            9,
            8,
            6,
            10,
            10,
            9,
            10,
        ]  # number of runs for each sbjID

        # list of participants who did retinotopy with MT localizer
        self.sbjID_with_MTlocalizer = [
            "AI",
            "BGW",
            "HB",
            "HO",
            "KO",
            "KY",
            "MI",
            "NN",
            "RS",
            "SF",
            "TD",
            "TM",
        ]

        # ## set up RDS parameters (the rds is square) -> original setup
        rDot = 0.045  # dot radius, deg
        dotDens = 0.25  # dot density
        size_rds_bg_deg = 2.5  # rds size, deg
        size_rds_ct_deg = 1.25  # center rds size, deg
        deg_per_pix = 0.02  # the size of a pixel, deg

        # # ## set up RDS parameters (the rds is square)
        # rDot = 0.045 # dot radius, deg
        # dotDens = 0.5 # dot density
        # size_rds_bg_deg = 1 # rds size, deg
        # size_rds_ct_deg = 0.6 # center rds size, deg
        # deg_per_pix = 0.01 # the size of a pixel, deg

        ## set up 1-D rds parameters
        # rDot = 0.2 # dot radius, deg
        # dotDens = 0.35 # dot density
        # size_rds_bg_deg = 7 # rds size, deg
        # size_rds_ct_deg = 3.5 # center rds size, deg
        # deg_per_pix = 0.02 # the size of a pixel, deg

        self.rDot = rDot  # dot radius
        self.dotDens = dotDens  # dot density
        self.size_rds_bg_deg = size_rds_bg_deg  # rds size, deg
        self.size_rds_ct_deg = size_rds_ct_deg  # center rds size, deg
        self.size_rds_bg_pix = self._compute_deg2pix(size_rds_bg_deg)
        self.size_rds_ct_pix = self._compute_deg2pix(size_rds_ct_deg)
        self.size_rds_bg = (self.size_rds_bg_pix, self.size_rds_bg_pix)
        self.size_rds_ct = (self.size_rds_ct_pix, self.size_rds_ct_pix)
        self.deg_per_pix = deg_per_pix

        self.ROIs = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "MT"]
        self.n_ROIs = len(self.ROIs)

        self.conds = ["fix", "n_c0", "f_c0", "n_c50", "f_c50", "n_c100", "f_c100"]
        self.n_conds = len(self.conds) - 1  # 6, excluding fixation
        # a list of comparison pairs, each element refers to self.conds.
        self.comp_pair_all = [[1, 2], [3, 4], [5, 6]]

        # n_blocks_per_run : scalar
        # the number of blocks per run: 24 (# conditions x # blocks_per_condition = 6 x 4).
        self.n_blocks_per_run = 24  # number of blocks per run

        # n_timepoints_per_block : scalar
        # the number of timepoints in each block: 8.
        self.n_timepoints_per_block = 8  # number of timpoints per block

        # there are 203 timepoints in each run: 3 fix, 192 stim, 8 fix
        self.n_timepoints_per_run = (
            3 + self.n_blocks_per_run * self.n_timepoints_per_block + 8
        )

        # regressors for experimental conditions, excluding fixation
        self.n_regressors = 6

        # number of additional regressors: 7 -> 6 nuissance_reg + 1 constant
        # 6 nuissance regressors: 3 translation (in x, y, and z) and 3 rotation
        self.n_nuissance_reg = 7

    def load_rtc_file(self, sbjID, run):
        """
        load rtc file to get nuissance regressors

        Parameters
        ----------
        sbj : TYPE
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.

        Returns
        -------
        rtc : TYPE
            DESCRIPTION.

        """

        # sbjID = self.sbjID_all[sbj]

        if (run + 1) < 10:
            runID = "0" + str(run + 1)
        elif (run + 1) >= 10:
            runID = str(run + 1)

        print("Loading rtc file: {}_{}_design_{}.rtc".format(sbjID, sbjID, runID))

        rtc = np.loadtxt(
            "../../../../Data/rtc/{}/{}_{}_design_{}.rtc".format(
                sbjID, sbjID, sbjID, runID
            ),
            skiprows=6,
        )

        return rtc

    def load_vtc(self, sbj):

        sbjID = self.sbjID_all[sbj]
        nRuns = self.nRuns_all[sbj]

        col_names = ["roi", "run", "timepoint", "vox", "vtc_value"]
        vtc_list = []
        for roi in range(self.n_ROIs):
            # for roi in range(n_ROIs):

            # load vrc mat file [run, timepoint, voxel]
            vtc_mat = sio.loadmat(
                "../../../Data/VTC_extract_smoothed/vtc_{}_{}.mat".format(
                    sbjID, self.ROIs[roi]
                )
            )["vtc_extract"]

            nVox = np.size(vtc_mat, 2)
            N = nRuns * self.n_timepoints_per_run * nVox
            # N = nRuns * n_timepoints_per_run * nVox
            vtc_roi = np.zeros(
                (N, 5), dtype=np.float32
            )  # [roi, run, timepoints, vox, vtc_value]

            count = 0
            for run in range(nRuns):

                for t in range(self.n_timepoints_per_run):
                    # for t in range(n_timepoints_per_run):

                    print(
                        "Reading vtc, sbjID={}, ROI={}, run={}, t={}".format(
                            sbjID, self.ROIs[roi], run + 1, t + 1
                        )
                    )

                    id_start = count
                    id_end = id_start + nVox

                    vtc_roi[id_start:id_end, 0] = roi
                    vtc_roi[id_start:id_end, 1] = run
                    vtc_roi[id_start:id_end, 2] = t
                    vtc_roi[id_start:id_end, 3] = np.arange(nVox)
                    vtc_roi[id_start:id_end, 4] = vtc_mat[run, t]

                    count = id_end

            vtc_roi_df = pd.DataFrame(vtc_roi, columns=col_names)

            vtc_list.append(vtc_roi_df)

            # roi = 7
            # run = 1
            # vox = 2
            # a = vtc_df.loc[(vtc_df.roi==roi) &
            #                (vtc_df.run==run) &
            #                (vtc_df.vox==vox)]

            # b = vtc_mat[run,:,vox]

        vtc_df = pd.concat(vtc_list)

        return vtc_df

    def load_vtc_normalized_vox(self, sbjID, nVox_to_analyze):
        """
        load vtc data that has been labeled.

        Parameters
        ----------
        sbjID : string
            participant's id.

        Returns
        -------
        vtc_norm : pd.DataFrame, [roi, run, rep, vox, stimID, cond, vtc_norm]
            normalized vtc data.

        """

        # [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        vtc_norm = pd.read_pickle(
            "../../../Data/VTC_normalized/vtc_norm_{}_{}".format(sbjID, nVox_to_analyze)
        )

        return vtc_norm

    def load_vtc_normalized(self, sbjID, nVox_to_analyze):
        """
        load vtc data that has been shifted backward by 2TR and z-scored.

        Parameters
        ----------
        sbjID : string
            participant's id.

        Returns
        -------
        vtc_norm : pd.DataFrame, [roi, run, rep, vox, stimID, cond, vtc_norm]
            normalized vtc data.

        """

        # [roi, vox, stimID, cond, run, rep, vtc_norm]
        vtc_norm = pd.read_pickle(
            "../../../Data/VTC_normalized/vtc_shift_norm_{}_{}.pkl".format(
                sbjID, nVox_to_analyze
            )
        )

        return vtc_norm

    def load_vtc_labeled(self, sbjID):
        """
        load vtc data that has been labeled.

        Parameters
        ----------
        sbjID : string
            participant's id.

        Returns
        -------
        vtc_labeled : pd.dataframe, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
            vtc data that has been labeled.

        """

        # [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        vtc_labeled = pd.read_pickle(
            "../../../Data/VTC_labeled/vtc_labeled_{}".format(sbjID)
        )

        return vtc_labeled

    def label_vtc_at_run(self, vtc, vtc_stimID, run):
        """
        label vtc dataset at a run with the stimID

        there are 203 timepoints in each run: 3 fix, 192 stim, 8 fix

        Parameters
        ----------
        vtc : pd.DataFrame
            ["run", "roi", "timepoint", "vox", "vtc_value"].

        vtc_stimID : [26, nRuns] np.array
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.

        Returns
        -------
        vtc_run : TYPE
            DESCRIPTION.

        """
        # filter dataframe associcated with run
        vtc_run = vtc.loc[(vtc.run == run)]

        # allocate a column for stimulus label
        vtc_run = vtc_run.assign(stimID=0)
        vtc_run = vtc_run.assign(cond="fix")

        # allocate a column for repetition label
        vtc_run = vtc_run.assign(rep=0)

        # allocate a column for blockID
        # vtc_run = vtc_run.assign(blockID=0)

        # label fixation, t<=3 and t>=196
        # vtc_run.loc[vtc.timepoint.isin(range(4))]["label"] = 0

        # nBlocks_per_run = 24
        # nTimepoints_per_block = 8

        count_cond1 = 1
        count_cond2 = 1
        count_cond3 = 1
        count_cond4 = 1
        count_cond5 = 1
        count_cond6 = 1
        for t in range(1, self.n_blocks_per_run + 1):
            # for t in range(glm.n_blocks_per_run):
            t_start = 3 + (t - 1) * self.n_timepoints_per_block
            t_end = t_start + self.n_timepoints_per_block

            ## label stimulus ID
            stimID = vtc_stimID[t, run]

            vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "stimID"] = (
                stimID
            )
            cond = self.conds[stimID]
            # cond = glm.conds[stimID]
            vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "cond"] = cond

            ## label repetition
            # if stimID==0:
            #     vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = 0

            if stimID == 1:
                vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = (
                    count_cond1
                )
                count_cond1 = count_cond1 + 1

            elif stimID == 2:
                vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = (
                    count_cond2
                )
                count_cond2 = count_cond2 + 1

            elif stimID == 3:
                vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = (
                    count_cond3
                )
                count_cond3 = count_cond3 + 1

            elif stimID == 4:
                vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = (
                    count_cond4
                )
                count_cond4 = count_cond4 + 1

            elif stimID == 5:
                vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = (
                    count_cond5
                )
                count_cond5 = count_cond5 + 1

            elif stimID == 6:
                vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "rep"] = (
                    count_cond6
                )
                count_cond6 = count_cond6 + 1

            # label block ID
            # vtc_run.loc[vtc_run.timepoint.isin(range(t_start, t_end)), "blockID"] = t + 1

        return vtc_run

    def label_vtc(self, vtc, vtc_stimID):
        """
        label vtc dataset with stimID for the whole run.

        there are 203 timepoints in each run: 3 fix, 192 stim, 8 fix

        Parameters
        ----------
        vtc : pandas dataframe
                col_names = [roi, run, timepoint, vox, vtc_value]
            vtc data, obtained from self.load_vtc

        vtc_stimID : np.array
            stimulus timing parameters, give id to each condition.

            it is obtained from
            vtc_stimID = sio.loadmat("../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}"
                                 .format(sbjID))["paramIdFull"]

        Returns
        -------
        vtc_labeled : pandas dataframe
                    col_names = [roi, run, timepoint, vox, vtc_value, stimID, cond]
            DESCRIPTION.

        """

        nRuns = np.shape(vtc_stimID)[1]
        # t_start = timer()
        vtc_all = []
        vtc_all.append(
            Parallel(n_jobs=nRuns)(
                delayed(self.label_vtc_at_run)(vtc, vtc_stimID, run)
                for run in range(nRuns)
            )
        )

        # t_end = timer()
        # print(t_end - t_start) # ~ 10 sec, n_jobs=nRuns, backend="loky"

        # unpack
        vtc_labeled = pd.concat(vtc_all[0], ignore_index=True)

        return vtc_labeled

    def label_depth_in_vtc_data(self, vtc):
        """
        Make a column "depth" in vtc data to label the class "near" and "far".
        far=1, near=-1.

        """
        depth_label = vtc["stimID"] % 2 == 0
        vtc.loc[:, "depth"] = depth_label

        # rename depth label
        # far=1, near=-1
        vtc["depth"] = vtc["depth"].map({False: -1, True: 1})

        return vtc

    def _normalize_shift_save_vtc_sbj(self, nVox_to_analyze, sbj):
        """
        helper function for self.normalize_shift_save_vtc_all_sbjID

        Args:
            nVox_to_analyze (_type_): _description_
            sbj (_type_): _description_
        """

        sbjID = self.sbjID_all[sbj]
        t_stat_sbj = self.t_stat_all_sbjID[sbj]

        print("normalize, shift and save vtc, sbjID: {}".format(sbjID))

        ## load stimulus timing parameters
        # vtc_stimID = sio.loadmat(
        #     "../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        # )["paramIdFull"]

        ## load vtc file
        # vtc = self.load_vtc(sbj)  # ["run", "roi", "timepoint", "vox", "vtc_value"].
        # vtc = mvpa.load_vtc(sbj)

        ## label vtc data, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        # vtc_labeled = self.label_vtc(vtc, vtc_stimID)
        # vtc_labeled = mvpa.label_vtc(vtc, vtc_stimID)

        # safe vtc_labeled
        # vtc_labeled.to_pickle("../../../Data/VTC_labeled/vtc_labeled_{}"
        #                       .format(sbjID))

        # load vtc_labeled
        vtc_labeled = self.load_vtc_labeled(sbjID)

        ## normalize vtc : z-score and then time-shifted 2TR
        vtc_normalized = self.normalize_vtc(vtc_labeled, t_stat_sbj, nVox_to_analyze)

        # safe the normalized and shifted vtc
        vtc_normalized.to_pickle(
            "../../../../Data/VTC_normalized/vtc_norm_shift_{}_{}.pkl".format(
                sbjID, nVox_to_analyze
            )
        )

    def normalize_shift_save_vtc_all_sbjID(self, nVox_to_analyze):
        """
        z-score normalize, shift by 2TR and save vtc files for all sbjID

        Args:
            nVox_to_analyze (_type_): _description_
        """

        Parallel(n_jobs=11)(
            delayed(self._normalize_shift_save_vtc_sbj)(nVox_to_analyze, sbj)
            for sbj in range(self.n_sbjID)
        )

    def _shift_normalize_save_vtc_sbj(self, t_stat_all_sbjID, nVox_to_analyze, sbj):
        """
        helper function for self.shift_normalize_save_vtc_all_sbjID

        Args:
            nVox_to_analyze (_type_): _description_
            sbj (_type_): _description_
        """

        # sbj = 0
        # sbjID = "AI"
        sbjID = self.sbjID_all[sbj]
        t_stat_sbj = t_stat_all_sbjID[sbj]

        print("shift, normalize and save vtc, sbjID: {}".format(sbjID))

        ## load stimulus timing parameters
        # vtc_stimID = sio.loadmat(
        #     "../../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        # )["paramIdFull"]

        ## load vtc file
        # vtc = self.load_vtc(sbj)  # ["run", "roi", "timepoint", "vox", "vtc_value"].
        # vtc = mvpa.load_vtc(sbj)

        ## label vtc data, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        # vtc_labeled = self.label_vtc(vtc, vtc_stimID)
        # vtc_labeled = mvpa.label_vtc(vtc, vtc_stimID)

        # safe vtc_labeled
        # vtc_labeled.to_pickle("../../../Data/VTC_labeled/vtc_labeled_{}"
        #                       .format(sbjID))

        # load vtc_labeled
        vtc_labeled = self.load_vtc_labeled(sbjID)
        # vtc_labeled = glm.load_vtc_labeled(sbjID)

        ## normalize vtc : time-shifted 2TR and then z-score
        vtc_normalized = self.shift_and_normalize_vtc(
            vtc_labeled, t_stat_sbj, nVox_to_analyze
        )
        # vtc_normalized = glm.shift_and_normalize_vtc(
        #     vtc_labeled, t_stat_sbj, nVox_to_analyze
        # )

        # safe the shifted and normalized vtc
        vtc_normalized.to_pickle(
            "../../../../Data/VTC_normalized/vtc_shift_norm_{}_{}.pkl".format(
                sbjID, nVox_to_analyze
            )
        )

    def shift_normalize_save_vtc_all_sbjID(self, t_stat_all_sbjID, nVox_to_analyze):
        """
        shift by 2TR, z-score normalize, and save vtc files for all sbjID

        Args:
            nVox_to_analyze (_type_): _description_
        """

        Parallel(n_jobs=11)(
            delayed(self._shift_normalize_save_vtc_sbj)(
                t_stat_all_sbjID, nVox_to_analyze, sbj
            )
            for sbj in range(self.n_sbjID)
        )

    def shift_vtc_roi(self, vtc_normalized, t_stat_sbj, nVox_to_analyze, roi):

        ## get t_stat
        t_stat_roi = t_stat_sbj[self.ROIs[roi]]

        ## sort voxel in t_stat descending order
        # a  = np.random.rand(10)
        # id_sort = a.argsort()[::-1]
        # a[id_sort]
        id_sort = t_stat_roi.argsort()[::-1]

        # check if nVox_to_analyze < nVox in a ROI
        if nVox_to_analyze > len(id_sort):
            nVox_to_analyze = len(id_sort)

        # get the first nVox_to_analyze voxel id based on t-stat
        # vox_to_use = id_sort[:nVox_to_analyze]

        vtc_vox_all = []
        for v in range(nVox_to_analyze):

            vox = id_sort[v]
            vtc_vox = vtc_normalized.loc[
                (vtc_normalized.roi == roi) & (vtc_normalized.vox == vox)
            ]

            # shift by 2 TR
            temp = vtc_vox.vtc_norm.shift(-2, fill_value=0)
            vtc_vox = vtc_vox.assign(vtc_norm=temp)

            # rename the vox to order the voxel according to t-stat
            vtc_vox = vtc_vox.assign(vox=v)

            vtc_vox_all.append(vtc_vox)

        vtc_roi_df = pd.concat(vtc_vox_all, ignore_index=True)

        return vtc_roi_df

    def normalize_vtc(self, vtc_labeled, t_stat_sbj, nVox_to_analyze):
        """
        Normalize vtc time series data and arrange the voxel in a descending
        order according to the t_stat_sbj.

        The normalization is computed by first z-score for each voxel in
        each roi and run. After that, the time-series is shifted for 2 TR
        to account for the hrf delay. And then perform the second "normalization"
        by subtracting the z-score vtc time series with the "global mean"
        (the average across voxel for each roi, run, rep).

        Parameters
        ----------
        vtc_labeled : pd.DataFrame, [roi, run, timepoint, vox, vtc_value, stimID,
                                     cond, rep]
            vtc time series data.

        t_stat_sbj: t-stat of a given sbjID

        Returns
        -------
        vtc_normalized : pd.DataFrame, [roi, vox, stimID, cond, run, rep, vtc_norm]
            normalized vtc.

        """

        # normalize with time-shifted 2TR to account for hrf delay.

        ## z-score fmri time-series for each roi, run, vox
        avg = vtc_labeled.groupby(["roi", "run", "vox"]).vtc_value.transform("mean")
        std = vtc_labeled.groupby(["roi", "run", "vox"]).vtc_value.transform(np.std)
        temp = (vtc_labeled.vtc_value - avg) / std

        vtc_normalized = vtc_labeled.assign(vtc_norm=temp)

        ## shift the time course by 2TRs to account for the hemodynamic delay
        # t_start = timer()
        vtc_roi_all = []
        vtc_roi_all.append(
            Parallel(n_jobs=self.n_ROIs)(
                delayed(self.shift_vtc_roi)(
                    vtc_normalized, t_stat_sbj, nVox_to_analyze, roi
                )
                for roi in range(self.n_ROIs)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        vtc_roi_df = pd.concat(vtc_roi_all[0])

        # remove fixation condition
        vtc_roi_df = vtc_roi_df.drop(vtc_roi_df[vtc_roi_df.stimID == 0].index)

        # average across timepoint (8 TR) for each roi, vox, run, rep, cond
        vtc_avg = (
            vtc_roi_df.groupby(["roi", "vox", "stimID", "cond", "run", "rep"])
            .vtc_norm.agg(["mean"])
            .reset_index()
        )
        vtc_avg = vtc_avg.rename(columns={"mean": "vtc_norm"})

        # normalize again with respect to global mean (average across voxel
        # for each roi, run, rep)
        avg_global = vtc_avg.groupby(["roi", "run", "rep"]).vtc_norm.transform("mean")
        temp = vtc_avg.vtc_norm - avg_global
        vtc_normalized = vtc_avg.assign(vtc_norm=temp)

        return vtc_normalized

    def shift_and_normalize_vtc_roi(
        self, vtc_labeled, t_stat_sbj, nVox_to_analyze, roi
    ):

        ## get t_stat
        # roi = 0
        # t_stat_roi = t_stat_sbj[glm.ROIs[roi]]
        t_stat_roi = t_stat_sbj[self.ROIs[roi]]

        ## sort voxel in t_stat descending order
        # a  = np.random.rand(10)
        # id_sort = a.argsort()[::-1]
        # a[id_sort]
        id_sort = t_stat_roi.argsort()[::-1]

        # check if nVox_to_analyze < nVox in a ROI
        if nVox_to_analyze > len(id_sort):
            nVox_to_analyze = len(id_sort)

        # get the first nVox_to_analyze voxel id based on t-stat
        # vox_to_use = id_sort[:nVox_to_analyze]

        vtc_vox_all = []
        for v in range(nVox_to_analyze):

            vox = id_sort[v]
            vtc_vox = vtc_labeled.loc[
                (vtc_labeled.roi == roi) & (vtc_labeled.vox == vox)
            ]

            # shift backward by 2 TR
            temp = vtc_vox.vtc_value.shift(-2, fill_value=0)
            vtc_vox = vtc_vox.assign(vtc_value=temp)

            # rename the vox to order the voxel according to t-stat
            vtc_vox = vtc_vox.assign(vox=v)

            vtc_vox_all.append(vtc_vox)

        vtc_roi_df = pd.concat(vtc_vox_all, ignore_index=True)

        return vtc_roi_df

    def shift_and_normalize_vtc(self, vtc_labeled, t_stat_sbj, nVox_to_analyze):
        """
        Normalize vtc time series data and arrange the voxel in a descending
        order according to the t_stat_sbj.

        The normalization is computed by first shifted backward for 2 TR
        to account for the hrf delay. After that, z-score for each voxel in
        each roi and run. And then perform the second "normalization"
        by subtracting the z-score vtc time series with the "global mean"
        (the average across voxel for each roi, run, rep).

        Parameters
        ----------
        vtc_labeled : pd.DataFrame, [roi, run, timepoint, vox, vtc_value, stimID,
                                     cond, rep]
            vtc time series data.

        t_stat_sbj: t-stat of a given sbjID

        Returns
        -------
        vtc_normalized : pd.DataFrame, [roi, vox, stimID, cond, run, rep, vtc_norm]
            normalized vtc.

        """

        ## shift the time course backward by 2TRs to account for the hemodynamic delay
        # t_start = timer()
        vtc_roi_all = []
        vtc_roi_all.append(
            Parallel(n_jobs=self.n_ROIs)(
                delayed(self.shift_and_normalize_vtc_roi)(
                    vtc_labeled, t_stat_sbj, nVox_to_analyze, roi
                )
                for roi in range(self.n_ROIs)
            )
        )
        vtc_roi_df = pd.concat(vtc_roi_all[0])

        # vtc_roi_all = []
        # vtc_roi_all.append(
        #     Parallel(n_jobs=glm.n_ROIs)(
        #         delayed(glm.shift_and_normalize_vtc_roi)(
        #             vtc_labeled, t_stat_sbj, nVox_to_analyze, roi
        #         )
        #         for roi in range(glm.n_ROIs)
        #     )
        # )

        # t_end = timer()
        # print(t_end - t_start)
        # vtc_roi_df = pd.concat(vtc_roi_all[0])

        ## z-score fmri time-series for each roi, run, vox
        avg = vtc_roi_df.groupby(["roi", "run", "vox"]).vtc_value.transform("mean")
        std = vtc_roi_df.groupby(["roi", "run", "vox"]).vtc_value.transform(np.std)
        temp = (vtc_roi_df.vtc_value - avg) / std
        vtc_roi_df = vtc_roi_df.assign(vtc_norm=temp)

        # remove fixation condition
        vtc_roi_df = vtc_roi_df.drop(vtc_roi_df[vtc_roi_df.stimID == 0].index)

        # average across timepoint (8 TR) for each roi, vox, run, rep, cond
        vtc_avg = (
            vtc_roi_df.groupby(["roi", "vox", "stimID", "cond", "run", "rep"])
            .vtc_norm.agg(["mean"])
            .reset_index()
        )
        vtc_avg = vtc_avg.rename(columns={"mean": "vtc_norm"})

        # normalize again with respect to global mean (average across voxel
        # for each roi, run, rep)
        avg_global = vtc_avg.groupby(["roi", "run", "rep"]).vtc_norm.transform("mean")
        temp = vtc_avg.vtc_norm - avg_global
        vtc_normalized = vtc_avg.assign(vtc_norm=temp)

        return vtc_normalized

    def normalize_vtc_roi(self, vtc_roi, mtd_normalization=2):
        """
        normalize voxel values in vtc_roi (within an ROI) according to mtd_normalization

        Parameters
        -------
        vtc_roi : pandas dataframe
                    col_names = ["run", "roi", "timepoint", "vox", "vtc_value"]

            this dataframe is obtained from vtc data as follows:

            # load vtc file
            sbjID = "AI"
            col_names = ["run", "roi", "timepoint", "vox", "vtc_value"]
            vtc = pd.read_csv("../../../../Data/VTC_extract/vtc_{}.csv"
                              .format(sbjID),
                              names=col_names)

            # label vtc data
            ["roi", "run", "timepoint", "vox", "vtc_value", "stimID", "cond", "rep"]
            vtc_labeled = glm.label_vtc(vtc, vtc_stimID)
            vtc_labeled = glm.load_vtc_labeled(sbjID)

            # filter dataframe according to roi and nVox
            ["roi", "run", "timepoint", "vox", "vtc_value", "stimID", "cond", "rep"]
            vtc_roi = vtc_labeled.loc[(vtc_labeled.roi==roi)]

        mtd_normalization : scalar, either 0, 1, or 2.
                            default = 2
            the normalization method:
                - 0 : z-score
                - 1 : percent-bold-change with respect to rest baseline
                - 2 : percent-bold-change with respect to avg across timepoint
                        (default)

        Returns
        -------
        y : [n_timepoints, vox] np.array
            normalized voxel values.

        """

        vtc_copy = vtc_roi.copy()

        if mtd_normalization == -1:  # no normalization
            y = np.array(
                vtc_copy.pivot_table(
                    index=["run", "timepoint"], columns="vox", values="vtc_value"
                ),
                dtype=np.float32,
            )

        elif mtd_normalization == 0:  # z-score

            # average across timepoint in each run
            avg = vtc_copy.groupby(["roi", "run", "vox"]).vtc_value.transform("mean")
            std = vtc_copy.groupby(["roi", "run", "vox"]).vtc_value.transform(np.std)

            # avg = (vtc_copy.groupby(["roi", "vox"])["vtc_value"]
            #                               .transform("mean"))
            # std = (vtc_copy.groupby(["roi", "vox"])["vtc_value"]
            #                               .transform(np.std))
            # vtc_roi = vtc_roi.assign(vtc_norm = avg)

            # standardize (z-score)
            temp = (vtc_copy.vtc_value - avg) / std
            vtc_copy = vtc_copy.assign(vtc_norm=temp)

            # construct array of voxel pattern
            y = np.array(
                vtc_copy.pivot_table(
                    index=["run", "timepoint"], columns="vox", values="vtc_norm"
                ),
                dtype=np.float32,
            )

        elif (
            mtd_normalization == 1
        ):  # percent-bold-change with respect to rest-baseline

            # find baseline (fixation) signal
            temp = vtc_copy.groupby(["roi", "run", "vox", "cond"])[
                "vtc_value"
            ].transform("mean")
            vtc_copy = vtc_copy.assign(avg=temp)

            # create baseline dataframe
            baseline = vtc_copy.loc[
                (vtc_copy.stimID == 0) & (vtc_copy.timepoint == 1),
                ["roi", "run", "vox", "avg"],
            ]
            baseline = baseline.rename(columns={"avg": "baseline"})

            # merge dataframe
            merge_df = pd.merge(vtc_copy, baseline, on=["roi", "run", "vox"])
            delta = (merge_df.vtc_value - merge_df.baseline) / merge_df.baseline * 100
            merge_df = merge_df.assign(vtc_norm=delta)

            # a = merge_df.loc[(merge_df.vox==1) &
            #                  (merge_df.run==1)]

            # plt.plot(a.bold_change)

            # construct array of voxel pattern
            y = np.array(
                merge_df.pivot_table(
                    index=["run", "timepoint"], columns="vox", values="bold_change"
                ),
                dtype=np.float32,
            )

        elif (
            mtd_normalization == 2
        ):  # percent-bold-change relative to the avg across timepoint
            # for each roi, run, and vox

            # average across timepoint for each roi, run, and vox
            avg = vtc_copy.groupby(["roi", "run", "vox"]).vtc_value.transform("mean")
            # avg2 = (vtc_copy.groupby(["roi", "vox"])["vtc_value"]
            #                 .transform("mean"))
            # vtc_roi = vtc_roi.assign(vtc_norm = avg)

            # normalize
            temp = (vtc_copy.vtc_value - avg) / avg * 100
            vtc_copy = vtc_copy.assign(vtc_norm=temp)

            # construct array of voxel pattern, [timepoint, vox]
            y = np.array(
                vtc_copy.pivot_table(
                    index=["run", "timepoint"], columns="vox", values="vtc_norm"
                ),
                dtype=np.float32,
            )

        elif (
            mtd_normalization == 3
        ):  # z-score across timepoint for each roi, run, vox, cond

            # average across timepoint for each roi, run, vox, cond
            avg = vtc_copy.groupby(
                ["roi", "run", "vox", "stimID", "cond"]
            ).vtc_value.transform("mean")
            std = vtc_copy.groupby(
                ["roi", "run", "vox", "stimID", "cond"]
            ).vtc_value.transform(np.std)

            # standardize z-score
            temp = (vtc_copy.vtc_value - avg) / std
            vtc_copy = vtc_copy.assign(vtc_norm=temp)

            # construct array of voxel pattern
            y = np.array(
                vtc_copy.pivot_table(
                    index=["run", "timepoint"], columns="vox", values="vtc_norm"
                ),
                dtype=np.float32,
            )

        return y

    def load_P_data(self, sbjID, nRuns):
        """
        load P_data.

        # conds: Fix, n_d25_c0, f_d25_c0,
        #               n_d25_c50, f_d25_c50,
        #               n_d25_c100, f_d25_c100
        #
        # 1. n_d25_c0 vs f_d25_c0         : [2,3]
        # 3. n_d25_c50 vs f_d25_c50       : [4,5]
        # 4. n_d25_c100 vs f_d25_c100     : [6,7]



        % =====================================================================
        % IMPORTANT NOTES:
        % The condition's indexes used for SVM analysis with Python should
        % follow these rules:
        %
        %       - "fix" is denoted as 1
        %       - "n_d25_c0" is denoted as 2
        %       - "f_d25_c0" is denoted as 3,
        %       - so on
        % conds = ['fix', 'n_d25_c0', 'f_d25_c0',
        %                 'n_d25_c50', 'f_d25_c50',
        %                 'n_d25_c100', 'f_d25_c100']
        %
        % In P_data["cond"], index 1 refers to "fix" condition
        %                   index 2 refers to "n_d6_c0", and so on
        % =====================================================================

        """
        colNames = ["roi", "cond", "rep", "vox", "run", "voxVal"]

        P_data = pd.read_csv(
            "../../../../Data/P_Matrix/P_Matrix_%s.csv" % sbjID,
            sep=",",
            header=None,
            names=colNames,
        )

        return P_data

    def label_P_data(self, P_data):
        """
        Make a column "depth" in P_data to label the class "near" and "far".
        far=1, near=-1.

        """

        P_temp = P_data.copy()
        P_temp["label"] = P_temp["cond"] % 2 == 0

        # rename depth label
        # far=1, near=-1
        P_temp["label"] = P_temp["label"].map({False: 1, True: -1})

        P_temp["depth"] = P_temp["cond"]

        # assign the depth label to the original dataframe
        P_data = P_temp.copy()

        return P_data

    def normalize_P_data(self, P_data):
        """
        Do feature standardization, zero mean and unit variance.

        Two kinds of normalization performed here:
            1. Normalization across stimuli for each voxel (method="acrStim" -> default)
                The averaging way in this scheme is done for each voxel. So, for a
                given voxel, the pattern responses are averaged in that voxel.
                Therefore, it seems that this kind of averaging does not take ROI
                into account.
            2. Normalization across voxel for each stimulus (method="acrVox")
                The averaging way in this scheme is done in each ROI for a given stimulus.
                So, for a given stimulus, the voxel responses are averaged in that
                ROI. This kind of averaging seems to be similar to spatial averaging.
            3. Normalization by subtracting each voxel with a global mean (method=
                "global". The global mean here means the average voxel values
                of a scan-run. Each voxel will be subtracted with the global mean

        (Normalization method no. 1 & 2 are based on Misaki, et.al,
             Neuroimage 2010: Comparison of multivariate classifiers
             and response normalizations for pattern-information fMRI)
        Though the paper claims that there is no significant difference betwee
        "acrStim" and "acrVox method, but the default method used in
        this script is "across-stimuli" for each voxel in order to conserve the response
        pattern in voxel space (see Fig. 2 and Table 2 in the paper).
        """

        P_temp = P_data.copy()

        ## remove baseline
        mean_cond = P_temp.groupby(["roi", "cond", "vox", "run"]).voxVal.transform(
            "mean"
        )
        P_temp["mean_cond"] = mean_cond

        P_fix = P_temp.loc[(P_temp.cond == 1) & (P_temp.rep == 1)]

        P_fix = P_fix.drop(["cond", "rep", "voxVal", "depth", "label"], 1)

        P_temp2 = P_data.loc[P_data.cond != 1]
        P_temp = pd.merge(P_temp2, P_fix, on=["roi", "vox", "run"])

        P_temp["voxVal_noBaseline"] = P_temp.voxVal - P_temp.mean_cond

        ## z-standardize across all runs
        # mean_run = (P_temp.groupby(["roi", "vox"])
        #                     .voxVal_noBaseline
        #                     .transform("mean"))
        # std_run = (P_temp.groupby(["roi", "vox"])
        #                    .voxVal_noBaseline
        #                    .transform(np.std))
        # P_temp["voxVal_norm"] = (P_temp.voxVal_noBaseline - mean_run)/std_run

        # Normalization "across-stimuli" for each voxel
        # This normalization is based on
        # Brouwer, Journal of Neurophysiology 2011: Cross-orientation suppression in human visual cortex
        # for each run, do
        # unit normalization by removing the baseline: B = B - m(m.T * B)
        # B = [voxel, condition], vector of response amplitudes
        # m = [voxel], mean vector of B, average across all stimulus condition ->
        #                                normalized to a unit vector (divided by the norm)

        mean_acrStim = P_temp.groupby(
            ["roi", "vox", "run"]
        ).voxVal_noBaseline.transform("mean")
        P_temp["voxVal_mean"] = mean_acrStim
        norm = P_temp.groupby(["roi", "run", "rep"]).voxVal_mean.transform(
            np.linalg.norm
        )

        a = P_temp["voxVal_mean"] / norm
        P_temp["voxVal_acrStim"] = P_temp.voxVal - a * a * P_temp.voxVal

        ## z-standardize across all runs
        mean_run = P_temp.groupby(["roi", "vox"]).voxVal_acrStim.transform("mean")
        std_run = P_temp.groupby(["roi", "vox"]).voxVal_acrStim.transform(np.std)
        P_temp["voxVal_norm"] = (P_temp.voxVal_noBaseline - mean_run) / std_run

        # roi = 1
        # cond = 2
        # run = 1
        # a = P_temp.loc[(P_temp.roi==roi) &
        #                (P_temp.cond==cond) &
        #                (P_temp.run==run)].voxVal_norm
        # b = P_temp.loc[(P_temp.roi==roi) &
        #                (P_temp.cond==cond) &
        #                (P_temp.run==run)].voxVal_noBaseline

        # roi = 1
        # cond = 3
        # run = 1
        # c = P_temp.loc[(P_temp.roi==roi) &
        #                (P_temp.cond==cond) &
        #                (P_temp.run==run)].voxVal_norm
        # d = P_temp.loc[(P_temp.roi==roi) &
        #                (P_temp.cond==cond) &
        #                (P_temp.run==run)].voxVal_noBaseline

        # Normalization "across-voxels" for each stimulus/condition
        # mean_acrVox = (P_temp.groupby(["cond", "run"])
        #                     .voxVal
        #                     .transform("mean"))
        # std_acrVox = (P_temp.groupby(["cond", "run"])
        #                     .voxVal
        #                     .transform(np.std))
        # P_temp["voxVal_acrVox"] = (P_temp.voxVal - mean_acrVox)/std_acrVox

        P_data = P_temp.copy()

        return P_data

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

    def _compute_kendalltau_up_single_roi(
        self, rdm_fmri_all, sbjID_bootsrap, n_bootstrap, roi
    ):
        """
        inner function for compute_noiseCeiling.
        it calculates the upper bound of noise ceiling

        Parameters
        ----------
        rdm_fmri_all : TYPE
            DESCRIPTION.
        sbjID_bootsrap : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.
        roi : TYPE
            DESCRIPTION.

        Returns
        -------
        kendalltau_up_roi : [n_roi, n_bootstrap]
            DESCRIPTION.

        """

        rdm_fmri_roi = np.mean(rdm_fmri_all, axis=0)[roi]
        # get above diagonal element
        rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

        kendalltau_up_roi = np.zeros(n_bootstrap, dtype=np.float32)
        for i in range(n_bootstrap):

            id_sample = np.random.choice(sbjID_bootsrap, size=len(sbjID_bootsrap))
            rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]

            rdm_boot_mean = np.mean(rdm_fmri_bootstrap, axis=0)

            # get above diagonal element
            rdm_boot_above = rdm_boot_mean[np.triu_indices(6, k=1)]

            kendalltau_up_roi[i] = kendalltau(rdm_fmri_above, rdm_boot_above)[0]

        return kendalltau_up_roi

    def _compute_kendalltau_low_single_roi(
        self, rdm_fmri_all, sbjID_bootsrap, n_bootstrap, sbj, roi
    ):
        """
        inner function for compute_noiseCeiling.
        it calculates the lower bound of noise ceiling.


        Parameters
        ----------
        rdm_fmri_all : TYPE
            DESCRIPTION.
        sbjID_bootsrap : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.
        sbj : TYPE
            DESCRIPTION.
        roi : TYPE
            DESCRIPTION.

        Returns
        -------
        kendalltau_low_roi : [n_roi, n_bootstrap, n_sbj]
            DESCRIPTION.

        """

        rdm_fmri_roi = rdm_fmri_all[sbj, roi]
        # get above diagonal
        rdm_fmri_above = rdm_fmri_roi[np.triu_indices(6, k=1)]

        kendalltau_low_roi = np.zeros(n_bootstrap, dtype=np.float32)
        for i in range(n_bootstrap):

            id_sample = np.random.choice(sbjID_bootsrap, size=len(sbjID_bootsrap))
            rdm_fmri_bootstrap = rdm_fmri_all[id_sample, roi]

            rdm_boot_mean = np.mean(rdm_fmri_bootstrap, axis=0)

            # get above diagonal element
            rdm_boot_above = rdm_boot_mean[np.triu_indices(6, k=1)]

            kendalltau_low_roi[i] = kendalltau(rdm_fmri_above, rdm_boot_above)[0]

        return kendalltau_low_roi

    def compute_noiseCeiling(self, rdm_fmri_all, n_bootstrap):
        """
        compute the lower and upper bound of noise ceiling.
        based on:
            Nili, et.al, plos 2014
            Ban, et.al, journal of neuroscience 2015


        Parameters
        ----------
        rdm_fmri_all : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.

        Returns
        -------
        kendalltau_up : [n_roi, n_bootstrap]
            DESCRIPTION.
        kendalltau_low : [n_roi, n_bootstrap, n_sbj]
            DESCRIPTION.

        """

        n_sbj, n_roi, _, _ = rdm_fmri_all.shape

        # calculating upper bound
        print("compute upper bound noise ceiling")
        sbjID_bootsrap = np.arange(n_sbj)
        temp = []
        temp.append(
            Parallel(n_jobs=8)(
                delayed(self._compute_kendalltau_up_single_roi)(
                    rdm_fmri_all, sbjID_bootsrap, n_bootstrap, roi
                )
                for roi in range(n_roi)
            )
        )

        # unpack
        kendalltau_up = np.zeros((n_roi, n_bootstrap), dtype=np.float32)
        for roi in range(n_roi):
            kendalltau_up[roi] = temp[0][roi]

        # calculating lower bound
        kendalltau_low = np.zeros((n_roi, n_bootstrap, n_sbj), dtype=np.float32)
        sbjID_all = np.arange(n_sbj)

        for sbj in range(n_sbj):
            print(
                "compute lower bound noise ceiling, sbj_out: {}/{}".format(
                    str(sbj + 1), str(n_sbj)
                )
            )

            sbjID_bootsrap = sbjID_all[sbjID_all != sbj]

            temp = []
            temp.append(
                Parallel(n_jobs=8)(
                    delayed(self._compute_kendalltau_low_single_roi)(
                        rdm_fmri_all, sbjID_bootsrap, n_bootstrap, sbj, roi
                    )
                    for roi in range(n_roi)
                )
            )

            # unpack
            for roi in range(n_roi):
                kendalltau_low[roi, :, sbj] = temp[0][roi]

        return kendalltau_low, kendalltau_up


class PlotGeneral(General):
    def __init__(self):

        super().__init__()
        self.homedir_addr = (
            "../../../../Plots/CMM_v2"  # home directory address for saving plots
        )

        # roi name for plotting
        self.ROI_plotname = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "hMT+"]

        # cmap = sns.diverging_palette(255, 10, s=99, l=40, n=8, as_cmap=True)

        all_barcolor = (
            135 / 255,
            231 / 255,
            176 / 255,
        )  # barplot color for avg across all sbjID

        # crossedLabel_barcolor = (127/255,135/255,143/255) # barplot color for test label: crossed RDS
        # uncrossedLabel_barcolor = (200/255,200/255,203/255) # barplot color for test label: uncrossed RDS

        self.all_linecolor = all_barcolor  # lineplot color for avg across all sbjID

        # ards_linecolor = (154/255,0/255,121/255) # lineplot color for ards
        # hmrds_linecolor = (53/255,161/255,107/255) # lineplot color for hmrds
        # crds_linecolor = (0/255,65/255,255/255) # lineplot color for crds
        line_color_val = np.logspace(-1, 0, 8)
        line_color_val = (line_color_val - np.min(line_color_val)) / (
            np.max(line_color_val) - np.min(line_color_val)
        )
        self.ards_linecolor = plt.cm.Greys(line_color_val[5])  # lineplot color for ards
        self.hmrds_linecolor = plt.cm.Greys(
            line_color_val[6]
        )  # lineplot color for hmrds
        self.crds_linecolor = plt.cm.Greys(line_color_val[7])  # lineplot color for crds

        # error_kw for barplot
        self.error_kw_bar = {"capsize": 15, "ecolor": "0.2", "elinewidth": 4}

        # properties for boxplot
        self.facecolors = [
            self.ards_linecolor,
            self.hmrds_linecolor,
            self.crds_linecolor,
            self.ards_linecolor,
            self.hmrds_linecolor,
            self.crds_linecolor,
            self.ards_linecolor,
            self.hmrds_linecolor,
            self.crds_linecolor,
        ]
        # boxprops = dict(linewidth=1, color='Black', facecolor = "black", alpha=0.6)
        self.boxprops = dict(
            linewidth=3, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        self.medianprops = dict(linestyle="-", linewidth=3, color="black")
        self.meanprops = dict(
            marker="D", markersize=17, markerfacecolor="black", markeredgecolor="black"
        )
        self.whiskerprops = dict(linewidth=3)
        self.capprops = dict(linewidth=3)

        # color data point in boxplot
        self.color_point = ["green", "orange", "magenta"]  # crds, hmrds, ards
