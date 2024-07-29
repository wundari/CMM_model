"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/MVPA/MVPA_XDecode.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/MVPA
Created Date: 2022-06-30 13:42:08
Author: Bayu G. Wundari
-----
Last Modified: 2022-06-30 13:51:54
Modified By: Bayu G. Wundari
-----
Copyright (c) 2022 National Institute of Information and Communications Technology (NICT)

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
cross-decoding analysis
"""

import numpy as np

# enable intel cpu optimization for scikit
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.svm import SVC

import scipy.io as sio
from scipy.stats import sem

from statsmodels.stats.anova import AnovaRM

import pandas as pd
from joblib import Parallel, delayed

from timeit import default_timer as timer
from itertools import permutations
from sklearn.model_selection import LeaveOneGroupOut

from Common.Common import General


class MVPA_XDecode(General):
    def __init__(self, beta_all_sbjID, t_stat_all_sbjID):
        super().__init__()

        ## get complete permutation of labels (for permutation test)
        label = [-1, -1, -1, -1, 1, 1, 1, 1]
        self.perms = np.array(
            list(set(permutations(label))), dtype=np.int32
        )  # randomized label

        self.conds = ["fix", "n_c0", "f_c0", "n_c50", "f_c50", "n_c100", "f_c100"]

        ## t_stat, list
        self.t_stat_all_sbjID = t_stat_all_sbjID

        ## beta values
        self.beta_all_sbjID = beta_all_sbjID

    def permuteXDecode_onePermute(self, X_train, X_test, run_group, nRuns):
        id_permute = np.random.choice(len(self.perms), nRuns, replace=False)

        y_permute = np.int32(np.reshape(self.perms[id_permute], -1))

        logo = LeaveOneGroupOut()

        score = []
        for id_train, id_test in logo.split(X_train, y_permute, groups=run_group):
            # define classifier
            model = SVC(kernel="linear", cache_size=50000)

            # get training and test dataset
            X_train2, X_test2 = X_train[id_train], X_test[id_test]
            y_train, y_test = y_permute[id_train], y_permute[id_test]

            # train model
            model.fit(X_train2, y_train)

            # test model
            score.append(model.score(X_test2, y_test))

        score_onePermute = np.mean(score, dtype=np.float32)

        return score_onePermute

    def permuteXDecode_roi(
        self,
        sbj,
        vtc_norm,
        t_stat_sbj,
        comp_pair_train,
        comp_pair_test,
        nVox_to_analyze,
        run_group,
        n_permute,
        roi,
    ):
        """


        Args:
            P_data (pd.DataFrame): DESCRIPTION.
            sbjID (str): DESCRIPTION.
            nRuns (int): DESCRIPTION.
            run_group (list): [i for i in range(nRuns) for j in range(8)].
            ROIs (list): DESCRIPTION.
            roi (int): id_roi, start from 1.
            nVox (TYPE): DESCRIPTION.
            useVoxVal (TYPE): DESCRIPTION.
            comp_pair_train (TYPE): DESCRIPTION.
            comp_pair_test (TYPE): DESCRIPTION.
            n_permute (TYPE): DESCRIPTION.

        Returns:
            score_permute_df (TYPE): DESCRIPTION.

        """

        ## get t_stat
        t_stat_roi = t_stat_sbj[self.ROIs[roi]]

        ## sort voxel in t_stat descending order
        # a  = np.random.rand(10)
        # id_sort = a.argsort()[::-1]
        # a[id_sort]
        id_sort = t_stat_roi.argsort()[::-1]

        # check if nVox_to_analyze < nVox_in_a_ROI
        if nVox_to_analyze > len(id_sort):
            nVox_to_analyze = len(id_sort)

        # # get the first nVox_to_analyze voxel id based on t-stat
        # vox_to_use = id_sort[:nVox_to_analyze]
        # # add voxel ranking
        # vox_id_sorted = np.vstack([vox_to_use, np.arange(nVox_to_analyze)])
        # # get id of unsorted voxel
        # vox_id_unsorted = vox_id_sorted[:, vox_id_sorted[0, :].argsort()]

        # comp_pair = np.array([1, 2]).astype(np.int32)

        ## prepare training dataset
        vtc_vox_train = vtc_norm.loc[
            (vtc_norm.roi == roi)
            & (vtc_norm.stimID.isin(comp_pair_train))
            & (vtc_norm.vox.isin(range(nVox_to_analyze)))
        ]

        X_train = np.array(
            vtc_vox_train.pivot_table(
                index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
            ),
            dtype=np.float32,
        )

        ## prepare test dataset
        vtc_vox_test = vtc_norm.loc[
            (vtc_norm.roi == roi)
            & (vtc_norm.stimID.isin(comp_pair_test))
            & (vtc_norm.vox.isin(range(nVox_to_analyze)))
        ]
        X_test = np.array(
            vtc_vox_test.pivot_table(
                index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
            ),
            dtype=np.float32,
        )

        score_permute = []
        score_permute.append(
            Parallel(n_jobs=1)(
                delayed(self.permuteXDecode_onePermute)(
                    X_train, X_test, run_group, self.nRuns_all[sbj]
                )
                for i in range(n_permute)
            )
        )

        score_permute_df = pd.DataFrame(score_permute[0], columns=["acc"])

        score_permute_df["sbjID"] = self.sbjID_all[sbj]
        score_permute_df["roi"] = roi
        score_permute_df["roi_str"] = self.ROIs[roi]
        score_permute_df["nVox"] = nVox_to_analyze
        score_permute_df["permute_i"] = np.arange(n_permute)

        comp_train_str = (
            self.conds[comp_pair_train[0]] + "," + self.conds[comp_pair_train[1]]
        )
        comp_test_str = (
            self.conds[comp_pair_test[0]] + "," + self.conds[comp_pair_test[1]]
        )

        score_permute_df["comp_train"] = comp_train_str
        score_permute_df["comp_test"] = comp_test_str

        return score_permute_df

    def permuteXDecode_allROI(
        self,
        sbj,
        t_stat_sbj,
        run_group,
        nVox_list,
        comp_pair_train,
        comp_pair_test,
        n_permute,
        vox_id,
    ):
        sbjID = self.sbjID_all[sbj]
        nVox_to_analyze = nVox_list[vox_id]

        # load vtc_norm, vtc data that has been shifted backward 2TR and z-scored
        vtc_norm = self.load_vtc_normalized(sbjID, nVox_to_analyze)

        ## add a column for depth label (near:-1 or far:1) in the vtc data
        vtc_norm = self.label_depth_in_vtc_data(vtc_norm)

        score_permute = []
        # t_start = timer()
        # print("run permutation test, sbjID: %s" %sbjID)
        score_permute.append(
            Parallel(n_jobs=1)(
                delayed(self.permuteXDecode_roi)(
                    sbj,
                    vtc_norm,
                    t_stat_sbj,
                    comp_pair_train,
                    comp_pair_test,
                    nVox_to_analyze,
                    run_group,
                    n_permute,
                    roi,
                )
                for roi in range(self.n_ROIs)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # concetenate all df
        score_permute_allROI_df = pd.concat(score_permute[0], ignore_index=True)

        return score_permute_allROI_df

    def permuteXDecode_sbj(
        self,
        sbj,
        comp_pair_train,
        comp_pair_test,
        nVox_list,
        n_permute,
    ):
        t_stat_sbj = self.t_stat_all_sbjID[sbj]

        run_group = [i for i in range(self.nRuns_all[sbj]) for j in range(8)]

        # start permutation test
        # t_start = timer()
        # print("run permutation test, sbjID: %s" %sbjID)

        score_permute = []
        score_permute.append(
            Parallel(n_jobs=2)(
                delayed(self.permuteXDecode_allROI)(
                    sbj,
                    t_stat_sbj,
                    run_group,
                    nVox_list,
                    comp_pair_train,
                    comp_pair_test,
                    n_permute,
                    vox_id,
                )
                for vox_id in range(len(nVox_list))
            )
        )

        score_permute_df = pd.concat(score_permute[0], ignore_index=True)

        # t_end = timer()
        # print(t_end-t_start)

        return score_permute_df

    def permuteXDecode_allSbjID(
        self, comp_pair_train, comp_pair_test, nVox_list, n_permute, rds_train_test
    ):
        """
        compute cross-decoding in two ways:
            train: comp_pair_train, test: comp_pair_test
            train: comp_pair_test, test: comp_pair_train

        the resulting cross-decoding performance of both train-test schemes are
        averaged.

        Parameters
        ----------
        sbjID_all : TYPE
            DESCRIPTION.
        nRuns_all : TYPE
            DESCRIPTION.
        ROIs : TYPE
            DESCRIPTION.
        nVox : TYPE
            DESCRIPTION.
        useVoxVal : TYPE
            DESCRIPTION.
        comp_pair_train : TYPE
            DESCRIPTION.
        comp_pair_test : TYPE
            DESCRIPTION.
        rds_train_test : TYPE
            DESCRIPTION.
        n_permute : TYPE
            DESCRIPTION.

        Returns
        -------
        permute_all_df : TYPE
            DESCRIPTION.

        """

        # comp_pair_train -> comp_pair_test
        # t_start = timer()
        score_permute = []
        score_permute.append(
            Parallel(n_jobs=2)(
                delayed(self.permuteXDecode_sbj)(
                    sbj, comp_pair_train, comp_pair_test, nVox_list, n_permute
                )
                for sbj in range(self.n_sbjID)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # comp_pair_test -> comp_pair_train
        # t_start = timer()
        # score_permute = []
        score_permute.append(
            Parallel(n_jobs=2)(
                delayed(self.permuteXDecode_sbj)(
                    sbj, comp_pair_test, comp_pair_train, nVox_list, n_permute
                )
                for sbj in range(self.n_sbjID)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # unpack svm_output
        temp_df = pd.concat(score_permute[0], ignore_index=True)
        temp_df2 = pd.concat(score_permute[1], ignore_index=True)

        temp_df3 = temp_df.append(temp_df2, ignore_index=True)

        # average
        permute_all_df = (
            temp_df3.groupby(["sbjID", "roi", "roi_str", "nVox", "permute_i"])
            .acc.agg("mean")
            .reset_index()
        )

        ## give task label. Here, the task labels crds-ards and ards-crds become
        ## one: crds-ards
        # rds_train_test = "ards_vs_crds"
        permute_all_df["rds_train_test"] = rds_train_test

        return permute_all_df

    def XDecode_roi(
        self,
        sbj,
        vtc_norm,
        t_stat_sbj,
        comp_pair_train,
        comp_pair_test,
        nVox_to_analyze,
        flip_label,
        roi,
    ):
        """
        Perform cross decoding using SVM for a given roi.
        The training dataset uses the data labeled according to comp_pair_train.
        The test dataset uses the data labeled according to comp_pair_test.

        ex: comp_pair_train = [5, 6] -> crds
            comp_pair_test = [1, 2] -> ards
            train: "n_c100 vs f_c100", test: "n_c0 vs f_c0"

        Inputs:
            - sbj: integer
                participant's ID number.
                ex: 0 -> "AI"

            - vtc_norm : pd.DataFrame, [roi, vox, stimID, cond, run, rep, vtc_norm, depth]
                vtc data that has been shifted backward by 2TR and z-scored.
                the last column "depth" of vtc_norm has been added

                loaded from
                vtc_norm = pd.read_pickle(
                            "../../../../Data/VTC_normalized/vtc_shift_norm_{}_{}.pkl"
                            .format(sbjID, nVox_to_analyze))

                ## add a column for depth label (near:-1 or far:1) in the vtc data
                vtc_norm = self.label_depth_in_vtc_data(vtc_norm)

            - t_stat_sbj : dict
                t-stat of a participant for each roi.
                Obtained from GLM_v2

            - comp_pair_train: list of integer
                a comparison index for training data set.
                    "comp_pair_train" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"

            - comp_pair_test: list of integer
                a comparison index for test data set.
                    "comp_pair_test" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"

            - nVox_to_analyze: integer
                the number of voxels to be used for analysis

            - flip_label [int]: flip the label on test dataset.
                                1 (no flip) or -1 (flip)

            - roi: integer
                ROI id
                    "roi" starts from 0, indexing ROIs
                    self.ROIs = ['V1','V2','V3','V3A','V3B','hV4','V7','MT']
                    (0->V1, 1->V2,... )

        Outputs:
            xDecode_score: list
                cross-decoding prediction accuracy for a given ROI

        """
        ## for debugging
        # sbj = 0
        # sbjID = mvpa_xDecode.sbjID_all[sbj]
        # t_stat_sbj = mvpa_xDecode.t_stat_all_sbjID[sbj]
        # t_stat_roi = t_stat_sbj[mvpa_xDecode.ROIs[roi]]
        # load vtc_norm, vtc data that has been shifted backward 2TR and z-scored
        # vtc_norm = mvpa_xDecode.load_vtc_normalized(sbjID, nVox_to_analyze)
        # add a column for depth label (near:-1 or far:1) in the vtc data
        # vtc_norm = mvpa_xDecode.label_depth_in_vtc_data(vtc_norm)

        ## get t_stat
        t_stat_roi = t_stat_sbj[self.ROIs[roi]]

        ## sort voxel in t_stat descending order
        # a  = np.random.rand(10)
        # id_sort = a.argsort()[::-1]
        # a[id_sort]
        id_sort = t_stat_roi.argsort()[::-1]

        # check if nVox_to_analyze < nVox_in_a_ROI
        if nVox_to_analyze > len(id_sort):
            nVox_to_analyze = len(id_sort)

        # # get the first nVox_to_analyze voxel id based on t-stat
        # vox_to_use = id_sort[:nVox_to_analyze]
        # # add voxel ranking
        # vox_id_sorted = np.vstack([vox_to_use, np.arange(nVox_to_analyze)])
        # # get id of unsorted voxel
        # vox_id_unsorted = vox_id_sorted[:, vox_id_sorted[0, :].argsort()]

        # comp_pair = np.array([1, 2]).astype(np.int32)

        ## prepare training dataset
        vtc_vox_train = vtc_norm.loc[
            (vtc_norm.roi == roi)
            & (vtc_norm.stimID.isin(comp_pair_train))
            & (vtc_norm.vox.isin(range(nVox_to_analyze)))
        ]

        X_train = np.array(
            vtc_vox_train.pivot_table(
                index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
            ),
            dtype=np.float32,
        )

        y_train = np.array(
            vtc_vox_train.pivot_table(
                index=["run", "cond", "rep"], columns="vox", values="depth"
            ),
            dtype=np.float32,
        )[:, 0]

        ## prepare test dataset
        vtc_vox_test = vtc_norm.loc[
            (vtc_norm.roi == roi)
            & (vtc_norm.stimID.isin(comp_pair_test))
            & (vtc_norm.vox.isin(range(nVox_to_analyze)))
        ]
        X_test = np.array(
            vtc_vox_test.pivot_table(
                index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
            ),
            dtype=np.float32,
        )

        # flip or not flip the label sign on test dataset by multiplying with flip_label
        y_test = (
            flip_label
            * np.array(
                vtc_vox_test.pivot_table(
                    index=["run", "cond", "rep"], columns="vox", values="depth"
                ),
                dtype=np.float32,
            )[:, 0]
        )

        # create group id based on run-scan number
        run_group = [i for i in range(self.nRuns_all[sbj]) for j in range(8)]
        # run_group = [i for i in range(mvpa_xDecode.nRuns_all[sbj]) for j in range(8)]
        logo = LeaveOneGroupOut()

        ## start xDecoding
        comp_train = (
            self.conds[comp_pair_train[0]] + "," + self.conds[comp_pair_train[1]]
        )
        comp_test = self.conds[comp_pair_test[0]] + "," + self.conds[comp_pair_test[1]]
        # comp_train = (
        #     mvpa_xDecode.conds[comp_pair_train[0]]
        #     + ","
        #     + mvpa_xDecode.conds[comp_pair_train[1]]
        # )
        # comp_test = (
        #     mvpa_xDecode.conds[comp_pair_test[0]]
        #     + ","
        #     + mvpa_xDecode.conds[comp_pair_test[1]]
        # )
        print(
            "run cross-decoding, sbjID: %s, roi: %s, nVox: %s, train: %s, test: %s"
            % (
                self.sbjID_all[sbj],
                self.ROIs[roi],
                str(nVox_to_analyze),
                comp_train,
                comp_test,
            )
        )

        # t_start = timer()
        xDecode_score = []
        xDecode_y_near_given_x_near = []
        xDecode_y_far_given_x_far = []
        for train_idx, test_idx in logo.split(X_train, y_train, groups=run_group):
            # define classifier
            model = SVC(kernel="linear", cache_size=1000)

            # get training and test dataset associated with logo (leave-one-group-out)
            X_train2, X_test2 = X_train[train_idx], X_test[test_idx]
            y_train2, y_test2 = y_train[train_idx], y_test[test_idx]

            # train model
            model.fit(X_train2, y_train2)

            # test model
            score = model.score(X_test2, y_test2)
            xDecode_score.append(score)

            # get model prediction
            # of flip_label = 1 -> prediction -1 is near, 1 is far
            pred = model.predict(X_test2)

            ## compute disparity bias
            # stimuli_near (-1) is predicted near (-1)
            p_y_near_given_x_near = np.sum(pred[y_test2 == -1] == -1) / np.sum(
                y_test2 == -1
            )
            xDecode_y_near_given_x_near.append(p_y_near_given_x_near)

            # stimuli_far (1) is predicted far (1)
            p_y_far_given_x_far = np.sum(pred[y_test2 == 1] == 1) / np.sum(y_test2 == 1)
            xDecode_y_far_given_x_far.append(p_y_far_given_x_far)

        # t_end = timer()
        # print(t_end - t_start)

        return xDecode_score, xDecode_y_near_given_x_near, xDecode_y_far_given_x_far

    def XDecode_allROI(
        self,
        sbj,
        t_stat_sbj,
        comp_pair_train,
        comp_pair_test,
        nVox_list,
        flip_label,
        vox_id,
    ):
        """
        Perform cross decoding using SVM for all ROIs.
        The training dataset uses the data labeled according to comp_pair_train.
        The test dataset uses the data labeled according to comp_pair_test.

        ex: comp_pair_train = [5, 6] -> crds
            comp_pair_test = [1, 2] -> ards
            train: "n_c100 vs f_c100", test: "n_c0 vs f_c0"


        Args:
            - sbj: integer
                participant's ID number.
                ex: 0 -> "AI"

            - vtc_norm : pd.DataFrame, [roi, vox, stimID, cond, run, rep, vtc_norm]
                vtc data that has been shifted backward by 2TR and z-scored.

                loaded from
                vtc_norm = pd.read_pickle(
                            "../../../../Data/VTC_normalized/vtc_shift_norm_{}_{}.pkl"
                            .format(sbjID, nVox_to_analyze))

            - t_stat_sbj : dict
                t-stat of a participant for each roi.
                Obtained from GLM_v2

            - comp_pair_train: list of integer
                a comparison index for training data set.
                    "comp_pair_train" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"

            - comp_pair_test: list of integer
                a comparison index for test data set.
                    "comp_pair_test" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"


            - nVox_to_analyze: integer
                the number of voxels to be used for analysis

            - flip_label [int]: flip the label on test dataset.
                                1 (no flip) or -1 (flip)

            - roi: integer
                ROI id
                    "roi" starts from 0, indexing ROIs
                    self.ROIs = ['V1','V2','V3','V3A','V3B','hV4','V7','MT']
                    (0->V1, 1->V2,... )

            vox_id (int): index denoting the nVox in nVox_all to be used.

        Returns:
            xDecode_score_allROI (list):
            SVM cross-decoding performance output for all ROIs.

        """

        sbjID = self.sbjID_all[sbj]
        nVox_to_analyze = nVox_list[vox_id]

        # load vtc_norm, vtc data that has been shifted backward 2TR and z-scored
        vtc_norm = self.load_vtc_normalized(sbjID, nVox_to_analyze)

        ## add a column for depth label (near:-1 or far:1) in the vtc data
        vtc_norm = self.label_depth_in_vtc_data(vtc_norm)

        # t_start = timer()
        xDecode_score_allROI = []
        xDecode_score_allROI.append(
            Parallel(n_jobs=2)(
                delayed(self.XDecode_roi)(
                    sbj,
                    vtc_norm,
                    t_stat_sbj,
                    comp_pair_train,
                    comp_pair_test,
                    nVox_to_analyze,
                    flip_label,
                    roi,
                )
                for roi in range(self.n_ROIs)
            )
        )
        # t_end = timer()
        # print(t_end - t_start)

        # xDecode_score_allROI.append(
        #     Parallel(n_jobs=2)(
        #         delayed(mvpa_xDecode.XDecode_roi)(
        #             sbj,
        #             vtc_norm,
        #             t_stat_sbj,
        #             comp_pair_train,
        #             comp_pair_test,
        #             nVox_to_analyze,
        #             flip_label,
        #             roi,
        #         )
        #         for roi in range(mvpa_xDecode.n_ROIs)
        #     )
        # )

        return xDecode_score_allROI

    def XDecode_sbj(self, sbj, comp_pair_train, comp_pair_test, nVox_list, flip_label):
        """
        Perform cross decoding using SVM for a single participant.
        The training dataset uses the data labeled according to comp_pair_train.
        The test dataset uses the data labeled according to comp_pair_test.

        ex: comp_pair_train = [5, 6] -> crds
            comp_pair_test = [1, 2] -> ards
            train: "n_c100 vs f_c100", test: "n_c0 vs f_c0"


        Args:
            - sbj: integer
                participant's ID number.
                ex: 0 -> "AI"

            - comp_pair_train: list of integer
                a comparison index for training data set.
                    "comp_pair_train" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"

            - comp_pair_test: list of integer
                a comparison index for test data set.
                    "comp_pair_test" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"


            - nVox_list : no.array, float
                    a list of the number of voxels used for analysis.
                    for ex: np.arange(25, 325, 25)

            - flip_label : int
                flip the label on test dataset.
                1 (no flip) or -1 (flip)

        Returns:
            xDecode_score_sbj (list):
            SVM cross-decoding performance output for a single participant.
        """

        t_stat_sbj = self.t_stat_all_sbjID[sbj]

        # start cross decoding: comp_pair_train -> comp_pair_test
        xDecode_score_sbj = []
        # t_start = timer()
        xDecode_score_sbj.append(
            Parallel(n_jobs=2)(
                delayed(self.XDecode_allROI)(
                    sbj,
                    t_stat_sbj,
                    comp_pair_train,
                    comp_pair_test,
                    nVox_list,
                    flip_label,
                    vox_id,
                )
                for vox_id in range(len(nVox_list))
            )
        )

        # xDecode_score_sbj.append(
        #     Parallel(n_jobs=2)(
        #         delayed(mvpa_xDecode.XDecode_allROI)(
        #             sbj,
        #             t_stat_sbj,
        #             comp_pair_train,
        #             comp_pair_test,
        #             nVox_list,
        #             flip_label,
        #             vox_id,
        #         )
        #         for vox_id in range(len(nVox_list))
        #     )
        # )

        # t_end = timer()
        # print(t_end-t_start)

        return xDecode_score_sbj

    def XDecode_allSbjID_twoway(
        self, comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
    ):
        """
        Perform cross decoding using SVM for all participants.
        The cross-decoding is performed in two ways:
        1. train the classifier with dataset associated with comp_pair_train,
            and test it with the data associated with test comp_pair_test
        2. the opposite of the first scheme, which is
            train the classifier with dataset associated with comp_pair_test,
            and test it with the data associated with test comp_pair_train

        the final cross-decoding accuracy is the average of the two cross-decoding schemes.

        The training dataset uses the data labeled according to comp_pair_train.
        The test dataset uses the data labeled according to comp_pair_test.

        ex: comp_pair_train = [5, 6] -> crds
            comp_pair_test = [1, 2] -> ards
            train: "n_c100 vs f_c100", test: "n_c0 vs f_c0"


        Args:

            - comp_pair_train: list of integer
                a comparison index for training data set.
                    "comp_pair_train" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"

            - comp_pair_test: list of integer
                a comparison index for test data set.
                    "comp_pair_test" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"


            - nVox_list : no.array, float
                    a list of the number of voxels used for analysis.
                    for ex: np.arange(25, 325, 25)

            - flip_label : int
                flip the label on test dataset.
                1 (no flip) or -1 (flip)

            - rds_train_test: string
                a string denoting which rds types are being cross-decoded.

                example: rds_train_test = "crds_ards"
                it means that the cross-decoding is done between crds and ards

        Returns:
            xDecode_group_df (pd.Dataframe):
            SVM cross-decoding performance output for all participants.

            columns=[
                    "nVox",
                    "roi",
                    "fold_id",
                    "acc",
                    "acc_y_near_given_x_near",
                    "acc_y_far_given_x_far",
                ],

        """

        xDecode_score_all = []
        # t_start = timer()
        # comp_pair_train -> comp_pair_test
        xDecode_score_all.append(
            Parallel(n_jobs=4)(
                delayed(self.XDecode_sbj)(
                    sbj, comp_pair_train, comp_pair_test, nVox_list, flip_label
                )
                for sbj in range(self.n_sbjID)
            )
        )

        # t_start = timer()
        # xDecode_score_all = []
        # xDecode_score_all.append(
        #     Parallel(n_jobs=6)(
        #         delayed(mvpa_xDecode.XDecode_sbj)(
        #             sbj, comp_pair_train, comp_pair_test, nVox_list, flip_label
        #         )
        #         for sbj in range(mvpa_xDecode.n_sbjID)
        #     )
        # )

        # t_end = timer()
        # print(t_end - t_start)

        # unpack decode_all
        xDecode_unpack_all = []
        for sbj in range(self.n_sbjID):
            sbjID = self.sbjID_all[sbj]
            nRuns = self.nRuns_all[sbj]
            # nRuns = mvpa_xDecode.nRuns_all[sbj]

            xDecode_sbj = xDecode_score_all[0][sbj][0]  # [len(nVox_list)]
            # len(xDecode_sbj)

            # [nVox, roi_id, fold_id, svm_score_each_fold,
            # xDecode_y_near_given_x_near_each_fold,
            # xDecode_y_far_given_x_far_each_fold]
            xDecode_unpack = np.zeros(
                (len(nVox_list) * self.n_ROIs * nRuns, 6), dtype=np.float32
            )

            for vox_id in range(len(nVox_list)):
                nVox = nVox_list[vox_id]

                # [xDecode_score, xDecode_y_near_given_x_near, xDecode_y_far_given_x_far]
                xDecode_vox = xDecode_sbj[vox_id][0]  # [nROIs]
                # len(xDecode_vox)

                for roi in range(self.n_ROIs):
                    id_start = (vox_id * self.n_ROIs + roi) * nRuns
                    id_end = id_start + nRuns

                    xDecode_unpack[id_start:id_end, 0] = nVox  # nVox
                    xDecode_unpack[id_start:id_end, 1] = roi  # roi_id
                    xDecode_unpack[id_start:id_end, 2] = range(nRuns)  # fold_id

                    ## svm_score
                    # svm_score for each fold
                    xDecode_unpack[id_start:id_end, 3] = xDecode_vox[roi][0]
                    # # mean svm_score
                    # xDecode_unpack[id_start:id_end, 4] = np.mean(xDecode_vox[roi][0])

                    ## xDecode_y_near_given_x_near
                    # xDecode_y_near_given_x_near for each fold
                    xDecode_unpack[id_start:id_end, 4] = xDecode_vox[roi][1]
                    # # mean xDecode_y_near_given_x_near
                    # xDecode_unpack[id_start:id_end, 6] = np.mean(xDecode_vox[roi][1])

                    ## xDecode_y_far_given_x_far
                    # xDecode_y_far_given_x_far for each fold
                    xDecode_unpack[id_start:id_end, 5] = xDecode_vox[roi][2]
                    # # mean xDecode_y_far_given_x_far
                    # xDecode_unpack[id_start:id_end, 8] = np.mean(xDecode_vox[roi][2])

            # create dataframe
            xDecode_sbj = pd.DataFrame(
                xDecode_unpack,
                columns=[
                    "nVox",
                    "roi",
                    "fold_id",
                    "acc",
                    "acc_y_near_given_x_near",
                    "acc_y_far_given_x_far",
                ],
            )

            xDecode_sbj["sbjID"] = sbjID
            xDecode_sbj["roi_str"] = [
                self.ROIs[i]
                for k in range(len(nVox_list))
                for i in range(self.n_ROIs)
                for j in range(nRuns)
            ]
            # xDecode_sbj["comp_train"] = ",".join(
            #     [self.conds[i] for i in np.array(comp_pair_train)]
            # )
            # xDecode_sbj["comp_test"] = ",".join(
            #     [self.conds[i] for i in np.array(comp_pair_test)]
            # )

            xDecode_unpack_all.append(xDecode_sbj.copy())

        # concatenate all decode_sbj_df
        xDecode_all_df = pd.concat(xDecode_unpack_all, ignore_index=True)

        # comp_pair_test -> comp_pair_train
        xDecode_score_all = []
        # t_start = timer()
        xDecode_score_all.append(
            Parallel(n_jobs=4)(
                delayed(self.XDecode_sbj)(
                    sbj, comp_pair_test, comp_pair_train, nVox_list, flip_label
                )
                for sbj in range(self.n_sbjID)
            )
        )
        # t_end = timer()
        # print(t_end - t_start)

        # unpack decode_all
        xDecode_unpack_all = []
        for sbj in range(self.n_sbjID):
            sbjID = self.sbjID_all[sbj]
            nRuns = self.nRuns_all[sbj]

            xDecode_sbj = xDecode_score_all[0][sbj][0]  # [len(nVox_list)]
            # len(xDecode_sbj)

            # [nVox, roi_id, fold_id, svm_score_each_fold, mean_svm_score,
            # xDecode_y_near_given_x_near_each_fold, mean_xDecode_y_near_given_x_near,
            # xDecode_y_far_given_x_far_each_fold, mean_xDecode_y_far_given_x_far]
            xDecode_unpack = np.zeros(
                (len(nVox_list) * self.n_ROIs * nRuns, 6), dtype=np.float32
            )

            for vox_id in range(len(nVox_list)):
                nVox = nVox_list[vox_id]

                xDecode_vox = xDecode_sbj[vox_id][0]  # [nROIs]

                for roi in range(self.n_ROIs):
                    id_start = (vox_id * self.n_ROIs + roi) * nRuns
                    id_end = id_start + nRuns

                    xDecode_unpack[id_start:id_end, 0] = nVox  # nVox
                    xDecode_unpack[id_start:id_end, 1] = roi  # roi_id
                    xDecode_unpack[id_start:id_end, 2] = range(nRuns)  # fold_id

                    ## svm_score
                    # svm_score for each fold
                    xDecode_unpack[id_start:id_end, 3] = xDecode_vox[roi][0]
                    # # mean svm_score
                    # xDecode_unpack[id_start:id_end, 4] = np.mean(xDecode_vox[roi][0])

                    ## xDecode_y_near_given_x_near
                    # xDecode_y_near_given_x_near for each fold
                    xDecode_unpack[id_start:id_end, 4] = xDecode_vox[roi][1]
                    # # mean xDecode_y_near_given_x_near
                    # xDecode_unpack[id_start:id_end, 6] = np.mean(xDecode_vox[roi][1])

                    ## xDecode_y_far_given_x_far
                    # xDecode_y_far_given_x_far for each fold
                    xDecode_unpack[id_start:id_end, 5] = xDecode_vox[roi][2]
                    # # mean xDecode_y_far_given_x_far
                    # xDecode_unpack[id_start:id_end, 8] = np.mean(xDecode_vox[roi][2])

            # create dataframe
            xDecode_sbj = pd.DataFrame(
                xDecode_unpack,
                columns=[
                    "nVox",
                    "roi",
                    "fold_id",
                    "acc",
                    "acc_y_near_given_x_near",
                    "acc_y_far_given_x_far",
                ],
            )

            xDecode_sbj["sbjID"] = sbjID
            xDecode_sbj["roi_str"] = [
                self.ROIs[i]
                for k in range(len(nVox_list))
                for i in range(self.n_ROIs)
                for j in range(nRuns)
            ]
            # xDecode_sbj["comp_train"] = ",".join(
            #     [self.conds[i] for i in np.array(comp_pair_test)]
            # )
            # xDecode_sbj["comp_test"] = ",".join(
            #     [self.conds[i] for i in np.array(comp_pair_train)]
            # )

            xDecode_unpack_all.append(xDecode_sbj.copy())

        # concatenate xDecode_unpack_all
        xDecode_all_df2 = pd.concat(xDecode_unpack_all, ignore_index=True)

        # concatenate all decode_sbj_df
        # xDecode_all_df3 = xDecode_all_df.append(xDecode_all_df2, ignore_index=True)
        xDecode_all_df3 = pd.concat(
            [xDecode_all_df, xDecode_all_df2], ignore_index=True
        )

        xDecode_group_df = (
            xDecode_all_df3.groupby(
                [
                    "sbjID",
                    "roi",
                    "roi_str",
                    "nVox",
                    "fold_id",
                ]
            )
            .agg("mean")
            .reset_index()
        )

        # average across fold_id
        xDecode = (
            xDecode_group_df.groupby(["sbjID", "roi", "roi_str", "nVox"])
            .agg("mean")
            .reset_index()
        )
        xDecode.loc[:, "rds_train_test"] = rds_train_test

        return xDecode

    def XDecode_allSbjID_oneway(
        self, comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
    ):
        """
        Perform cross decoding using SVM for all participants.
        The cross-decoding is performed in one way:
        1. train the classifier with dataset associated with comp_pair_train,
            and test it with the data associated with test comp_pair_test

        The training dataset uses the data labeled according to comp_pair_train.
        The test dataset uses the data labeled according to comp_pair_test.

        ex: comp_pair_train = [5, 6] -> crds
            comp_pair_test = [1, 2] -> ards
            train: "n_c100 vs f_c100", test: "n_c0 vs f_c0"


        Args:

            - comp_pair_train: list of integer
                a comparison index for training data set.
                    "comp_pair_train" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"

            - comp_pair_test: list of integer
                a comparison index for test data set.
                    "comp_pair_test" starts from 0 (fixation cond),
                    indexing comparisons list

                    for example: [1, 2] -> "n_c0 vs f_c0"
                                 [3, 4] -> "n_c50 vs f_c50"
                                 [5, 6] -> "n_c100 vs f_c100"


            - nVox_list : no.array, float
                    a list of the number of voxels used for analysis.
                    for ex: np.arange(25, 325, 25)

            - flip_label : int
                flip the label on test dataset.
                1 (no flip) or -1 (flip)

            - rds_train_test: string
                a string denoting which rds types are being cross-decoded.

                example: rds_train_test = "crds_ards"
                it means that the cross-decoding is done between crds and ards

        Returns:
            xDecode_group_df (pd.Dataframe):
            SVM cross-decoding performance output for all participants.

            columns=[
                    "nVox",
                    "roi",
                    "fold_id",
                    "acc",
                    "acc_y_near_given_x_near",
                    "acc_y_far_given_x_far",
                ],

        """

        xDecode_score_all = []
        # t_start = timer()
        # comp_pair_train -> comp_pair_test
        xDecode_score_all.append(
            Parallel(n_jobs=4)(
                delayed(self.XDecode_sbj)(
                    sbj, comp_pair_train, comp_pair_test, nVox_list, flip_label
                )
                for sbj in range(self.n_sbjID)
            )
        )

        # t_start = timer()
        # xDecode_score_all = []
        # xDecode_score_all.append(
        #     Parallel(n_jobs=6)(
        #         delayed(mvpa_xDecode.XDecode_sbj)(
        #             sbj, comp_pair_train, comp_pair_test, nVox_list, flip_label
        #         )
        #         for sbj in range(mvpa_xDecode.n_sbjID)
        #     )
        # )

        # t_end = timer()
        # print(t_end - t_start)

        # unpack decode_all
        xDecode_unpack_all = []
        for sbj in range(self.n_sbjID):
            sbjID = self.sbjID_all[sbj]
            nRuns = self.nRuns_all[sbj]
            # nRuns = mvpa_xDecode.nRuns_all[sbj]

            xDecode_sbj = xDecode_score_all[0][sbj][0]  # [len(nVox_list)]
            # len(xDecode_sbj)

            # [nVox, roi_id, fold_id, svm_score_each_fold,
            # xDecode_y_near_given_x_near_each_fold,
            # xDecode_y_far_given_x_far_each_fold]
            xDecode_unpack = np.zeros(
                (len(nVox_list) * self.n_ROIs * nRuns, 6), dtype=np.float32
            )

            for vox_id in range(len(nVox_list)):
                nVox = nVox_list[vox_id]

                # [xDecode_score, xDecode_y_near_given_x_near, xDecode_y_far_given_x_far]
                xDecode_vox = xDecode_sbj[vox_id][0]  # [nROIs]
                # len(xDecode_vox)

                for roi in range(self.n_ROIs):
                    id_start = (vox_id * self.n_ROIs + roi) * nRuns
                    id_end = id_start + nRuns

                    xDecode_unpack[id_start:id_end, 0] = nVox  # nVox
                    xDecode_unpack[id_start:id_end, 1] = roi  # roi_id
                    xDecode_unpack[id_start:id_end, 2] = range(nRuns)  # fold_id

                    ## svm_score
                    # svm_score for each fold
                    xDecode_unpack[id_start:id_end, 3] = xDecode_vox[roi][0]
                    # # mean svm_score
                    # xDecode_unpack[id_start:id_end, 4] = np.mean(xDecode_vox[roi][0])

                    ## xDecode_y_near_given_x_near
                    # xDecode_y_near_given_x_near for each fold
                    xDecode_unpack[id_start:id_end, 4] = xDecode_vox[roi][1]
                    # # mean xDecode_y_near_given_x_near
                    # xDecode_unpack[id_start:id_end, 6] = np.mean(xDecode_vox[roi][1])

                    ## xDecode_y_far_given_x_far
                    # xDecode_y_far_given_x_far for each fold
                    xDecode_unpack[id_start:id_end, 5] = xDecode_vox[roi][2]
                    # # mean xDecode_y_far_given_x_far
                    # xDecode_unpack[id_start:id_end, 8] = np.mean(xDecode_vox[roi][2])

            # create dataframe
            xDecode_sbj = pd.DataFrame(
                xDecode_unpack,
                columns=[
                    "nVox",
                    "roi",
                    "fold_id",
                    "acc",
                    "acc_y_near_given_x_near",
                    "acc_y_far_given_x_far",
                ],
            )

            xDecode_sbj["sbjID"] = sbjID
            xDecode_sbj["roi_str"] = [
                self.ROIs[i]
                for k in range(len(nVox_list))
                for i in range(self.n_ROIs)
                for j in range(nRuns)
            ]
            # xDecode_sbj["comp_train"] = ",".join(
            #     [self.conds[i] for i in np.array(comp_pair_train)]
            # )
            # xDecode_sbj["comp_test"] = ",".join(
            #     [self.conds[i] for i in np.array(comp_pair_test)]
            # )

            xDecode_unpack_all.append(xDecode_sbj.copy())

        # concatenate all decode_sbj_df
        xDecode_all_df = pd.concat(xDecode_unpack_all, ignore_index=True)

        xDecode_group_df = (
            xDecode_all_df.groupby(
                [
                    "sbjID",
                    "roi",
                    "roi_str",
                    "nVox",
                    "fold_id",
                ]
            )
            .agg("mean")
            .reset_index()
        )

        # average across fold_id
        xDecode = (
            xDecode_group_df.groupby(["sbjID", "roi", "roi_str", "nVox"])
            .agg("mean")
            .reset_index()
        )
        xDecode.loc[:, "rds_train_test"] = rds_train_test

        return xDecode

    def generate_bootstrap_dist(self, rv_observed, n_bootstrap):
        """
        generate bootstrap distribution (resampling with replacement and with
                                         the same size with the original data)

        Parameters
        ----------
        rv_observed : [n_sbjID] np.array
            random variable

        n_bootstrap : scalar
            the number of bootstrap interation.

        Returns
        -------
        bootstrap_dist : [n_bootstrap] np.array
            bootstrap distribution.

        """

        N = len(rv_observed)
        bootstrap_dist = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # resample data
            id_resample = np.random.randint(N, size=N)

            data_resample = rv_observed[id_resample]

            bootstrap_dist[i] = np.mean(data_resample)

        return bootstrap_dist

    def compute_stat_xDecode_permute_bootstrap_cond(
        self,
        xDecode_fold_df,
        xDecode_group_df,
        permute_group_df,
        nVox_to_analyze,
        n_bootstrap,
        rds_train_test,
        alpha,
    ):
        """
        Parameters
        ----------
        xDecode_fold_df : TYPE
            DESCRIPTION.

        xDecode_group_df : TYPE
            DESCRIPTION.

        permute_group_df : TYPE
            DESCRIPTION.

        nVox_to_analyze: scalar
                the number of voxels used for analysis


        n_bootstrap : TYPE
            DESCRIPTION.

        alpha : scalar, optional
            significant leve. The default is 0.05.

        Returns
        -------
        stat_df : TYPE
            DESCRIPTION.

        """

        stat = np.zeros((self.n_ROIs, 5), dtype=np.float32)
        # stat = np.zeros((mvpa.n_ROIs, 5), dtype=np.float32)
        for roi in range(self.n_ROIs):
            # for roi in range(mvpa.n_ROIs):

            # get observed dataset
            rv_observed = np.array(
                xDecode_fold_df.loc[
                    (xDecode_fold_df.nVox == nVox_to_analyze)
                    & (xDecode_fold_df.roi == roi)
                    & (xDecode_fold_df.rds_train_test == rds_train_test)
                ].acc_cv
            )

            # generate boostrap distribution for empirical data
            rv_bootstrap = self.generate_bootstrap_dist(rv_observed, n_bootstrap)
            # rv_bootstrap = plot_mvpa.generate_bootstrap_dist(rv_observed, n_bootstrap)

            # filter dataframe
            permute_df = permute_group_df.loc[
                (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.roi == roi)
                & (permute_group_df.rds_train_test == rds_train_test)
            ]

            decode_df = xDecode_group_df.loc[
                (xDecode_group_df.nVox == nVox_to_analyze)
                & (xDecode_group_df.roi == roi)
                & (xDecode_group_df.rds_train_test == rds_train_test)
            ]

            mean_observation = np.float32(decode_df.acc.mean())

            if rds_train_test == "crds_hmrds":
                ## calculate p_val: proportions of rv_bootstrap which is less than
                # val_threshold (the right tail of rv_permute).
                # this stat test is very strict..
                # alpha_corrected = alpha/(2*len(ROIs))
                rv_permute = np.float32(permute_df.acc)
                baseline = np.percentile(rv_permute, (1 - alpha) * 100)
                p_val = np.sum(rv_bootstrap < baseline) / n_bootstrap

            elif (rds_train_test == "crds_ards") or (rds_train_test == "hmrds_ards"):
                rv_permute = np.float32(permute_df.acc)
                baseline = np.percentile(rv_permute, alpha * 100)
                p_val = np.sum(rv_bootstrap > baseline) / n_bootstrap

            stat[roi, 0] = np.int32(roi)
            stat[roi, 1] = np.mean(rv_permute)
            stat[roi, 2] = baseline
            stat[roi, 3] = mean_observation
            stat[roi, 4] = p_val

        stat_df = pd.DataFrame(
            stat, columns=["roi", "mean_permute", "baseline", "mean_obs", "p_val"]
        )
        stat_df["rds_train_test"] = rds_train_test
        stat_df["roi_str"] = self.ROIs
        stat_df["nVox"] = nVox_to_analyze

        return stat_df

    def compute_stat_xDecode_permute_bootstrap(
        self,
        xDecode_all_df,
        permuteXDecode_all_df,
        nVox_to_analyze,
        n_bootstrap,
        rds_train_test,
        alpha=0.05,
    ):
        """
        Compute p-val based on permutation and bootstrap distributions.

        Parameters
        ----------
        xDecode_all_df: pd.DataFrame
                            [sbjID, roi, roi_str, nVox, acc, rds_train_test].
            dataframe containing the cross-decoding performance for each participant

        permuteXDecode_all_df: pd.DataFrame
                        [sbjID, roi, roi_str, nVox, permute_i, acc, rds_train_test]
            dataframe containing the distribution of cross-decoding permutation
            (10000 iterations)

        nVox_to_analyze: scalar
                the number of voxels used for analysis

        n_bootstrap : TYPE
            DESCRIPTION.

        alpha : scalar, optional
            significant leve. The default is 0.05.

        Returns
        -------
        stat_df : pd.Dataframe,
            [roi, mean_permute, baseline, mean_obs, p_val, rds_train_test,
            roi_str, nVox, partiality]

            dataframe containing the statistic of cross-decoding analysis

        """

        # stat_all = []
        ## do statistical testing across sbjID

        # average across sbjID for each roi, nVox, permute_i, rds_train_test
        permute_group_df = (
            permuteXDecode_all_df.groupby(
                ["roi", "nVox", "permute_i", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # take average on xDecode_all_df across fold_id and then across sbjID
        xDecode_fold_df = (
            xDecode_all_df.groupby(
                ["sbjID", "roi", "roi_str", "nVox", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_fold_df = xDecode_fold_df.rename(columns={"mean": "acc_cv"})

        xDecode_group_df = (
            xDecode_fold_df.groupby(["roi", "roi_str", "nVox", "rds_train_test"])
            .acc_cv.agg(["mean"])
            .reset_index()
        )
        xDecode_group_df = xDecode_group_df.rename(columns={"mean": "acc"})

        stat_df = self.compute_stat_xDecode_permute_bootstrap_cond(
            xDecode_fold_df,
            xDecode_group_df,
            permute_group_df,
            nVox_to_analyze,
            n_bootstrap,
            rds_train_test,
            alpha,
        )

        stat_df.loc[:, "partiality"] = "all"

        return stat_df
