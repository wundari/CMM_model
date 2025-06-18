"""
File: Codes/Python/CMM/MVPA/MVPA_Decode.py
Project: Codes/Python/CMM/
Created Date: 2025-06-16
Author: Bayu G. Wundari
-----
Last Modified: 2025-06-16
Modified By: Bayu G. Wundari

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
script for MVPA
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


# sys.path.append("../Common")
from Common.Common import General


class MVPA_Decode(General):
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

    def count_nVox_max(self):
        """
        compute the max number of voxels in each ROI for each participant

        Returns:
            nVox_all [n_sbjID, n_ROIs]: max number of voxels in each roi
                                for each participant.
        """

        nVox_max_all = np.empty((self.n_sbjID, self.n_ROIs), dtype=np.int32)

        for sbj in range(self.n_sbjID):
            # load vtc
            vtc = self.load_vtc(sbj)

            for roi in range(self.n_ROIs):
                nVox_max_all[sbj, roi] = vtc[vtc.roi == roi].vox.max()

        return nVox_max_all

    def normalize_train_cv(self, X_train, X_test):
        """
        normalize data within cross-validation fold.
        The training and test dataset are normalized separately.
        The mean and the sigma of training dataset will be used to normalize test dataset.

        Args:
            X_train (TYPE): DESCRIPTION.
            X_test (TYPE): DESCRIPTION.

        Returns:
            X_train_norm (TYPE): DESCRIPTION.
            X_test_norm (TYPE): DESCRIPTION.

        """
        # standardize training dataset
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)

        # standardize training dataframe
        X_train_norm = (X_train - X_train_mean) / X_train_std

        # transfer the mean and std of training dataframe to test dataframe
        X_test_norm = (X_test - X_train_mean) / X_train_std

        return X_train_norm, X_test_norm

    def permuteDecode_onePermute(self, X, run_group, nRun):
        # print("run permutation test, sbjID: %s, roi: %s, comp_pair: %s_VS_%s, iter: %s"
        #           %(sbjID, ROIs[roi-1],
        #             conds[comp_pair[0]-1], conds[comp_pair[1]-1],
        #             str(i+1)))

        ## relabeling (dataset-wise scheme,
        # Etzel & Braver, Workshop on Pattern recognition in neuroimaging, 2013)

        # assigning new label in a dataset-wise scheme:
        # permuting (resampling without replacement) both training and test labels.

        id_permute = np.random.choice(len(self.perms), nRun, replace=False)
        # id_permute = np.random.choice(len(perms), nRun,
        #                                replace=False)
        y_permute = np.int32(np.reshape(self.perms[id_permute], -1))
        # y_permute = np.int32(np.reshape(perms[id_permute], -1))

        logo = LeaveOneGroupOut()

        score = []
        for id_train, id_test in logo.split(X, y_permute, groups=run_group):

            # define classifier
            model = SVC(kernel="linear", cache_size=50000)

            # get training and test dataset
            X_train, X_test = X[id_train], X[id_test]
            y_train, y_test = y_permute[id_train], y_permute[id_test]

            # standardize training dataset
            # X_train_norm, X_test_norm = self.normalize_train_cv(X_train, X_test)

            # train model
            # model.fit(X_train_norm, y_train)
            model.fit(X_train, y_train)

            # test model
            # score.append(model.score(X_test_norm, y_test))
            score.append(model.score(X_test, y_test))

        score_onePermute = np.mean(score, dtype=np.float32)

        return score_onePermute

    def permuteDecode_roi(
        self, sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_to_analyze, roi, n_permute
    ):
        """
        run permutation test within a ROI

        """

        ## get t_stat
        t_stat_roi = t_stat_sbj[self.ROIs[roi]]

        ## sort voxel in t_stat descending order
        # a  = np.random.rand(10)
        # id_sort = a.argsort()[::-1]
        # a[id_sort]
        # id_sort = t_stat_roi.argsort()[::-1]

        # check if nVox_to_analyze < nVox in a ROI
        if nVox_to_analyze > len(t_stat_roi):
            nVox_to_analyze = len(t_stat_roi)

        # comp_pair = np.array([1, 2]).astype(np.int32)
        vtc_vox = vtc_norm.loc[
            (vtc_norm.roi == roi)
            & (vtc_norm.stimID.isin(comp_pair))
            & (vtc_norm.vox.isin(range(nVox_to_analyze)))
        ]

        vtc_vox_x = vtc_vox.pivot_table(
            index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
        )

        X = np.array(vtc_vox_x, dtype=np.float32)

        # create group id based on run-scan number
        nRun = self.nRuns_all[sbj]
        run_group = [i for i in range(nRun) for j in range(8)]  # 8 -> 4 near, 4 far

        print(
            "run permutation test, sbjID: %s, roi: %s, nVox: %s, comp_pair: %s_VS_%s"
            % (
                self.sbjID_all[sbj],
                self.ROIs[roi],
                str(nVox_to_analyze),
                self.conds[comp_pair[0]],
                self.conds[comp_pair[1]],
            )
        )

        # t_start = timer()
        score_permute = []
        score_permute.append(
            Parallel(n_jobs=4)(
                delayed(self.permuteDecode_onePermute)(X, run_group, nRun)
                for i in range(n_permute)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # unpacking the "score_permute" list and convert it to dataframe
        score_permute_df = pd.DataFrame(score_permute[0], columns=["acc"])

        score_permute_df["sbjID"] = self.sbjID_all[sbj]
        score_permute_df["roi"] = roi
        score_permute_df["roi_str"] = self.ROIs[roi]
        score_permute_df["nVox"] = nVox_to_analyze
        score_permute_df["permute_i"] = np.arange(n_permute)
        score_permute_df["comp_pair"] = ",".join(
            [self.conds[i] for i in np.array(comp_pair)]
        )

        return score_permute_df

    def permuteDecode_roi_allVox_percentage(
        self, sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_list, roi, n_permute
    ):

        score_permute = []
        score_permute.append(
            Parallel(n_jobs=4)(
                delayed(self.permuteDecode_roi)(
                    sbj,
                    vtc_norm,
                    t_stat_sbj,
                    comp_pair,
                    nVox_to_analyze,
                    roi,
                    n_permute,
                )
                for nVox_to_analyze in nVox_list
            )
        )

        # concetenate all df
        decode_permute_roi_allVox_percentage_df = pd.concat(
            score_permute[0], ignore_index=True
        )

        return decode_permute_roi_allVox_percentage_df

    def permuteDecode_allROI(
        self, sbj, t_stat_sbj, comp_pair, nVox_list, vox_id, n_permute
    ):

        sbjID = self.sbjID_all[sbj]
        nVox_to_analyze = nVox_list[vox_id]

        ## load vtc_norm, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        vtc_norm = self.load_vtc_normalized(sbjID, nVox_to_analyze)

        score_permute = []
        # n_permute = 500
        # t_start = timer()
        # print("run permutation test, sbjID: %s" %sbjID)
        score_permute.append(
            Parallel(n_jobs=2)(
                delayed(self.permuteDecode_roi)(
                    sbj,
                    vtc_norm,
                    t_stat_sbj,
                    comp_pair,
                    nVox_to_analyze,
                    roi,
                    n_permute,
                )
                for roi in range(self.n_ROIs)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # concetenate all df
        score_permute_allROI_df = pd.concat(score_permute[0], ignore_index=True)

        return score_permute_allROI_df

    def permuteDecode_sbj(self, sbj, comp_pair, nVox_list, n_permute):

        t_stat_sbj = self.t_stat_all_sbjID[sbj]

        # start permutation test
        # t_start = timer()
        # print("run permutation test, sbjID: %s" %sbjID)

        score_permute = []
        score_permute.append(
            Parallel(n_jobs=2)(
                delayed(self.permuteDecode_allROI)(
                    sbj, t_stat_sbj, comp_pair, nVox_list, vox_id, n_permute
                )
                for vox_id in range(len(nVox_list))
            )
        )

        score_permute_df = pd.concat(score_permute[0], ignore_index=True)

        # t_end = timer()
        # print(t_end-t_start)

        return score_permute_df

    def permuteDecode_allSbj(self, comp_pair, nVox_list, n_permute):

        # t_start = timer()
        score_permute = []
        score_permute.append(
            Parallel(n_jobs=8)(
                delayed(self.permuteDecode_sbj)(sbj, comp_pair, nVox_list, n_permute)
                for sbj in range(self.n_sbjID)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # unpack svm_output
        permuteTest_df = pd.concat(score_permute[0], ignore_index=True)

        return permuteTest_df

    def decode_roi(self, sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_to_analyze, roi):
        """
        Perform decoding using SVM for a pair of comparison for a given roi.
        The training and test datasets come from the same RDS group:
            train: "n_c100 vs f_c100", test: "n_c100 vs f_c100",
            train: "n_c50 vs f_c50", test: "n_c50 vs f_c50",
            train: "n_c0 vs f_c0",  test: "n_c0 vs f_c0"


        Parameters
        ----------
        sbj : scalar
            participant's index.

        vtc_norm : pd.DataFrame, [roi, run, rep, vox, stimID, cond, vtc_norm]
            normalized vtc data.

        t_stat_sbj : dict
            t-stat of a participant for each roi.
            Obtained from GLM_v2

        comp_pair : np.array, integer
            a comparison index for training and testing data set.

                    for example: comp_pair = np.array([1, 2]).astype(np.int32)
                                    [1, 2] -> "n_c0 vs f_c0"
                                    [3, 4] -> "n_c50 vs f_c50"
                                    [5, 6] -> "n_c100 vs f_c100".

        nVox_to_analyze : scalar
            the number of voxels used for analysis.

        roi : scalar
            ROI index.
            "roi" starts from 0, indexing ROIs
                    self.ROIs = ['V1','V2','V3','V3A','V3B','hV4','V7','MT']
                    (0->V1, 1->V2,... )

        Returns
        -------
        score : list
            SVM performance output.

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

        vtc_vox = vtc_norm.loc[
            (vtc_norm.roi == roi)
            & (vtc_norm.stimID.isin(comp_pair))
            & (vtc_norm.vox.isin(range(nVox_to_analyze)))
        ]

        vtc_vox_x = vtc_vox.pivot_table(
            index=["run", "cond", "rep"], columns="vox", values="vtc_norm"
        )

        vtc_vox_y = vtc_vox.pivot_table(
            index=["run", "cond", "rep"], columns="vox", values="stimID"
        )

        X = np.array(vtc_vox_x, dtype=np.float32)
        y = np.array(vtc_vox_y, dtype=np.int32)[:, 0]

        # ## sort voxel
        # # add voxel id and ranking information
        # X = np.vstack([X, vox_id_unsorted])
        # # sort based on voxel ranking
        # X_sorted = X[:, X[-1, ].argsort()]
        # # remove the last 2 rows
        # X = X_sorted[:-2]

        # create group id based on run-scan number
        nRun = self.nRuns_all[sbj]
        run_group = [i for i in range(nRun) for j in range(8)]  # 8 -> 4 near, 4 far
        logo = LeaveOneGroupOut()

        # start decoding
        print(
            "run decoding, sbjID: %s, roi: %s, nVox: %s, comp_pair: %s_VS_%s"
            % (
                self.sbjID_all[sbj],
                self.ROIs[roi],
                str(nVox_to_analyze),
                self.conds[comp_pair[0]],
                self.conds[comp_pair[1]],
            )
        )

        # t_start = timer()
        score = []
        for train_idx, test_idx in logo.split(X, y, groups=run_group):

            # define classifier
            model = SVC(kernel="linear", cache_size=50000)

            # get training and test dataset
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # standardize training dataset
            # X_train_norm, X_test_norm = self.normalize_train_cv(X_train, X_test)

            # train model
            # model.fit(X_train_norm, y_train)
            model.fit(X_train, y_train)

            # test model
            # score.append(model.score(X_test_norm, y_test))
            score.append(model.score(X_test, y_test))

        # t_end = timer()
        # print(t_end - t_start)

        # decode_score = np.mean(score)

        return score

    def decode_roi_allVox_percentage(
        self, sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_list, roi
    ):

        decode_score_allVox_list = []
        decode_score_allVox_list.append(
            Parallel(n_jobs=5)(
                delayed(self.decode_roi)(
                    sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_to_analyze, roi
                )
                for nVox_to_analyze in nVox_list
            )
        )

        # unpack
        n_runs = self.nRuns_all[sbj]
        decode_score_allVox = np.empty((len(nVox_list), n_runs), dtype=np.float32)
        for i in range(len(nVox_list)):
            decode_score_allVox[i] = decode_score_allVox_list[0][i]

        return decode_score_allVox

    def decode_allROI(self, sbj, t_stat_sbj, comp_pair, nVox_list, vox_id):
        """
        Perform decoding using SVM for a pair of comparison for all ROIs.

        The training and test dataset come from the same RDS group:
            train: "n_c100 vs f_c100", test: "n_c100 vs f_c100",
            train: "n_c50 vs f_c50", test: "n_c50 vs f_c50",
            train: "n_c0 vs f_c0",  test: "n_c0 vs f_c0"


        Parameters
        ----------
        sbj : scalar
            participant's index.

        vtc_norm : pd.DataFrame, [roi, run, rep, vox, stimID, cond, vtc_norm]
            normalized vtc data.

        t_stat_sbj : dict
            t-stat of a participant for each roi.
            Obtained from GLM_v2

        comp_pair : np.array, integer
            a comparison index for training and testing data set.

                    for example: comp_pair = np.array([1, 2]).astype(np.int32)
                                    [1, 2] -> "n_c0 vs f_c0"
                                    [3, 4] -> "n_c50 vs f_c50"
                                    [5, 6] -> "n_c100 vs f_c100".

        nVox_list : no.array, float
            a list of the number of voxels used for analysis.
            for ex: np.arange( 25, 526, 25)

        vox_id : scalar
            a scalar indexing the nVox_to_analyse in nVox_list

        Returns
        -------
        decode_score_allROI : list
            SVM decoding performance output for each ROI.

        """

        sbjID = self.sbjID_all[sbj]
        nVox_to_analyze = nVox_list[vox_id]

        # load vtc_norm, vtc data that has been shifted backward 2TR and z-scored
        vtc_norm = self.load_vtc_normalized(sbjID, nVox_to_analyze)
        # vtc_norm = mvpa.load_vtc_normalized(sbjID, nVox_to_analyze)

        # t_start = timer()
        decode_score_allROI = []
        decode_score_allROI.append(
            Parallel(n_jobs=8)(
                delayed(self.decode_roi)(
                    sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_to_analyze, roi
                )
                for roi in range(self.n_ROIs)
            )
        )
        # t_end = timer()
        # print(t_end - t_start)

        return decode_score_allROI

    def decode_sbj(self, sbj, comp_pair, nVox_list):
        """
        perform decoding using SVM for a single participant and a pair of
        comparison.

        Parameters
        ----------
        sbj : scalar
            participant's index.

        comp_pair : np.array, integer
            a comparison index for training and testing data set.

                    for example: comp_pair = np.array([1, 2]).astype(np.int32)
                                    [1, 2] -> "n_c0 vs f_c0"
                                    [3, 4] -> "n_c50 vs f_c50"
                                    [5, 6] -> "n_c100 vs f_c100".

        nVox_list : no.array, float
            a list of the number of voxels used for analysis.
            for ex: np.arange(25, 325, 25)

        Returns
        -------
        decode_sbj : list
            svm output.

        """

        t_stat_sbj = self.t_stat_all_sbjID[sbj]

        # start decoding
        decode_sbj = []
        # t_start = timer()
        decode_sbj.append(
            Parallel(n_jobs=2)(
                delayed(self.decode_allROI)(
                    sbj, t_stat_sbj, comp_pair, nVox_list, vox_id
                )
                for vox_id in range(len(nVox_list))
            )
        )

        # t_end = timer()
        # print(t_end-t_start)

        return decode_sbj

    def decode_allSbj(self, comp_pair, nVox_list, n_jobs=4):
        """
        perform decoding with SVM for all participants and a pair of comparison

        Parameters
        ----------
        comp_pair : np.array, integer
            a comparison index for training and testing data set.

                    for example: comp_pair = np.array([1, 2]).astype(np.int32)
                                    [1, 2] -> "n_c0 vs f_c0"
                                    [3, 4] -> "n_c50 vs f_c50"
                                    [5, 6] -> "n_c100 vs f_c100".

        nVox_list : no.array, float
            a list of the number of voxels used for analysis.
            for ex: np.arange( 25, 526, 25)

        n_jobs : scalar, optional
            the number of workers for parallel processing. The default is 2.

        Returns
        -------
        decode_all_df : pd.DataFrame, [nVox, roi, fold_id, acc, acc_mean, acc_sem,
                                       sbjID, roi_str, comp_pair]
            decoding results of all participants.

        """

        decode_all = []
        # t_start = timer()
        decode_all.append(
            Parallel(n_jobs=n_jobs)(
                delayed(self.decode_sbj)(sbj, comp_pair, nVox_list)
                for sbj in range(self.n_sbjID)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # unpack decode_all
        decode_unpack_all = []
        for sbj in range(self.n_sbjID):

            sbjID = self.sbjID_all[sbj]
            nRun = self.nRuns_all[sbj]

            decode_sbj = decode_all[0][sbj][0]

            decode_unpack = np.zeros(
                (len(nVox_list) * self.n_ROIs * nRun, 6), dtype=np.float32
            )

            for vox_id in range(len(nVox_list)):

                nVox = nVox_list[vox_id]

                decode_vox = decode_sbj[vox_id][0]

                for roi in range(self.n_ROIs):
                    id_start = (vox_id * self.n_ROIs + roi) * nRun
                    id_end = id_start + nRun

                    decode_unpack[id_start:id_end, 0] = nVox  # nVox
                    decode_unpack[id_start:id_end, 1] = roi  # roi_id
                    decode_unpack[id_start:id_end, 2] = range(nRun)  # fold_id
                    decode_unpack[id_start:id_end, 3] = decode_vox[
                        roi
                    ]  # svm_score for each fold
                    decode_unpack[id_start:id_end, 4] = np.mean(
                        decode_vox[roi]
                    )  # mean svm_score
                    decode_unpack[id_start:id_end, 5] = sem(
                        decode_vox[roi]
                    )  # std svm_score

            # create dataframe
            decode_sbj_df = pd.DataFrame(
                decode_unpack,
                columns=["nVox", "roi", "fold_id", "acc", "acc_mean", "acc_sem"],
            )

            decode_sbj_df["sbjID"] = sbjID
            decode_sbj_df["roi_str"] = [
                self.ROIs[i]
                for k in range(len(nVox_list))
                for i in range(self.n_ROIs)
                for j in range(nRun)
            ]
            decode_sbj_df["comp_pair"] = ",".join([self.conds[i] for i in comp_pair])

            decode_unpack_all.append(decode_sbj_df.copy())

        # concatenate all decode_sbj_df
        decode_all_df = pd.concat(decode_unpack_all, ignore_index=True)

        return decode_all_df

    def decode_rdstype_roi(self, sbj, vtc_norm, t_stat_sbj, nVox_to_analyze, roi):
        """
        perform decoding with SVM for multiclass classification.
        the task is to classify ards, hmrds, or crds.

        Parameters
        ----------
        sbj : scalar
            participant's index.

        vtc_norm : pd.DataFrame, [roi, run, rep, vox, stimID, cond, vtc_norm]
            normalized vtc data.

        t_stat_sbj : dict
            t-stat of a participant for each roi.
            Obtained from GLM_v2

        nVox_to_analyze : scalar
            the number of voxels used for analysis.

        roi : scalar
            ROI index.
            "roi" starts from 0, indexing ROIs
                    self.ROIs = ['V1','V2','V3','V3A','V3B','hV4','V7','MT']
                    (0->V1, 1->V2,... )

        Returns
        -------
        score : list
            SVM performance output.

        """

        ## get vtc for ards
        vtc_ards = vtc_norm.loc[vtc_norm.stimID.isin(np.arange(1, 3))]
        # average near-far response
        vtc_ards_avg = (
            vtc_ards.groupby(["roi", "run", "vox", "rep"])
            .vtc_norm.agg(["mean"])
            .reset_index()
        )
        vtc_ards_avg = vtc_ards_avg.rename(columns={"mean": "vtc_norm"})
        # label ards (rds = 1)
        vtc_ards_avg = vtc_ards_avg.assign(rds=1)
        vtc_ards_avg = vtc_ards_avg.assign(rds_type="ards")

        ## get vtc for hmrds
        vtc_hmrds = vtc_norm.loc[vtc_norm.stimID.isin(np.arange(3, 5))]
        # average near-far response
        vtc_hmrds_avg = (
            vtc_hmrds.groupby(["roi", "run", "vox", "rep"])
            .vtc_norm.agg(["mean"])
            .reset_index()
        )
        vtc_hmrds_avg = vtc_hmrds_avg.rename(columns={"mean": "vtc_norm"})
        # label hmrds (rds = 2)
        vtc_hmrds_avg = vtc_hmrds_avg.assign(rds=2)
        vtc_hmrds_avg = vtc_hmrds_avg.assign(rds_type="hmrds")

        ## get vtc for crds
        vtc_crds = vtc_norm.loc[vtc_norm.stimID.isin(np.arange(5, 7))]
        # average near-far response
        vtc_crds_avg = (
            vtc_crds.groupby(["roi", "run", "vox", "rep"])
            .vtc_norm.agg(["mean"])
            .reset_index()
        )
        vtc_crds_avg = vtc_crds_avg.rename(columns={"mean": "vtc_norm"})
        # label crds (rds = 3)
        vtc_crds_avg = vtc_crds_avg.assign(rds=3)
        vtc_crds_avg = vtc_crds_avg.assign(rds_type="crds")

        ## merge df
        vtc_rds = pd.concat([vtc_ards_avg, vtc_hmrds_avg, vtc_crds_avg])

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
        vox_to_use = id_sort[:nVox_to_analyze]
        # add voxel ranking
        vox_id_sorted = np.vstack([vox_to_use, np.arange(nVox_to_analyze)])
        # get id of unsorted voxel
        vox_id_unsorted = vox_id_sorted[:, vox_id_sorted[0, :].argsort()]

        vtc_vox = vtc_rds.loc[(vtc_rds.roi == roi) & (vtc_rds.vox.isin(vox_to_use))]

        vtc_vox_x = vtc_vox.pivot_table(
            index=["run", "rds_type", "rep"], columns="vox", values="vtc_norm"
        )

        vtc_vox_y = vtc_vox.pivot_table(
            index=["run", "rds_type", "rep"], columns="vox", values="rds"
        )

        X = np.array(vtc_vox_x, dtype=np.float32)
        y = np.array(vtc_vox_y, dtype=np.int32)[:, 0]

        ## sort voxel
        # add voxel id and ranking information
        X = np.vstack([X, vox_id_unsorted])
        # sort based on voxel ranking
        X_sorted = X[
            :,
            X[-1,].argsort(),
        ]
        # remove the last 2 rows
        X = X_sorted[:-2]

        # create group id based on run-scan number
        nRun = self.nRuns_all[sbj]
        run_group = [
            i for i in range(nRun) for j in range(12)
        ]  # 12 -> 4 ards, 4 hmrds, 4 crds
        logo = LeaveOneGroupOut()

        # start decoding
        print(
            "run decoding, sbjID: %s, roi: %s, nVox: %s, comp_pair: ards_VS_hmrds_VS_crds"
            % (self.sbjID_all[sbj], self.ROIs[roi], str(nVox_to_analyze))
        )

        # t_start = timer()
        score = []
        for train_idx, test_idx in logo.split(X, y, groups=run_group):

            # define classifier
            model = SVC(kernel="linear", cache_size=50000)

            # get training and test dataset
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # standardize training dataset
            X_train_norm, X_test_norm = self.normalize_train_cv(X_train, X_test)

            # train model
            model.fit(X_train_norm, y_train)

            # test model
            score.append(model.score(X_test_norm, y_test))

        # t_end = timer()
        # print(t_end - t_start)

        # decode_score = np.mean(score)

        return score

    def decode_rdstype_allROI(self, sbj, vtc_norm, t_stat_sbj, nVox_list, vox_id):
        """
        perform decoding with SVM for multiclass classification for all ROIs.
        the task is to classify ards, hmrds, or crds.

        Parameters
        ----------
        sbj : scalar
            participant's index.

        vtc_norm : pd.DataFrame, [roi, run, rep, vox, stimID, cond, vtc_norm]
            normalized vtc data.

        t_stat_sbj : dict
            t-stat of a participant for each roi.
            Obtained from GLM_v2

        nVox_list : no.array, float
            a list of the number of voxels used for analysis.
            for ex: np.arange( 25, 526, 25)

        vox_id : scalar
            a scalar indexing the nVox_to_analyse in nVox_list

        Returns
        -------
        decode_score_allROI : list
            SVM decoding performance output for each ROI.

        """

        nVox_to_analyze = nVox_list[vox_id]

        # t_start = timer()
        decode_score_allROI = []
        decode_score_allROI.append(
            Parallel(n_jobs=8, backend="threading")(
                delayed(self.decode_rdstype_roi)(
                    sbj, vtc_norm, t_stat_sbj, nVox_to_analyze, roi
                )
                for roi in range(self.n_ROIs)
            )
        )
        # t_end = timer()
        # print(t_end - t_start)

        return decode_score_allROI

    def decode_rdstype_sbj(self, sbj, nVox_list):
        """
        perform decoding with SVM for multiclass classification for a single
        participant
        the task is to classify ards, hmrds, or crds.

        Parameters
        ----------
        sbj : scalar
            participant's index.

        nVox_list : no.array, float
            a list of the number of voxels used for analysis.
            for ex: np.arange( 25, 526, 25)

        Returns
        -------
        decode_sbj : list
            svm output.

        """

        sbjID = self.sbjID_all[sbj]
        t_stat_sbj = self.t_stat_all_sbjID[sbj]

        ## load vtc_norm, [run, roi, timepoint, vox, vtc_value, stimID, cond, rep].
        vtc_norm = self.load_vtc_normalized(sbjID)

        # start decoding
        decode_sbj = []
        # t_start = timer()
        decode_sbj.append(
            Parallel(n_jobs=len(nVox_list))(
                delayed(self.decode_rdstype_allROI)(
                    sbj, vtc_norm, t_stat_sbj, nVox_list, vox_id
                )
                for vox_id in range(len(nVox_list))
            )
        )

        # t_end = timer()
        # print(t_end-t_start)

        return decode_sbj

    def decode_rdstype_allSbj(self, nVox_list, n_jobs=2):
        """
        perform decoding with SVM for multiclass classification for all
        participants

        Parameters
        ----------
        nVox_list : no.array, float
            a list of the number of voxels used for analysis.
            for ex: np.arange( 25, 526, 25)

        n_jobs : scalar, optional
            the number of workers for parallel processing. The default is 2.

        Returns
        -------
        decode_all_df : pd.DataFrame, [nVox, roi, fold_id, acc, acc_mean, acc_sem,
                                       sbjID, roi_str]
            decoding results of all participants.

        """

        decode_all = []
        # t_start = timer()
        decode_all.append(
            Parallel(n_jobs=n_jobs)(
                delayed(self.decode_rdstype_sbj)(sbj, nVox_list)
                for sbj in range(self.n_sbjID)
            )
        )

        # t_end = timer()
        # print(t_end - t_start)

        # unpack decode_all
        decode_unpack_all = []
        for sbj in range(self.n_sbjID):

            sbjID = self.sbjID_all[sbj]
            nRun = self.nRuns_all[sbj]

            decode_sbj = decode_all[0][sbj][0]

            decode_unpack = np.zeros(
                (len(nVox_list) * self.n_ROIs * nRun, 6), dtype=np.float32
            )

            for vox_id in range(len(nVox_list)):

                nVox = nVox_list[vox_id]

                decode_vox = decode_sbj[vox_id][0]

                for roi in range(self.n_ROIs):
                    id_start = (vox_id * self.n_ROIs + roi) * nRun
                    id_end = id_start + nRun

                    decode_unpack[id_start:id_end, 0] = nVox  # nVox
                    decode_unpack[id_start:id_end, 1] = roi  # roi_id
                    decode_unpack[id_start:id_end, 2] = range(nRun)  # fold_id
                    decode_unpack[id_start:id_end, 3] = decode_vox[
                        roi
                    ]  # svm_score for each fold
                    decode_unpack[id_start:id_end, 4] = np.mean(
                        decode_vox[roi]
                    )  # mean svm_score
                    decode_unpack[id_start:id_end, 5] = sem(
                        decode_vox[roi]
                    )  # std svm_score

            # create dataframe
            decode_sbj_df = pd.DataFrame(
                decode_unpack,
                columns=["nVox", "roi", "fold_id", "acc", "acc_mean", "acc_sem"],
            )

            decode_sbj_df["sbjID"] = sbjID
            decode_sbj_df["roi_str"] = [
                self.ROIs[i]
                for k in range(len(nVox_list))
                for i in range(self.n_ROIs)
                for j in range(nRun)
            ]

            decode_unpack_all.append(decode_sbj_df.copy())

        # concatenate all decode_sbj_df
        decode_all_df = pd.concat(decode_unpack_all, ignore_index=True)

        return decode_all_df

    def compute_p_val_decode(
        self, decode_all_df, permuteDecode_all_df, comp_pair, nVox_to_analyze
    ):
        """
        compute p_value from the permutation distribution for each participant
        and ROI.

        the p-value here is obtained by counting the number of elements in
        the permutation distribution whose svm_score is larger than
        the observed svm_score, and then divided by the number of elements in
        the permutation distribution (the number of permutation iteration, 10000)

        Args:

            decode_all_df: pd.DataFrame
                            [nVox, roi, fold_id, acc, acc_mean, acc_sem, sbjID,
                             roi_str, comp_pair].
                dataframe containing the decoding performance for each participant

            permuteDecode_all_df: pd.DataFrame
                            [acc, sbjID, roi, roi_str, nVox, permute_i, comp_pair]
                dataframe containing the distribution of decoding permutation
                (10000 iterations)

            comp_pair : np.array, integer
            a comparison index for training and testing data set.

                    for example: comp_pair = np.array([1, 2]).astype(np.int32)
                                    [1, 2] -> "n_c0 vs f_c0"
                                    [3, 4] -> "n_c50 vs f_c50"
                                    [5, 6] -> "n_c100 vs f_c100".

            nVox_to_analyze: scalar
                the number of voxels used for analysis

        Returns:
            p_val_all_df: pd.DataFrame.
                        [acc_mean, p_val, sbjID, roi, roi_str]
                dataframe containing p-values for MVPA for each participant and
                ROI

        """

        comp_pair_str = self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]

        p_val_all = []
        for sbj in range(self.n_sbjID):

            sbjID = self.sbjID_all[sbj]

            p_val_sbj = np.zeros((self.n_ROIs, 2), dtype=np.float32)

            for roi in range(self.n_ROIs):

                decode_mean = np.mean(
                    decode_all_df.loc[
                        (decode_all_df.sbjID == sbjID)
                        & (decode_all_df.roi == roi)
                        & (decode_all_df.nVox == nVox_to_analyze)
                        & (decode_all_df.comp_pair == comp_pair_str)
                    ].acc
                )

                permute_acc = permuteDecode_all_df.loc[
                    (permuteDecode_all_df.sbjID == sbjID)
                    & (permuteDecode_all_df.roi == roi)
                    & (permuteDecode_all_df.nVox == nVox_to_analyze)
                    & (permuteDecode_all_df.comp_pair == comp_pair_str)
                ].acc

                # calculate p_val
                p_val = np.sum(permute_acc >= decode_mean) / len(permute_acc)

                p_val_sbj[roi, 0] = decode_mean
                p_val_sbj[roi, 1] = p_val

            p_val_df = pd.DataFrame(p_val_sbj, columns=["acc_mean", "p_val"])
            p_val_df["sbjID"] = sbjID
            p_val_df["roi"] = np.arange(self.n_ROIs)
            p_val_df["roi_str"] = self.ROIs

            p_val_all.append(p_val_df.copy())

        p_val_all_df = pd.concat(p_val_all)

        return p_val_all_df

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

    def compute_stat_decode_permute_bootstrap_cond(
        self,
        decode_fold_df,
        decode_group_df,
        permute_group_df,
        nVox_to_analyze,
        n_bootstrap,
        c,
        alpha=0.05,
    ):
        """


        Parameters
        ----------
        decode_fold_df : TYPE
            DESCRIPTION.
        decode_group_df : TYPE
            DESCRIPTION.
        permute_group_df : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.025.

        Returns
        -------
        stat_df : TYPE
            DESCRIPTION.

        """

        comp_pair = self.comp_pair_all[c]
        comp_pair_str = self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
        # comp_pair_str = mvpa.conds[comp_pair[0]] + "," + mvpa.conds[comp_pair[1]]

        stat = np.zeros((self.n_ROIs, 5), dtype=np.float32)
        # stat = np.zeros((mvpa.n_ROIs, 5), dtype=np.float32)
        for roi in range(self.n_ROIs):
            # for roi in range(mvpa.n_ROIs):

            # get observed dataset
            rv_observed = np.array(
                decode_fold_df.loc[
                    (decode_fold_df.nVox == nVox_to_analyze)
                    & (decode_fold_df.roi == roi)
                    & (decode_fold_df.comp_pair == comp_pair_str)
                ].acc_cv
            )

            # generate boostrap distribution for empirical data
            rv_bootstrap = self.generate_bootstrap_dist(rv_observed, n_bootstrap)
            # rv_bootstrap = plot_mvpa.generate_bootstrap_dist(rv_observed, n_bootstrap)

            # filter dataframe
            permute_df = permute_group_df.loc[
                (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.roi == roi)
                & (permute_group_df.comp_pair == comp_pair_str)
            ]

            decode_df = decode_group_df.loc[
                (decode_group_df.nVox == nVox_to_analyze)
                & (decode_group_df.roi == roi)
                & (decode_group_df.comp_pair == comp_pair_str)
            ]

            mean_observation = np.float32(decode_df.acc.mean())
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            p_val = np.sum(rv_bootstrap < baseline) / n_bootstrap
            # p_val = np.sum(rv_permute>baseline)/n_bootstrap

            stat[roi, 0] = np.int32(roi)
            stat[roi, 1] = np.mean(rv_permute)
            stat[roi, 2] = baseline
            stat[roi, 3] = mean_observation
            stat[roi, 4] = p_val

        stat_df = pd.DataFrame(
            stat, columns=["roi", "mean_permute", "baseline", "mean_obs", "p_val"]
        )
        stat_df["comp_pair"] = comp_pair_str
        stat_df["roi_str"] = self.ROIs
        stat_df["nVox"] = nVox_to_analyze

        return stat_df

    def compute_stat_decode_permute_bootstrap(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        n_bootstrap,
        alpha=0.05,
    ):
        """
        Compute p-val based on permutation and bootstrap distributions.

        Parameters
        ----------
        decode_all_df : TYPE
            DESCRIPTION.
        permuteDecode_all_df : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        stat_df : TYPE
            DESCRIPTION.

        """

        # stat_all = []
        ## do statistical testing across sbjID

        # average across sbjID for each roi, nVox, permute_i, comp_pair
        permute_group_df = (
            permuteDecode_all_df.groupby(["roi", "nVox", "permute_i", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # take average on decode_all_df across fold_id and then across sbjID
        decode_fold_df = (
            decode_all_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_fold_df = decode_fold_df.rename(columns={"mean": "acc_cv"})

        decode_group_df = (
            decode_fold_df.groupby(["roi", "roi_str", "nVox", "comp_pair"])
            .acc_cv.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc"})

        stat_list = []
        stat_list.append(
            Parallel(n_jobs=-1)(
                delayed(self.compute_stat_decode_permute_bootstrap_cond)(
                    decode_fold_df,
                    decode_group_df,
                    permute_group_df,
                    nVox_to_analyze,
                    n_bootstrap,
                    c,
                    alpha,
                )
                for c in range(len(self.comp_pair_all))
            )
        )

        stat_df = pd.concat(stat_list[0], ignore_index=True)
        stat_df["partiality"] = "all"

        return stat_df

    def compute_stat_decode_vox_fixed_percent_permute_bootstrap_cond(
        self,
        decode_score_allSbj,
        permute_group_df,
        nVox_percentage_list,
        n_bootstrap,
        c,
        v,
        alpha=0.05,
    ):
        """


        Parameters
        ----------
        decode_score_allSbj : np.array [
                n_sbjID,
                n_ROIs,
                len(mvpa_decode.comp_pair_all),
                len(nVox_percentage_list)]
        roi order = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "hMT+"]

        permute_group_df,

        nVox_percentage_list : np.array([0.1, 0.2, 0.5, 0.75, 1])

        n_bootstrap : int

        c : int
            condition comparison index

        v : int
            voxel percentage index

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        stat_df : TYPE
            DESCRIPTION.

        """

        comp_pair = self.comp_pair_all[c]
        comp_pair_str = self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
        # comp_pair = mvpa_decode.comp_pair_all[c]
        # comp_pair_str = mvpa_decode.conds[comp_pair[0]] + "," + mvpa_decode.conds[comp_pair[1]]

        voxPercent_to_analyze = nVox_percentage_list[v]
        stat = np.zeros((self.n_ROIs, 5), dtype=np.float32)
        # stat = np.zeros((mvpa.n_ROIs, 5), dtype=np.float32)
        for roi in range(self.n_ROIs):
            # for roi in range(mvpa.n_ROIs):

            # get observed dataset
            rv_observed = decode_score_allSbj[:, roi, c, v]

            # generate boostrap distribution for empirical data
            rv_bootstrap = self.generate_bootstrap_dist(rv_observed, n_bootstrap)
            # rv_bootstrap = mvpa_decode.generate_bootstrap_dist(rv_observed, n_bootstrap)

            # filter dataframe
            permute_df = permute_group_df.loc[
                (permute_group_df.voxPercent == voxPercent_to_analyze)
                & (permute_group_df.roi == roi)
                & (permute_group_df.comp_pair == comp_pair_str)
            ]

            mean_observation = np.float32(rv_observed.mean())
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            p_val = np.sum(rv_bootstrap < baseline) / n_bootstrap
            # p_val = np.sum(rv_permute>baseline)/n_bootstrap

            stat[roi, 0] = np.int32(roi)
            stat[roi, 1] = np.mean(rv_permute)
            stat[roi, 2] = baseline
            stat[roi, 3] = mean_observation
            stat[roi, 4] = p_val

        stat_df = pd.DataFrame(
            stat, columns=["roi", "mean_permute", "baseline", "mean_obs", "p_val"]
        )
        stat_df["comp_pair"] = comp_pair_str
        stat_df["roi_str"] = self.ROIs
        stat_df["voxPercent"] = voxPercent_to_analyze

        return stat_df

    def compute_stat_decode_vox_fixed_percent_permute_bootstrap(
        self,
        decode_score_allSbj,
        decode_permute_allSbj_df,
        nVox_percentage_list,
        n_bootstrap,
        alpha=0.05,
    ):
        """
        Compute p-val based on permutation and bootstrap distributions.

        Parameters
        ----------
        decode_score_allSbj : [n_sbjID,
                    n_ROIs,
                    len(comp_pair_all),
                    len(nVox_percentage_list)]
            roi order = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "hMT+"]

        permuteDecode_all_df : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.
        n_bootstrap : TYPE
            DESCRIPTION.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        stat_df : TYPE
            DESCRIPTION.

        """

        # stat_all = []
        ## do statistical testing across sbjID

        # average across sbjID for each roi, voxPercent, permute_i, comp_pair
        permute_group_df = (
            decode_permute_allSbj_df.groupby(
                ["roi", "voxPercent", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # take average on decode_all across sbjID
        # [n_ROIs, len(comp_pair_all), len(nVox_percentage_list)]
        # decode_avg = decode_score_allSbj.mean(axis=0)

        stat_list = []
        stat_list.append(
            Parallel(n_jobs=-1)(
                delayed(
                    self.compute_stat_decode_vox_fixed_percent_permute_bootstrap_cond
                )(
                    decode_score_allSbj,
                    permute_group_df,
                    nVox_percentage_list,
                    n_bootstrap,
                    c,
                    v,
                    alpha,
                )
                for c in range(len(self.comp_pair_all))
                for v in range(len(nVox_percentage_list))
            )
        )

        stat_df = pd.concat(stat_list[0], ignore_index=True)
        stat_df["partiality"] = "all"

        return stat_df

    def compute_f_stat(self, decode_all_df, nVox_to_analyze):

        ## filter dataframe based on condition
        decode_vox = decode_all_df.loc[decode_all_df.nVox == nVox_to_analyze]

        # average across fold_id
        decode_avg = (
            decode_vox.groupby(["roi", "sbjID", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_avg = decode_avg.rename(columns={"mean": "acc"})

        aovrm2way = AnovaRM(decode_avg, "acc", "sbjID", within=["roi", "comp_pair"])
        res2way = aovrm2way.fit()

        print(res2way)