"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/MVPA/PlotMVPA_XDecode.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/
Created Date: 2022-06-20 22:57:31
Author: Bayu G. Wundari
-----
Last Modified: 2022-06-29 12:02:32
Modified By: Bayu G. Wundari

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
script for plotting MVPA cross-decoding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erfinv
from scipy.stats import sem
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import os

from Common.Common import PlotGeneral


# %%
class PlotMVPA_XDecode(PlotGeneral):
    def __init__(self):
        super().__init__()

        super().__init__()
        self.plot_dir = os.getcwd()[:-16] + "Plots/CMM/MVPA"
        self.stat_dir = os.getcwd()[:-16] + "Data/MVPA"

    def plotBox_xDecode_at_nVox(
        self,
        xDecode_all_df,
        permuteXDecode_all_df,
        nVox_to_analyze,
        rds_train_test,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average cross-decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

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

        alpha : scalar, optional
            significant level. The default is 0.025.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        xDecode_group_df = (
            xDecode_all_df.groupby(
                ["sbjID", "roi", "roi_str", "nVox", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_group_df = xDecode_group_df.rename(columns={"mean": "acc_cv_mean"})

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteXDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (9, 7)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM XDecoding, AVG, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        bar_width = 2
        step = 4
        pos_init = step * 8
        pos_roi = np.arange(0, pos_init, step)
        linewidth = 1

        # upper and lower boxplot y-axis
        x_low = pos_roi[0] - bar_width
        x_up = pos_roi[-1] + bar_width
        y_low = 0.3  # 0.2
        y_up = 0.81
        y_step = 0.1

        # collect data ards, hmrds, crds
        data = []
        baseline_list = []
        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa_xDecode.n_ROIs):

            # crossDecode_avg across all sbjID
            df = xDecode_group_df.loc[
                (xDecode_group_df["rds_train_test"] == rds_train_test)
                & (xDecode_group_df.roi == roi)
                & (xDecode_group_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data.append(np.array(df.acc_cv_mean))

            ## add permutation distribution data to the plot
            # permute_df = permute_group_df.loc[
            #     (permute_group_df.roi == roi)
            #     & (permute_group_df.nVox == nVox_to_analyze)
            #     & (permute_group_df.rds_train_test == rds_train_test)
            # ]
            permute_df = permute_group_df.loc[
                (permute_group_df.roi == roi)
                & (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.rds_train_test == rds_train_test)
            ]

            ## find baseline
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            baseline_list.append(baseline)

        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="x", markersize=15, markerfacecolor="blue", markeredgecolor="blue"
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)

        axes.boxplot(
            data,
            widths=bar_width,
            patch_artist=True,
            positions=pos_roi,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # add data point
        # for roi in range(self.n_ROIs):
        for roi in range(self.n_ROIs):
            y = data[roi]
            jitter = np.random.normal(0, 0.05, size=len(y))
            x = pos_roi[roi] + jitter
            axes.plot(x, y, ".", color="gray", markersize=15)

        # plot acc threshold
        prob_thresh = np.mean(baseline_list)  # 1 - np.mean(baseline_list)
        axes.plot(
            [x_low, x_up],
            [prob_thresh, prob_thresh],
            "r--",
            linewidth=3,
        )

        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        # y_ticklabels = []
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_xlabel("")
        axes.set_xticks(pos_roi)
        axes.set_xticklabels(self.ROI_plotname)
        # axes.set_xticklabels(plot_mvpa.ROIs)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                f"{self.plot_dir}/PlotBox_xDecode_AVG_{rds_train_test}_nVox_{nVox_to_analyze}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotBox_xDecode_hmrds_ards_at_nVox(
        self,
        xDecode_all_df,
        permuteXDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average cross-decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

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

        alpha : scalar, optional
            significant level. The default is 0.025.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        xDecode_group_df = (
            xDecode_all_df.groupby(
                ["sbjID", "roi", "roi_str", "nVox", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_group_df = xDecode_group_df.rename(columns={"mean": "acc_cv_mean"})

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteXDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (9, 7)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM XDecoding, AVG, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        bar_width = 2
        step = 4
        pos_init = step * 8
        pos_roi = np.arange(0, pos_init, step)
        linewidth = 1

        # upper and lower boxplot y-axis
        x_low = pos_roi[0] - bar_width
        x_up = pos_roi[-1] + bar_width
        y_low = 0.0
        y_up = 0.65
        y_step = 0.1

        # collect data ards, hmrds, crds
        data = []
        baseline_list = []
        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa_xDecode.n_ROIs):

            # crossDecode_avg across all sbjID
            df = xDecode_group_df.loc[
                (xDecode_group_df["rds_train_test"] == "hmrds_ards")
                & (xDecode_group_df.roi == roi)
                & (xDecode_group_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data.append(np.array(df.acc_cv_mean))

            ## add permutation distribution data to the plot
            # permute_df = permute_group_df.loc[
            #     (permute_group_df.roi == roi)
            #     & (permute_group_df.nVox == nVox_to_analyze)
            #     & (permute_group_df.rds_train_test == rds_train_test)
            # ]
            permute_df = permute_group_df.loc[
                (permute_group_df.roi == roi)
                & (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.rds_train_test == "hmrds_ards")
            ]

            ## find baseline
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            baseline_list.append(baseline)

        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="x", markersize=15, markerfacecolor="blue", markeredgecolor="blue"
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)

        axes.boxplot(
            data,
            widths=bar_width,
            patch_artist=True,
            positions=pos_roi,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # add data point
        # for roi in range(self.n_ROIs):
        for roi in range(self.n_ROIs):
            y = data[roi]
            jitter = np.random.normal(0, 0.05, size=len(y))
            x = pos_roi[roi] + jitter
            axes.plot(x, y, ".", color="gray", markersize=15)

        # plot acc threshold
        prob_thresh = 1 - np.mean(baseline_list)
        axes.plot(
            [x_low, x_up],
            [prob_thresh, prob_thresh],
            "r--",
            linewidth=3,
        )

        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        # y_ticklabels = []
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_xlabel("")
        axes.set_xticks(pos_roi)
        axes.set_xticklabels(self.ROI_plotname)
        # axes.set_xticklabels(plot_mvpa.ROIs)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                f"{self.plot_dir}/PlotBox_xDecode_AVG_hmrds_ards_nVox_{nVox_to_analyze}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotBox_xDecode_at_nVox_all_comparison(
        self,
        decode_all_df,
        xDecode_crds_vs_ards_df,
        xDecode_crds_vs_hmrds_df,
        permuteXDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average cross-decoding accuracy at nVox_to_analyze
            for all ROIs, and all comparison: train vs. test
            cRDS vs aRDS, cRDS vs hmrds, cRDS vs cRDS

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

        alpha : scalar, optional
            significant level. The default is 0.05.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        # this is only for getting cRDS decoding results
        decode_group_df = (
            decode_all_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc_cv_mean"})

        # average xDecode_crds_vs_ards_df across fold_id for each sbjID
        xDecode_group_crds_ards_df = (
            xDecode_crds_vs_ards_df.groupby(
                ["sbjID", "roi", "roi_str", "nVox", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_group_crds_ards_df = xDecode_group_crds_ards_df.rename(
            columns={"mean": "acc_cv_mean"}
        )

        # average xDecode_crds_vs_hmrds_df across fold_id for each sbjID
        xDecode_group_crds_hmrds_df = (
            xDecode_crds_vs_hmrds_df.groupby(
                ["sbjID", "roi", "roi_str", "nVox", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_group_crds_hmrds_df = xDecode_group_crds_hmrds_df.rename(
            columns={"mean": "acc_cv_mean"}
        )

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteXDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # collect data ards, hmrds, crds
        data_all_rds = {}
        baseline_all_rds = {}

        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa.n_ROIs):
            data = []
            baseline_list = []

            # crds vs ards: crossDecode_avg across all sbjID
            df = xDecode_group_crds_ards_df.loc[
                (xDecode_group_crds_ards_df["rds_train_test"] == "ards_crds")
                & (xDecode_group_crds_ards_df.roi == roi)
                & (xDecode_group_crds_ards_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data.append(np.array(df.acc_cv_mean))

            # crds vs hmrds: crossDecode_avg across all sbjID
            df = xDecode_group_crds_hmrds_df.loc[
                (xDecode_group_crds_hmrds_df["rds_train_test"] == "hmrds_crds")
                & (xDecode_group_crds_hmrds_df.roi == roi)
                & (xDecode_group_crds_hmrds_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data.append(np.array(df.acc_cv_mean))

            # crds vs crds
            df = decode_group_df.loc[
                (decode_group_df["comp_pair"] == "n_c100,f_c100")
                & (decode_group_df.roi == roi)
                & (decode_group_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data.append(np.array(df.acc_cv_mean))

            ## add permutation distribution data to the plot
            # permute_df = permute_group_df.loc[
            #     (permute_group_df.roi == roi)
            #     & (permute_group_df.nVox == nVox_to_analyze)
            #     & (permute_group_df.rds_train_test == rds_train_test)
            # ]
            permute_df = permute_group_df.loc[
                (permute_group_df.roi == roi)
                & (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.rds_train_test == "crds_ards")
            ]

            ## find baseline
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            baseline_list.append(baseline)

            # gather all rds data for all roi into a dict
            data_all_rds[self.ROIs[roi]] = data
            baseline_all_rds[self.ROIs[roi]] = baseline_list

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (16, 12)
        n_row = 2
        n_col = 4
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM XDecoding, Train: cRDS, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)
        fig.text(0.5, -0.03, "Dot correlation of tested RDS", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.5, hspace=0.3)

        bar_width = 2
        pos_roi = [0, 2 * bar_width, 4 * bar_width]

        # upper and lower boxplot y-axis
        x_low = pos_roi[0] - bar_width
        x_up = pos_roi[-1] + bar_width
        y_low = 0.2
        y_up = 1.01
        y_step = 0.2

        boxprops = dict(
            linewidth=3, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=3, color="black")
        meanprops = dict(
            marker="x", markersize=17, markerfacecolor="blue", markeredgecolor="blue"
        )
        whiskerprops = dict(linewidth=3)
        capprops = dict(linewidth=3)

        for roi in range(self.n_ROIs):
            id_row = roi // n_col
            id_col = roi % n_col

            data = data_all_rds[self.ROIs[roi]]
            baseline_list = baseline_all_rds[self.ROIs[roi]]

            axes[id_row, id_col].boxplot(
                data,
                widths=bar_width,
                patch_artist=True,
                positions=pos_roi,
                medianprops=medianprops,
                boxprops=boxprops,
                meanprops=meanprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                showfliers=False,
                showmeans=True,
            )

            # add data point
            # for roi in range(self.n_ROIs):
            for i in range(3):  # 3 comparisons
                y = data[i]
                jitter = np.random.normal(0, 0.05, size=len(y))
                x = pos_roi[i] + jitter
                axes[id_row, id_col].plot(x, y, ".", color="gray", markersize=12)

            # plot acc threshold
            prob_thresh = np.mean(baseline_list)  # 1 - np.mean(baseline_list)
            axes[id_row, id_col].plot(
                [x_low, x_up],
                [prob_thresh, prob_thresh],
                "r--",
                linewidth=3,
            )

            axes[id_row, id_col].set_ylim(y_low, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low, y_up, y_step), 2)
            )

            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xticks(pos_roi)
            axes[id_row, id_col].set_xticklabels([-1.0, 0.0, 1.0])

            # set title
            axes[id_row, id_col].set_title(self.ROIs[roi])

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                "../../../../Plots/CMM/MVPA/PlotBox_xDecode_AVG_all_comp_nVox_{}.pdf".format(
                    nVox_to_analyze
                ),
                dpi=600,
                bbox_inches="tight",
            )

    def plotBox_xDecode_disp_bias_at_nVox(
        self, xDecode_all_df, nVox_to_analyze, rds_train_test, save_flag
    ):
        """
        box plot the average cross-decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

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

        alpha : scalar, optional
            significant leve. The default is 0.025.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        xDecode_group_df = (
            xDecode_all_df.groupby(
                ["sbjID", "roi", "roi_str", "nVox", "rds_train_test"]
            )
            .agg("mean")
            .reset_index()
        )

        # average permutation distribution across sbjID
        # permute_group_df = (
        #     permuteXDecode_all_df.groupby(
        #         ["roi", "roi_str", "nVox", "permute_i", "rds_train_test"]
        #     )
        #     .acc.agg(["mean"])
        #     .reset_index()
        # )
        # permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # %% ## plot
        plt.style.use("seaborn-colorblind")
        sns.set()
        sns.set(context="paper", style="white", font_scale=3, palette="deep")

        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = "Ubuntu"
        # plt.rcParams["font.monospace"] = "Ubuntu Mono"
        # plt.rcParams["axes.labelweight"] = "bold"

        figsize = (9, 9)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM XDecoding, disparity bias, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        bar_width = 2
        step = 6
        pos_init = step * 8
        pos_near_roi = np.arange(0, pos_init, step)
        pos_far_roi = np.arange(2, pos_init + 2, step)
        linewidth = 1

        # upper and lower boxplot y-axis
        x_low = pos_near_roi[0] - bar_width
        x_up = pos_far_roi[-1] + bar_width
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        # collect data ards, hmrds, crds
        data_y_near_given_x_near = []
        data_y_far_given_x_far = []
        # baseline_list = []
        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa_xDecode.n_ROIs):
            # crossDecode_avg across all sbjID
            df = xDecode_group_df.loc[
                (xDecode_group_df["rds_train_test"] == rds_train_test)
                & (xDecode_group_df.roi == roi)
                & (xDecode_group_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc) != 0:
                data_y_near_given_x_near.append(np.array(df.acc_y_near_given_x_near))
                data_y_far_given_x_far.append(np.array(df.acc_y_far_given_x_far))

            ## add permutation distribution data to the plot
            # permute_df = permute_group_df.loc[
            #     (permute_group_df.roi == roi)
            #     & (permute_group_df.nVox == nVox_to_analyze)
            #     & (permute_group_df.rds_train_test == rds_train_test)
            # ]

            # ## find baseline
            # rv_permute = np.float32(permute_df.acc)

            # # calculate p_val: proportions of rv_bootstrap which is less than
            # # val_threshold (the right tail of rv_permute).
            # # this stat test is very strict..
            # # alpha_corrected = alpha / (2 * self.n_ROIs)
            # baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            # baseline_list.append(baseline)

        # add data point
        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa_xDecode.n_ROIs):
            # y_near_given_x_near
            y = data_y_near_given_x_near[roi]
            jitter = np.random.normal(0, 0.05, size=len(y))
            x = pos_near_roi[roi] + jitter
            axes.plot(x, y, ".", color="red", markersize=6)

            # y_far_given_x_far
            y = data_y_far_given_x_far[roi]
            jitter = np.random.normal(0, 0.05, size=len(y))
            x = pos_far_roi[roi] + jitter
            axes.plot(x, y, ".", color="blue", markersize=6)

        # legend
        axes.legend(
            [
                "P(y=near|x=near)",
                "P(y=far|x=far)",
            ],
            fontsize=16,
        )

        ## plot y_near_given_x_near
        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="D", markersize=6, markerfacecolor="red", markeredgecolor="red"
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)
        axes.boxplot(
            data_y_near_given_x_near,
            widths=bar_width,
            patch_artist=True,
            positions=pos_near_roi,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        ## plot y_far_given_x_far
        meanprops = dict(
            marker="D",
            markersize=6,
            markerfacecolor="blue",
            markeredgecolor="blue",
        )
        axes.boxplot(
            data_y_far_given_x_far,
            widths=bar_width,
            patch_artist=True,
            positions=pos_far_roi,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # plot acc threshold
        # prob_thresh = np.mean(baseline_list)
        # axes.plot(
        #     [x_low, x_up],
        #     [prob_thresh, prob_thresh],
        #     "r",
        #     linewidth=2,
        # )
        # axes.plot(
        #     [x_low, x_up],
        #     [1 - prob_thresh, 1 - prob_thresh],
        #     "r",
        #     linewidth=2,
        # )
        axes.plot([x_low, x_up], [0.5, 0.5], "k--", linewidth=2)

        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        # y_ticklabels = []
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_xlabel("")
        axes.set_xticks((pos_near_roi + pos_far_roi) / 2)
        axes.set_xticklabels(self.ROIs)
        # axes.set_xticklabels(plot_mvpa_xDecode.ROIs)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # %%
        if save_flag == 1:
            fig.savefig(
                "../../../../Plots/CMM/MVPA/PlotBox_xDecode_disp_bias_{}_nVox_{}.pdf".format(
                    xDecode_group_df.rds_train_test.unique()[0], nVox_to_analyze
                ),
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_xDdecode_avg(
        self,
        xDecode_all_df,
        permuteXDecode_all_df,
        rds_train_test,
        save_flag,
        alpha=0.05,
    ):
        nVox_all = xDecode_all_df.nVox.unique()

        ## average across sbjID
        # average xDecode_all_df across fold_id and then across sbjID
        temp_df = (
            xDecode_all_df.groupby(
                ["sbjID", "roi", "nVox", "roi_str", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        temp_df = temp_df.rename(columns={"mean": "acc_cv"})

        xDecode_group_df = (
            temp_df.groupby(["roi", "nVox", "roi_str", "rds_train_test"])
            .acc_cv.agg(["mean", sem])
            .reset_index()
        )
        xDecode_group_df = xDecode_group_df.rename(
            columns={"mean": "acc_group_mean", "sem": "acc_group_sem"}
        )

        # compute probability threshold in each ROI from permutation dataset
        permute_group_df = (
            permuteXDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "rds_train_test"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        ## plot All group
        plt.style.use("seaborn-colorblind")
        sns.set()
        sns.set(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (18, 18)
        n_row = 3
        n_col = 3
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(-0.02, 0.5, "Prediction Accuracy", va="center", rotation=90)
        fig.text(0.5, -0.03, "# voxels", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        y_low = 0.45
        y_up = 0.66
        y_step = 0.05
        x_low = 0
        x_up = nVox_all[-1] + 26
        x_step = 50

        for roi in range(self.n_ROIs):
            id_row = roi // n_row
            id_col = roi % n_col

            ## plot crossDecode_avg of ALL group
            xDecode_df = xDecode_group_df.loc[
                (xDecode_group_df.roi == roi)
                & (xDecode_group_df.rds_train_test == rds_train_test)
            ]
            xDecode_df.plot.line(
                x="nVox",
                y="acc_group_mean",
                yerr="acc_group_sem",
                capsize=4,
                elinewidth=2,
                ecolor="k",
                color="black",
                linewidth=4,
                ax=axes[id_row, id_col],
            )

            ## get shuffled baseline
            baseline = np.zeros(len(nVox_all), dtype=np.float32)
            for v in range(len(nVox_all)):
                nVox = nVox_all[v]
                permute_df = permute_group_df.loc[
                    (permute_group_df.roi == roi)
                    & (permute_group_df.nVox == nVox)
                    & (permute_group_df.rds_train_test == rds_train_test)
                ]

                ## find baseline
                rv_permute = np.float32(permute_df.acc)

                ## calculate p_val: proportions of rv_bootstrap which is less than
                # val_threshold (the right tail of rv_permute).
                # this stat test is very strict..
                # alpha_corrected = alpha/(2*len(ROIs))
                baseline[v] = np.percentile(rv_permute, (1 - alpha) * 100)

            # draw the prob threshold line
            prob_thresh = baseline.mean()

            axes[id_row, id_col].plot(
                [x_low, x_up], [prob_thresh, prob_thresh], "r--", linewidth=3
            )

            axes[id_row, id_col].legend().set_visible(False)
            axes[id_row, id_col].set_title(self.ROIs[roi])
            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_xticklabels(
                np.arange(x_low, x_up, x_step, dtype=np.int32), rotation=45
            )

            axes[id_row, id_col].set_ylim(y_low, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low + 0.05, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low + 0.05, y_up, y_step), 2)
            )

            # remove the top and right frames
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        ## dummy plot, only for setting up the legend
        axes[id_row, id_col + 1].plot(
            [x_low, x_up],
            [prob_thresh, prob_thresh],
            color="black",
            linewidth=3,
        )

        axes[id_row, id_col + 1].plot(
            [x_low, x_up], [prob_thresh, prob_thresh], "r--", linewidth=3
        )
        axes[id_row, id_col + 1].legend(
            ["cross-decoding", "shuffled-baseline"],
            fontsize=18,
        )

        if save_flag == 1:
            fig.savefig(
                "../../../../Plots/CMM/MVPA/PlotLine_xDecode_AVG_{}.pdf".format(
                    xDecode_group_df.rds_train_test.unique()[0]
                ),
                dp=600,
                bbox_inches="tight",
            )

    def plotScatter_xDecode_ards_vs_decode_hmrds_each_roi(
        self, nVox_to_analyze, alpha=0.05, n_permute=10000
    ):
        """
        plot xDecode ards-crds vs decode hmrds for each roi

        Parameters
        ----------
        nVox_to_analyze : scalar
            number of voxels used for the analyses (e.g. 250)
        alpha : scalar, optional
            level of confidence. The default is 0.05.
        n_permute : scalar, optional
            number of permutation. The default is 10000.

        Returns
        -------
        None.

        """
        xDecode_copy_df = self.xDecode_crds_ards_twoway_flip_df.copy()
        decode_copy_df = self.decode_all_df.copy()
        permute_decode_copy_df = self.permuteDecode_all_df.copy()
        permute_xDecode_copy_df = self.permuteXDecode_all_df.copy()

        # average permute_decode_df across sbjID for each roi, nVox, permute_i, comp_pair
        permute_decode_group_df = (
            permute_decode_copy_df.groupby(["roi", "nVox", "permute_i", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_decode_group_df = permute_decode_group_df.rename(
            columns={"mean": "acc"}
        )

        # average permute_xDecode_df across sbjID for each roi, nVox, permute_i, comp_pair
        permute_xDecode_group_df = (
            permute_xDecode_copy_df.groupby(["roi", "nVox", "permute_i"])
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_xDecode_group_df = permute_xDecode_group_df.rename(
            columns={"mean": "acc"}
        )

        # average decode_all_df across fold_id and then across sbjID
        decode_group_df = (
            decode_copy_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc"})

        # average decode_all_df across fold_id and rds_train_test
        xDecode_group_df = (
            xDecode_copy_df.groupby(["sbjID", "roi", "roi_str", "nVox"])
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_group_df = xDecode_group_df.rename(columns={"mean": "acc"})

        # get shuffled baseline for hmrds decoding
        # alpha = 0.05 #
        temp_df = permute_decode_group_df.loc[
            (permute_decode_group_df.nVox == nVox_to_analyze)
            & (permute_decode_group_df.comp_pair == "n_c50,f_c50")
        ]
        decode_thresh_df = self._compute_acc_thresh(temp_df, alpha, n_permute)
        # decode_thresh_df = mvpa._compute_acc_thresh(temp_df,
        #                                           alpha, n_permute)

        # get shuffled baseline for ards-crds xDecode
        # alpha = 0.05 #
        temp_df = permute_xDecode_group_df.loc[
            (permute_xDecode_group_df.nVox == nVox_to_analyze)
        ]
        xDecode_thresh_df = self._compute_acc_thresh(temp_df, alpha, n_permute)
        # xDecode_thresh_df = mvpa._compute_acc_thresh(temp_df,
        #                                              alpha, n_permute)

        ## plot All group
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

        fig.text(
            0.5, 1.03, "SVM Cross-Decoding aRDS-cRDS VS. Decoding hmRDS", ha="center"
        )
        fig.text(
            -0.02,
            0.5,
            "Prop. correct cross-decoding aRDS-cRDS",
            va="center",
            rotation=90,
        )
        fig.text(0.5, -0.02, "Prop. correct decoding hmRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.3
        x_up = 0.91
        x_step = 0.1
        y_low = 0.3
        y_up = 0.91
        y_step = 0.1

        for roi in range(len(self.ROIs)):
            id_row = np.int(roi / n_row)
            id_col = roi % n_col

            ards = np.array(
                xDecode_group_df.loc[
                    (xDecode_group_df.nVox == nVox_to_analyze)
                    & (xDecode_group_df.roi == roi + 1)
                ].acc,
                dtype=np.float32,
            )

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi + 1)
                    & (decode_group_df.comp_pair == "n_c50,f_c50")
                ].acc,
                dtype=np.float32,
            )

            axes[id_row, id_col].plot(hmrds, ards, "k.", ms=25)

            ## plot shuffled baseline
            thresh = xDecode_thresh_df.loc[xDecode_thresh_df.roi == roi + 1].acc_thresh
            axes[id_row, id_col].plot(
                [x_low, x_up], [thresh, thresh], "r--", linewidth=4
            )
            thresh = decode_thresh_df.loc[decode_thresh_df.roi == roi + 1].acc_thresh
            axes[id_row, id_col].plot(
                [thresh, thresh], [y_low, y_up], "r--", linewidth=4
            )

            axes[id_row, id_col].set_title(self.ROIs[roi])

            ## linear fit ards vs hmrds
            reg = LinearRegression().fit(hmrds.reshape(-1, 1), ards.reshape(-1, 1))

            # slope
            slope = reg.coef_[0][0]
            y_predict = reg.predict(hmrds.reshape(-1, 1))
            r2 = explained_variance_score(ards, y_predict)

            # plot linear fit
            x = np.array([x_low, x_up])
            y = reg.predict(x.reshape(-1, 1))
            axes[id_row, id_col].plot(x, y, "b--", linewidth=4)

            axes[id_row, id_col].text(
                0.6,
                0.31,
                "slope={}\nR2={}".format(str(np.round(slope, 2)), str(np.round(r2, 2))),
            )

            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_xticklabels(
                np.round(np.arange(x_low, x_up, x_step), 2)
            )

            axes[id_row, id_col].set_ylabel("")
            axes[id_row, id_col].set_ylim(y_low, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low, y_up, y_step), 2)
            )

            # remove the top and right frames
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        fig.savefig(
            "../../../../Plots/CMM/MVPA/PlotScatter_xDecode_ards_vs_decode_hmrds.pdf",
            dpi=600,
            bbox_inches="tight",
        )

    def plotScatter_decode_ards_vs_decode_hmrds_each_roi(
        self, nVox_to_analyze, alpha=0.05, n_permute=10000
    ):
        """
        plot decode ards vs decode hmrds for each roi

        Parameters
        ----------
        nVox_to_analyze : scalar
            number of voxels used for the analyses (e.g. 250)
        alpha : scalar, optional
            level of confidence. The default is 0.05.
        n_permute : scalar, optional
            number of permutation. The default is 10000.

        Returns
        -------
        None.

        """
        decode_copy_df = self.decode_all_df.copy()
        permute_decode_copy_df = self.permuteDecode_all_df.copy()

        # average permute_decode_df across sbjID for each roi, nVox, permute_i, comp_pair
        permute_decode_group_df = (
            permute_decode_copy_df.groupby(["roi", "nVox", "permute_i", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_decode_group_df = permute_decode_group_df.rename(
            columns={"mean": "acc"}
        )

        # average decode_all_df across fold_id and then across sbjID
        decode_group_df = (
            decode_copy_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc"})

        # get shuffled baseline for ards decoding
        temp_df = permute_decode_group_df.loc[
            (permute_decode_group_df.nVox == nVox_to_analyze)
            & (permute_decode_group_df.comp_pair == "n_c0,f_c0")
        ]
        ards_thresh_df = self._compute_acc_thresh(temp_df, alpha, n_permute)

        # get shuffled baseline for hmrds decoding
        # alpha = 0.05 #
        temp_df = permute_decode_group_df.loc[
            (permute_decode_group_df.nVox == nVox_to_analyze)
            & (permute_decode_group_df.comp_pair == "n_c50,f_c50")
        ]
        hmrds_thresh_df = self._compute_acc_thresh(temp_df, alpha, n_permute)
        # decode_thresh_df = mvpa._compute_acc_thresh(temp_df,
        #                                           alpha, n_permute)

        ## plot All group
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

        fig.text(0.5, 1.03, "SVM Decoding aRDS VS. Decoding hmRDS", ha="center")
        fig.text(-0.02, 0.5, "Prop. correct decoding aRDS", va="center", rotation=90)
        fig.text(0.5, -0.02, "Prop. correct decoding hmRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.3
        x_up = 1.01
        x_step = 0.1
        y_low = 0.3
        y_up = 1.01
        y_step = 0.1

        for roi in range(len(self.ROIs)):
            id_row = np.int(roi / n_row)
            id_col = roi % n_col

            ards = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi + 1)
                    & (decode_group_df.comp_pair == "n_c0,f_c0")
                ].acc,
                dtype=np.float32,
            )

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi + 1)
                    & (decode_group_df.comp_pair == "n_c50,f_c50")
                ].acc,
                dtype=np.float32,
            )

            axes[id_row, id_col].plot(hmrds, ards, "k.", ms=25)

            ## plot shuffled baseline
            thresh = ards_thresh_df.loc[ards_thresh_df.roi == roi + 1].acc_thresh
            axes[id_row, id_col].plot(
                [x_low, x_up], [thresh, thresh], "r--", linewidth=4
            )
            thresh = hmrds_thresh_df.loc[hmrds_thresh_df.roi == roi + 1].acc_thresh
            axes[id_row, id_col].plot(
                [thresh, thresh], [y_low, y_up], "r--", linewidth=4
            )

            axes[id_row, id_col].set_title(self.ROIs[roi])

            ## linear fit ards vs hmrds
            reg = LinearRegression().fit(hmrds.reshape(-1, 1), ards.reshape(-1, 1))

            # slope
            slope = reg.coef_[0][0]
            y_predict = reg.predict(hmrds.reshape(-1, 1))
            r2 = explained_variance_score(ards, y_predict)

            # plot linear fit
            x = np.array([x_low, x_up])
            y = reg.predict(x.reshape(-1, 1))
            axes[id_row, id_col].plot(x, y, "b--", linewidth=4)

            axes[id_row, id_col].text(
                0.7,
                0.31,
                "slope={}\nR2={}".format(str(np.round(slope, 2)), str(np.round(r2, 2))),
            )

            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_xticklabels(
                np.round(np.arange(x_low, x_up, x_step), 2)
            )

            axes[id_row, id_col].set_ylabel("")
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

        fig.savefig(
            "../../../../Plots/CMM/MVPA/PlotScatter_decode_ards_vs_decode_hmrds.pdf",
            dpi=600,
            bbox_inches="tight",
        )

    def plotScatter_decode_ards_vs_decode_hmrds_all_roi(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        plot decode ards vs decode hmrds for all roi in one plot

        Parameters
        ----------
        nVox_to_analyze : scalar
            number of voxels used for the analyses (e.g. 250)

        Returns
        -------
        None.

        """

        # average decode_all_df across fold_id for each sbjID
        decode_group_df = (
            decode_all_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc"})

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        ## compute shuffled-baseline for ards and hmrds
        baseline_list_ards = []
        baseline_list_hmrds = []
        for roi in range(self.n_ROIs):
            ## find shuffled-baseline for ards
            permute_df = permute_group_df.loc[
                (permute_group_df.roi == roi)
                & (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.comp_pair == "n_c0,f_c0")
            ]

            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            baseline_list_ards.append(baseline)

            ## find shuffled-baseline for hmrds
            permute_df = permute_group_df.loc[
                (permute_group_df.roi == roi)
                & (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.comp_pair == "n_c50,f_c50")
            ]

            ## find baseline
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict..
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            baseline_list_hmrds.append(baseline)

        ## plot All group
        plt.style.use("seaborn-colorblind")
        sns.set()
        sns.set(context="paper", style="white", font_scale=6, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        figsize = (18, 18)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(
            0.5,
            1.03,
            "SVM Decoding aRDS VS. Decoding hmRDS nVox=%s" % str(nVox_to_analyze),
            ha="center",
        )
        fig.text(-0.04, 0.5, "Pred. accuracy decoding aRDS", va="center", rotation=90)
        fig.text(0.5, -0.02, "Pred. accuracy decoding hmRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.5
        x_up = 0.701
        x_step = 0.05
        y_low = 0.5
        y_up = 0.701
        y_step = 0.05

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        for roi in range(len(self.ROIs)):
            # for roi in range(len(mvpa.ROIs)):

            ards = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c0,f_c0")
                ].acc,
                dtype=np.float32,
            )

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c50,f_c50")
                ].acc,
                dtype=np.float32,
            )

            x = np.mean(hmrds)
            y = np.mean(ards)
            x_err = sem(hmrds)
            y_err = sem(ards)

            axes.errorbar(
                x,
                y,
                xerr=x_err,
                yerr=y_err,
                elinewidth=4,
                fmt=markers[roi],
                c="black",
                ms=22,
            )

        axes.legend(self.ROIs, bbox_to_anchor=(1.05, 1))
        # axes.legend(mvpa.ROIs, bbox_to_anchor=(1.05, 1))

        # draw diagonal line
        axes.plot([x_low - x_step, x_up], [x_low - x_step, x_up], "r--", linewidth=4)
        # draw shuffled-chance level
        y_chance = np.max(baseline_list_ards)
        axes.plot([x_low - x_step, x_up], [y_chance, y_chance], "r--", linewidth=4)
        x_chance = np.max(baseline_list_hmrds)
        axes.plot([x_chance, x_chance], [y_low - y_step, y_up], "r--", linewidth=4)

        axes.set_xlabel("")
        axes.set_xlim(x_low - x_step, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))

        axes.set_ylabel("")
        axes.set_ylim(y_low - y_step, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                "../../../../Plots/CMM/MVPA/PlotScatter_decode_ards_vs_decode_hmrds_allROIs.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotScatter_xDecode_ards_vs_decode_hmrds_all_roi(self, nVox_to_analyze):
        """
        plot decode ards vs decode hmrds for all roi in one plot

        Parameters
        ----------
        nVox_to_analyze : scalar
            number of voxels used for the analyses (e.g. 250)

        Returns
        -------
        None.

        """
        decode_copy_df = self.decode_all_df.copy()
        xDecode_copy_df = self.xDecode_crds_ards_twoway_flip_df.copy()
        # decode_copy_df = mvpa.decode_all_df.copy()
        # xDecode_copy_df = mvpa.xDecode_crds_ards_twoway_flip_df.copy()

        # average decode_all_df across fold_id and then across sbjID
        decode_group_df = (
            decode_copy_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc"})

        # average decode_all_df across fold_id and rds_train_test
        xDecode_group_df = (
            xDecode_copy_df.groupby(["sbjID", "roi", "roi_str", "nVox"])
            .acc.agg(["mean"])
            .reset_index()
        )
        xDecode_group_df = xDecode_group_df.rename(columns={"mean": "acc"})

        ## plot All group
        plt.style.use("seaborn-colorblind")
        sns.set()
        sns.set(context="paper", style="white", font_scale=3, palette="deep")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Ubuntu"
        plt.rcParams["font.monospace"] = "Ubuntu Mono"
        plt.rcParams["axes.labelweight"] = "bold"

        figsize = (7, 7)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(
            0.5, 1.03, "SVM Cross-Decoding aRDS-cRDS VS. Decoding hmRDS", ha="center"
        )
        fig.text(
            -0.09,
            0.5,
            "Prop. correct\ncross-decoding aRDS-cRDS",
            va="center",
            ha="center",
            rotation=90,
        )
        fig.text(0.5, -0.07, "Prop. correct\ndecoding hmRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.45
        x_up = 0.66
        x_step = 0.05
        y_low = 0.45
        y_up = 0.66
        y_step = 0.05

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        for roi in range(len(self.ROIs)):
            # for roi in range(len(mvpa.ROIs)):

            ards = np.array(
                xDecode_group_df.loc[
                    (xDecode_group_df.nVox == nVox_to_analyze)
                    & (xDecode_group_df.roi == roi + 1)
                ].acc,
                dtype=np.float32,
            )

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi + 1)
                    & (decode_group_df.comp_pair == "n_c50,f_c50")
                ].acc,
                dtype=np.float32,
            )

            x = np.mean(hmrds)
            y = np.mean(ards)
            x_err = sem(hmrds)
            y_err = sem(ards)

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
        # axes.legend(mvpa.ROIs,
        #             bbox_to_anchor=(1.05, 1))

        axes.plot([x_low, x_up], [x_low, x_up], "r--", linewidth=4)

        axes.set_xlabel("")
        axes.set_xlim(x_low, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))

        axes.set_ylabel("")
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        fig.savefig(
            "../../../../Plots/CMM/MVPA/PlotScatter_xDecode_ards_vs_decode_hmrds_allROIs.pdf",
            dpi=600,
            bbox_inches="tight",
        )
