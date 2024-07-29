"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/MVPA/PlotMVPA_Decode.py
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
script for plotting MVPA
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
class PlotMVPA_Decode(PlotGeneral):
    def __init__(self):

        super().__init__()
        self.plot_dir = os.getcwd()[:-16] + "Plots/CMM/MVPA"
        self.stat_dir = os.getcwd()[:-16] + "Data/MVPA"

    def plotBox_decode_at_nVox(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

        Parameters
        ----------
        decode_all_df: pd.DataFrame
                            [nVox, roi, fold_id, acc, acc_mean, acc_sem, sbjID,
                             roi_str, comp_pair].
                dataframe containing the decoding performance for each participant

            permuteDecode_all_df: pd.DataFrame
                            [acc, sbjID, roi, roi_str, nVox, permute_i, comp_pair]
                dataframe containing the distribution of decoding permutation
                (10000 iterations)

        nVox_to_analyze: scalar
                the number of voxels used for analysis

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.025.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        decode_group_df = (
            decode_all_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc_cv_mean"})

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # comparison pairs for plotting
        comp_for_plot = [[5, 6], [3, 4], [1, 2]]  # crds, hmrds, ards

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=6, palette="deep")

        # figsize = (18, 6)
        figsize = (20, 18)
        n_row = 2
        n_col = 4
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM Decoding, AVG, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        bar_width = 6
        step = 24
        pos_init = step * 8
        pos_roi = [
            np.arange(0, pos_init, step),  # ards
            np.arange(bar_width, pos_init + bar_width, step),  # hmrds
            np.arange(2 * bar_width, pos_init + 2 * bar_width, step),
        ]  # crds

        # upper and lower boxplot y-axis
        yBox_low = 0.2
        yBox_up = 1.01

        # collect data
        for i in range(len(comp_for_plot)):

            comp_pair = comp_for_plot[i]
            comp_pair_str = self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
            # comp_pair_str = plot_mvpa_decode.conds[comp_pair[0]] + "," + plot_mvpa_decode.conds[comp_pair[1]]

            # collect data ards, hmrds, crds
            data = []
            baseline_list = []
            for roi in range(self.n_ROIs):
                # for roi in range(plot_mvpa.n_ROIs):

                # crossDecode_avg across all sbjID
                df = decode_group_df.loc[
                    (decode_group_df["comp_pair"] == comp_pair_str)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.nVox == nVox_to_analyze)
                ]
                if len(df.acc_cv_mean) != 0:
                    data.append(np.array(df.acc_cv_mean))

                ## add permutation distribution data to the plot
                permute_df = permute_group_df.loc[
                    (permute_group_df.roi == roi)
                    & (permute_group_df.nVox == nVox_to_analyze)
                    & (permute_group_df.comp_pair == comp_pair_str)
                ]

                ## find baseline
                rv_permute = np.float32(permute_df.acc)

                ## calculate p_val: proportions of rv_bootstrap which is less than
                # val_threshold (the right tail of rv_permute).
                # this stat test is very strict..
                # alpha_corrected = alpha/(2*len(ROIs))
                baseline = np.percentile(rv_permute, (1 - alpha) * 100)
                baseline_list.append(baseline)

            axes.boxplot(
                data,
                widths=bar_width,
                patch_artist=True,
                positions=pos_roi[i],
                medianprops=self.medianprops,
                boxprops=self.boxprops,
                meanprops=self.meanprops,
                whiskerprops=self.whiskerprops,
                capprops=self.capprops,
                showfliers=False,
                showmeans=True,
            )

            # add data point
            # for roi in range(self.n_ROIs):
            for roi in range(self.n_ROIs):
                y = data[roi]
                jitter = np.random.normal(0, 0.05, size=len(y))
                x = pos_roi[i][roi] + jitter
                axes.plot(x, y, ".", color=self.color_point[i], markersize=18)

            # plot acc threshold
            prob_thresh = np.mean(baseline_list)
            axes.plot(
                [pos_roi[0][0] - 2 * bar_width, pos_roi[-1][-1] + 2 * bar_width],
                [prob_thresh, prob_thresh],
                "r",
                linewidth=3,
            )

        axes.set_ylim(yBox_low, yBox_up)
        axes.set_yticks(np.arange(yBox_low, yBox_up, 0.1))
        # y_ticklabels = []
        axes.set_yticklabels(np.round(np.arange(yBox_low, yBox_up, 0.1), 2))

        axes.set_xlim(pos_roi[0][0] - 2 * bar_width, pos_roi[-1][-1] + 2 * bar_width)
        axes.set_xlabel("")
        axes.set_xticks(pos_roi[1])
        axes.set_xticklabels(self.ROIs)
        # axes.set_xticklabels(plot_mvpa.ROIs)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotBox_decode_AVG_%s.pdf"
                % (str(nVox_to_analyze)),
                dpi=600,
                bbox_inches="tight",
            )

    def plotBox_decode_at_nVox_v2(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

        Parameters
        ----------
        decode_all_df: pd.DataFrame
                            [nVox, roi, fold_id, acc, acc_mean, acc_sem, sbjID,
                             roi_str, comp_pair].
                dataframe containing the decoding performance for each participant

            permuteDecode_all_df: pd.DataFrame
                            [acc, sbjID, roi, roi_str, nVox, permute_i, comp_pair]
                dataframe containing the distribution of decoding permutation
                (10000 iterations)

        nVox_to_analyze: scalar
                the number of voxels used for analysis

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        decode_group_df = (
            decode_all_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc_cv_mean"})

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # comparison pairs for plotting
        comp_for_plot = [[1, 2], [3, 4], [5, 6]]  # ards, hmrds, crds

        # collect data
        data_all_rds = {}
        baseline_all_rds = {}
        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa_decode.n_ROIs):
            # collect data ards, hmrds, crds
            data = []
            baseline_list = []

            for i in range(len(comp_for_plot)):

                comp_pair = comp_for_plot[i]
                comp_pair_str = (
                    self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
                )
                # comp_pair_str = (
                #     plot_mvpa_decode.conds[comp_pair[0]]
                #     + ","
                #     + plot_mvpa_decode.conds[comp_pair[1]]
                # )

                # crossDecode_avg across all sbjID
                df = decode_group_df.loc[
                    (decode_group_df["comp_pair"] == comp_pair_str)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.nVox == nVox_to_analyze)
                ]
                if len(df.acc_cv_mean) != 0:
                    data.append(np.array(df.acc_cv_mean))

                ## add permutation distribution data to the plot
                permute_df = permute_group_df.loc[
                    (permute_group_df.roi == roi)
                    & (permute_group_df.nVox == nVox_to_analyze)
                    & (permute_group_df.comp_pair == comp_pair_str)
                ]

                ## find baseline
                rv_permute = np.float32(permute_df.acc)

                ## calculate p_val: proportions of rv_bootstrap which is less than
                # val_threshold (the right tail of rv_permute).
                # this stat test is very strict.
                # alpha_corrected = alpha/(2*len(ROIs))
                baseline = np.percentile(rv_permute, (1 - alpha) * 100)
                baseline_list.append(baseline)

            data_all_rds[self.ROIs[roi]] = data
            baseline_all_rds[self.ROIs[roi]] = baseline_list
            # data_all_rds[plot_mvpa_decode.ROIs[roi]] = data
            # baseline_all_rds[plot_mvpa_decode.ROIs[roi]] = baseline_list

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (18, 15)
        n_row = 2
        n_col = 4
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM Decoding, AVG, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)
        fig.text(0.5, -0.03, "Dot correlation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.4, hspace=0.2)

        bar_width = 2
        pos_roi = [0, 2 * bar_width, 4 * bar_width]

        # upper and lower boxplot y-axis
        yBox_low = 0.2
        yBox_up = 1.01

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
            for i in range(len(comp_for_plot)):
                y = data[i]
                jitter = np.random.normal(0, 0.05, size=len(y))
                x = pos_roi[i] + jitter
                # axes[id_row, id_col].plot(
                #     x, y, ".", color=self.color_point[i], markersize=18
                # )
                axes[id_row, id_col].plot(x, y, ".", color="gray", markersize=12)

                # plot acc threshold
                prob_thresh = baseline_list[i]
                axes[id_row, id_col].plot(
                    [pos_roi[i] - bar_width, pos_roi[i] + bar_width],
                    [prob_thresh, prob_thresh],
                    "r--",
                    linewidth=3,
                )

            axes[id_row, id_col].set_ylim(yBox_low, yBox_up)
            axes[id_row, id_col].set_yticks(np.arange(yBox_low, yBox_up, 0.1))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(yBox_low, yBox_up, 0.1), 2)
            )

            axes[id_row, id_col].set_xlim(
                pos_roi[0] - bar_width, pos_roi[-1] + bar_width
            )
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
                "../../../Plots/CMM/MVPA/PlotBox_decode_AVG_%s_v2.pdf"
                % (str(nVox_to_analyze)),
                dpi=600,
                bbox_inches="tight",
            )

    def plotBox_decode_and_dprime_at_nVox(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

        Parameters
        ----------
        decode_all_df: pd.DataFrame
                            [nVox, roi, fold_id, acc, acc_mean, acc_sem, sbjID,
                             roi_str, comp_pair].
                dataframe containing the decoding performance for each participant

            permuteDecode_all_df: pd.DataFrame
                            [acc, sbjID, roi, roi_str, nVox, permute_i, comp_pair]
                dataframe containing the distribution of decoding permutation
                (10000 iterations)

        nVox_to_analyze: scalar
                the number of voxels used for analysis

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        None.

        """

        # averate decode_all_df across fold_id for each sbjID
        decode_group_df = (
            decode_all_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(columns={"mean": "acc_cv_mean"})

        # average permutation distribution across sbjID
        permute_group_df = (
            permuteDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        # comparison pairs for plotting
        comp_for_plot = [[1, 2], [3, 4], [5, 6]]  # ards, hmrds, crds

        # collect data
        data_all_rds = {}
        dprime_all_rds = {}
        baseline_all_rds = {}
        for roi in range(self.n_ROIs):
            # for roi in range(plot_mvpa_decode.n_ROIs):
            # collect data ards, hmrds, crds
            data = []
            dprime = []
            baseline_list = []

            for i in range(len(comp_for_plot)):

                comp_pair = comp_for_plot[i]
                comp_pair_str = (
                    self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
                )
                # comp_pair_str = (
                #     plot_mvpa_decode.conds[comp_pair[0]]
                #     + ","
                #     + plot_mvpa_decode.conds[comp_pair[1]]
                # )

                # crossDecode_avg across all sbjID
                df = decode_group_df.loc[
                    (decode_group_df["comp_pair"] == comp_pair_str)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.nVox == nVox_to_analyze)
                ]
                if len(df.acc_cv_mean) != 0:
                    data.append(np.array(df.acc_cv_mean))
                    dprime.append(2 * erfinv(2 * df.acc_cv_mean - 1))

                ## add permutation distribution data to the plot
                permute_df = permute_group_df.loc[
                    (permute_group_df.roi == roi)
                    & (permute_group_df.nVox == nVox_to_analyze)
                    & (permute_group_df.comp_pair == comp_pair_str)
                ]

                ## find baseline
                rv_permute = np.float32(permute_df.acc)

                ## calculate p_val: proportions of rv_bootstrap which is less than
                # val_threshold (the right tail of rv_permute).
                # this stat test is very strict.
                # alpha_corrected = alpha/(2*len(ROIs))
                baseline = np.percentile(rv_permute, (1 - alpha) * 100)
                baseline_list.append(baseline)

            data_all_rds[self.ROIs[roi]] = data
            dprime_all_rds[self.ROIs[roi]] = dprime
            baseline_all_rds[self.ROIs[roi]] = baseline_list
            # data_all_rds[plot_mvpa_decode.ROIs[roi]] = data
            # baseline_all_rds[plot_mvpa_decode.ROIs[roi]] = baseline_list

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (21, 14)
        n_row = 2
        n_col = 4
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM Decoding, AVG, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.02, 0.5, "Prediction Accuracy", va="center", rotation=90)
        fig.text(1.04, 0.5, "Sensitivity (d')", va="center", rotation=90, c="magenta")
        fig.text(0.5, -0.03, "Dot correlation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.8, hspace=0.2)

        bar_width = 2
        pos_roi = [0, 2 * bar_width, 4 * bar_width]

        # upper and lower boxplot y-axis
        yBox_low = 0.2
        yBox_up = 0.95

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
            dprime = dprime_all_rds[self.ROIs[roi]]
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
            for i in range(len(comp_for_plot)):
                y = data[i]
                jitter = np.random.normal(0, 0.05, size=len(y))
                x = pos_roi[i] + jitter
                # axes[id_row, id_col].plot(
                #     x, y, ".", color=self.color_point[i], markersize=18
                # )
                axes[id_row, id_col].plot(x, y, ".", color="gray", markersize=12)

                # plot acc threshold
                prob_thresh = baseline_list[i]
                axes[id_row, id_col].plot(
                    [pos_roi[i] - bar_width, pos_roi[i] + bar_width],
                    [prob_thresh, prob_thresh],
                    "r--",
                    linewidth=3,
                )

                # # plot dprime
                # y = dprime[i]
                # axes2.plot(x, y, ".", color="red", markersize=12)

            # set title
            axes[id_row, id_col].set_title(self.ROIs[roi])

            axes[id_row, id_col].set_ylim(yBox_low, yBox_up)
            axes[id_row, id_col].set_yticks(np.arange(yBox_low, yBox_up, 0.1))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(yBox_low, yBox_up, 0.1), 2)
            )

            axes[id_row, id_col].set_xlim(
                pos_roi[0] - bar_width, pos_roi[-1] + bar_width
            )
            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xticks(pos_roi)
            axes[id_row, id_col].set_xticklabels([-1.0, 0.0, 1.0])

            axes2 = axes[id_row, id_col].twinx()
            axes2.set_ylim(axes[id_row, id_col].get_ylim())
            axes2.set_yticks(axes[id_row, id_col].get_yticks())
            y2_label = 2 * erfinv(2 * np.arange(yBox_low, yBox_up, 0.1) - 1)
            axes2.set_yticklabels(np.round(y2_label, 2), color="magenta")

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)
            axes2.spines["top"].set_visible(False)
            axes2.spines["left"].set_visible(False)
            axes2.spines["right"].set_color("magenta")

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")
            axes2.xaxis.set_ticks_position("bottom")
            axes2.yaxis.set_ticks_position("right")

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotBox_decode_and_dprime_AVG_%s.pdf"
                % (str(nVox_to_analyze)),
                dpi=600,
                bbox_inches="tight",
            )

    def plotBox_decode_at_nVox_MT(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average decoding accuracy in MT at nVox_to_analyze.
            This plots MT defined with_localizer vs without_localizer
            RDS: aRDS, hmrds, cRDS

        Parameters
        ----------
        decode_all_df: pd.DataFrame
                            [nVox, roi, fold_id, acc, acc_mean, acc_sem, sbjID,
                             roi_str, comp_pair].
                dataframe containing the decoding performance for each participant

        permuteDecode_all_df: pd.DataFrame
                        [acc, sbjID, roi, roi_str, nVox, permute_i, comp_pair]
            dataframe containing the distribution of decoding permutation
            (10000 iterations)

        nVox_to_analyze: scalar
                the number of voxels used for analysis

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        None.

        """
        # obtain data for MT that uses localizer only
        decode_MTloc_df = decode_all_df[
            (decode_all_df.roi_str == "MT")
            & (decode_all_df.sbjID.isin(self.sbjID_with_MTlocalizer))
        ]
        permuteDecode_MT_df = permuteDecode_all_df[
            (permuteDecode_all_df.roi_str == "MT")
        ]
        # obtain data for MT that does not use localizer
        decode_MTnoloc_df = decode_all_df[
            (decode_all_df.roi_str == "MT")
            & ~(decode_all_df.sbjID.isin(self.sbjID_with_MTlocalizer))
        ]
        # permuteDecode_MTnoloc_df = permuteDecode_all_df[
        #     (permuteDecode_all_df.roi_str == "MT")
        #     & ~(permuteDecode_all_df.sbjID.isin(self.sbjID_with_MTlocalizer))
        # ]

        # averate decode_MTloc_df across fold_id for each sbjID
        decode_MTloc_group_df = (
            decode_MTloc_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_MTloc_group_df = decode_MTloc_group_df.rename(
            columns={"mean": "acc_cv_mean"}
        )

        # averate decode_MTnoloc_group_df across fold_id for each sbjID
        decode_MTnoloc_group_df = (
            decode_MTnoloc_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_MTnoloc_group_df = decode_MTnoloc_group_df.rename(
            columns={"mean": "acc_cv_mean"}
        )

        # average permutation distribution across sbjID for MT localizer
        permute_MT_group_df = (
            permuteDecode_MT_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_MT_group_df = permute_MT_group_df.rename(columns={"mean": "acc"})

        # # average permutation distribution across sbjID for MT without localizer
        # permute_MTnoloc_group_df = (
        #     permuteDecode_MTnoloc_df.groupby(
        #         ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
        #     )
        #     .acc.agg(["mean"])
        #     .reset_index()
        # )
        # permute_MTnoloc_group_df = permute_MTnoloc_group_df.rename(
        #     columns={"mean": "acc"}
        # )

        # comparison pairs for plotting
        comp_for_plot = [[1, 2], [3, 4], [5, 6]]  # ards, hmrds, crds

        # collect data ards, hmrds, crds
        data_MTloc = []
        baseline_list_MT = []
        data_MTnoloc = []

        for i in range(len(comp_for_plot)):

            comp_pair = comp_for_plot[i]
            comp_pair_str = self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
            # comp_pair_str = (
            #     plot_mvpa_decode.conds[comp_pair[0]]
            #     + ","
            #     + plot_mvpa_decode.conds[comp_pair[1]]
            # )

            # crossDecode_avg across all sbjID
            df = decode_MTloc_group_df.loc[
                (decode_MTloc_group_df["comp_pair"] == comp_pair_str)
                & (decode_MTloc_group_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data_MTloc.append(np.array(df.acc_cv_mean))
            df = decode_MTnoloc_group_df.loc[
                (decode_MTnoloc_group_df["comp_pair"] == comp_pair_str)
                & (decode_MTnoloc_group_df.nVox == nVox_to_analyze)
            ]
            if len(df.acc_cv_mean) != 0:
                data_MTnoloc.append(np.array(df.acc_cv_mean))

            ## add permutation distribution data to the plot
            permute_df = permute_MT_group_df.loc[
                (permute_MT_group_df.nVox == nVox_to_analyze)
                & (permute_MT_group_df.comp_pair == comp_pair_str)
            ]
            ## find baseline
            rv_permute = np.float32(permute_df.acc)

            ## calculate p_val: proportions of rv_bootstrap which is less than
            # val_threshold (the right tail of rv_permute).
            # this stat test is very strict.
            # alpha_corrected = alpha/(2*len(ROIs))
            baseline = np.percentile(rv_permute, (1 - alpha) * 100)
            baseline_list_MT.append(baseline)

        ## plot
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        # figsize = (18, 15)
        figsize = (12, 9)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "SVM Decoding, AVG, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction Accuracy", va="center", rotation=90)
        fig.text(0.5, -0.03, "Dot correlation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.4, hspace=0.2)

        bar_width = 2
        pos_roi = [0, 2 * bar_width, 4 * bar_width]

        # upper and lower boxplot y-axis
        yBox_low = 0.3
        yBox_up = 1.01

        boxprops = dict(
            linewidth=3, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=3, color="black")
        meanprops = dict(
            marker="x", markersize=17, markerfacecolor="blue", markeredgecolor="blue"
        )
        whiskerprops = dict(linewidth=3)
        capprops = dict(linewidth=3)

        ## MT localizer
        axes[0].boxplot(
            data_MTloc,
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
        # MT no localizer
        axes[1].boxplot(
            data_MTnoloc,
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

        # add data point, MT localizer
        for i in range(len(comp_for_plot)):
            y = data_MTloc[i]
            jitter = np.random.normal(0, 0.05, size=len(y))
            x = pos_roi[i] + jitter
            axes[0].plot(x, y, ".", color="gray", markersize=12)

            # plot acc threshold
            prob_thresh = baseline_list_MT[i]
            axes[0].plot(
                [pos_roi[i] - bar_width, pos_roi[i] + bar_width],
                [prob_thresh, prob_thresh],
                "r--",
                linewidth=3,
            )
            # add data point, MT no localizer
            # for i in range(len(comp_for_plot)):
            y = data_MTnoloc[i]
            jitter = np.random.normal(0, 0.05, size=len(y))
            x = pos_roi[i] + jitter
            axes[1].plot(x, y, ".", color="gray", markersize=12)

            # plot acc threshold
            prob_thresh = baseline_list_MT[i]
            axes[1].plot(
                [pos_roi[i] - bar_width, pos_roi[i] + bar_width],
                [prob_thresh, prob_thresh],
                "r--",
                linewidth=3,
            )

        axes[0].set_ylim(yBox_low, yBox_up)
        axes[0].set_yticks(np.arange(yBox_low, yBox_up, 0.1))
        axes[0].set_yticklabels(np.round(np.arange(yBox_low, yBox_up, 0.1), 2))
        axes[0].set_xlim(pos_roi[0] - bar_width, pos_roi[-1] + bar_width)
        axes[0].set_xlabel("")
        axes[0].set_xticks(pos_roi)
        axes[0].set_xticklabels([-1.0, 0.0, 1.0])
        axes[1].set_ylim(yBox_low, yBox_up)
        axes[1].set_yticks(np.arange(yBox_low, yBox_up, 0.1))
        axes[1].set_yticklabels(np.round(np.arange(yBox_low, yBox_up, 0.1), 2))
        axes[1].set_xlim(pos_roi[0] - bar_width, pos_roi[-1] + bar_width)
        axes[1].set_xlabel("")
        axes[1].set_xticks(pos_roi)
        axes[1].set_xticklabels([-1.0, 0.0, 1.0])

        # set title
        axes[0].set_title("hMT+ with localizer")
        axes[1].set_title("hMT+ without localizer")

        # remove top and right frame
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes[0].xaxis.set_ticks_position("bottom")
        axes[0].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotBox_decode_AVG_%s_MT.pdf"
                % (str(nVox_to_analyze)),
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_decode_avg(
        self, decode_all_df, permuteDecode_all_df, save_flag, alpha=0.05
    ):

        nVox_all = decode_all_df.nVox.unique()

        ## average across sbjID
        # average decode_all_df across fold_id and then across sbjID
        temp_df = (
            decode_all_df.groupby(["sbjID", "roi", "nVox", "roi_str", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        temp_df = temp_df.rename(columns={"mean": "acc_cv"})

        decode_group_df = (
            temp_df.groupby(["roi", "nVox", "roi_str", "comp_pair"])
            .acc_cv.agg(["mean", sem])
            .reset_index()
        )
        decode_group_df = decode_group_df.rename(
            columns={"mean": "acc_group_mean", "sem": "acc_group_sem"}
        )

        # compute probability threshold in each ROI from permutation dataset
        permute_group_df = (
            permuteDecode_all_df.groupby(
                ["roi", "roi_str", "nVox", "permute_i", "comp_pair"]
            )
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_group_df = permute_group_df.rename(columns={"mean": "acc"})

        ## plot All group
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (18, 18)
        n_row = 3
        n_col = 3
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        # fig.text(0.5, 1.03, "SVM Decoding AVG", ha="center")
        fig.text(-0.02, 0.5, "Prediction Accuracy", va="center", rotation=90)
        fig.text(0.5, -0.03, "# voxels", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        y_low = 0.45
        y_up = 0.85
        y_step = 0.1
        x_low = 0
        x_up = nVox_all[-1] + 26
        x_step = 50

        for roi in range(self.n_ROIs):
            id_row = roi // n_row
            id_col = roi % n_col

            ## plot crossDecode_avg of ALL group
            # ards
            comp_pair_str = "n_c0,f_c0"
            decode_df = decode_group_df.loc[
                (decode_group_df.roi == roi)
                & (decode_group_df.comp_pair == comp_pair_str)
            ]
            decode_df.plot.line(
                x="nVox",
                y="acc_group_mean",
                yerr="acc_group_sem",
                capsize=4,
                elinewidth=2,
                ecolor="k",
                color=self.ards_linecolor,
                linewidth=4,
                ax=axes[id_row, id_col],
            )

            # hmrds
            comp_pair_str = "n_c50,f_c50"
            decode_df = decode_group_df.loc[
                (decode_group_df.roi == roi)
                & (decode_group_df.comp_pair == comp_pair_str)
            ]
            decode_df.plot.line(
                x="nVox",
                y="acc_group_mean",
                yerr="acc_group_sem",
                capsize=4,
                elinewidth=2,
                ecolor="k",
                color=self.hmrds_linecolor,
                linewidth=4,
                ax=axes[id_row, id_col],
            )

            # crds
            comp_pair_str = "n_c100,f_c100"
            decode_df = decode_group_df.loc[
                (decode_group_df.roi == roi)
                & (decode_group_df.comp_pair == comp_pair_str)
            ]
            decode_df.plot.line(
                x="nVox",
                y="acc_group_mean",
                yerr="acc_group_sem",
                capsize=4,
                elinewidth=2,
                ecolor="k",
                color=self.crds_linecolor,
                linewidth=4,
                ax=axes[id_row, id_col],
            )

            ## get shuffled baseline
            baseline = np.zeros(
                (len(self.comp_pair_all), len(nVox_all)), dtype=np.float32
            )
            for c in range(len(self.comp_pair_all)):
                comp_pair = self.comp_pair_all[c]
                comp_pair_str = (
                    self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
                )

                for v in range(len(nVox_all)):
                    nVox = nVox_all[v]
                    permute_df = permute_group_df.loc[
                        (permute_group_df.roi == roi)
                        & (permute_group_df.nVox == nVox)
                        & (permute_group_df.comp_pair == comp_pair_str)
                    ]

                    ## find baseline
                    rv_permute = np.float32(permute_df.acc)

                    ## calculate p_val: proportions of rv_bootstrap which is less than
                    # val_threshold (the right tail of rv_permute).
                    # this stat test is very strict..
                    # alpha_corrected = alpha/(2*len(ROIs))
                    baseline[c, v] = np.percentile(rv_permute, (1 - alpha) * 100)

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

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        ## dummy plot, only for setting up the legend
        axes[id_row, id_col + 1].plot(
            [x_low, x_up],
            [prob_thresh, prob_thresh],
            color=self.crds_linecolor,
            linewidth=3,
        )
        axes[id_row, id_col + 1].plot(
            [x_low, x_up],
            [prob_thresh, prob_thresh],
            color=self.hmrds_linecolor,
            linewidth=3,
        )
        axes[id_row, id_col + 1].plot(
            [x_low, x_up],
            [prob_thresh, prob_thresh],
            color=self.ards_linecolor,
            linewidth=3,
        )
        axes[id_row, id_col + 1].plot(
            [x_low, x_up], [prob_thresh, prob_thresh], "r--", linewidth=3
        )
        axes[id_row, id_col + 1].legend(
            ["cRDS", "hmRDS", "aRDS", "shuffled-baseline"],
            fontsize=18,
        )

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotLine_decode_AVG.pdf",
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
            "../../../Plots/CMM/MVPA/PlotScatter_xDecode_ards_vs_decode_hmrds.pdf",
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
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

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
            "../../../Plots/CMM/MVPA/PlotScatter_decode_ards_vs_decode_hmrds.pdf",
            dpi=600,
            bbox_inches="tight",
        )

    def plotScatter_decode_ards_vs_decode_hmrds_MT(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        plot decode ards vs decode hmrds for MT.


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

        # obtain data for MT that uses localizer only
        decode_MTloc_df = decode_all_df[
            (decode_all_df.roi_str == "MT")
            & (decode_all_df.sbjID.isin(self.sbjID_with_MTlocalizer))
        ]
        permuteDecode_MT_df = permuteDecode_all_df[
            (permuteDecode_all_df.roi_str == "MT")
        ]
        # obtain data for MT that does not use localizer
        decode_MTnoloc_df = decode_all_df[
            (decode_all_df.roi_str == "MT")
            & ~(decode_all_df.sbjID.isin(self.sbjID_with_MTlocalizer))
        ]

        # averate decode_MTloc_df across fold_id for each sbjID
        decode_MTloc_group_df = (
            decode_MTloc_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_MTloc_group_df = decode_MTloc_group_df.rename(
            columns={"mean": "acc_cv_mean"}
        )

        # averate decode_MTnoloc_group_df across fold_id for each sbjID
        decode_MTnoloc_group_df = (
            decode_MTnoloc_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        decode_MTnoloc_group_df = decode_MTnoloc_group_df.rename(
            columns={"mean": "acc_cv_mean"}
        )

        # average permute_decode_df across sbjID for each roi, nVox, permute_i, comp_pair
        permute_MT_group_df = (
            permuteDecode_MT_df.groupby(["roi", "nVox", "permute_i", "comp_pair"])
            .acc.agg(["mean"])
            .reset_index()
        )
        permute_MT_group_df = permute_MT_group_df.rename(columns={"mean": "acc"})

        ## find shuffled-baseline for ards
        permute_df = permute_MT_group_df.loc[
            (permute_MT_group_df.nVox == nVox_to_analyze)
            & (permute_MT_group_df.comp_pair == "n_c0,f_c0")
        ]
        rv_permute = np.float32(permute_df.acc)

        ## calculate p_val: proportions of rv_bootstrap which is less than
        # val_threshold (the right tail of rv_permute).
        # this stat test is very strict.
        # alpha_corrected = alpha/(2*len(ROIs))
        baseline_ards = np.percentile(rv_permute, (1 - alpha) * 100)

        ## find shuffled-baseline for hmrds
        permute_df = permute_MT_group_df.loc[
            (permute_MT_group_df.nVox == nVox_to_analyze)
            & (permute_MT_group_df.comp_pair == "n_c50,f_c50")
        ]
        ## find baseline
        rv_permute = np.float32(permute_df.acc)

        ## calculate p_val: proportions of rv_bootstrap which is less than
        # val_threshold (the right tail of rv_permute).
        # this stat test is very strict.
        # alpha_corrected = alpha/(2*len(ROIs))
        baseline_hmrds = np.percentile(rv_permute, (1 - alpha) * 100)

        ## plot All group
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (18, 9)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(0.5, 1.03, "SVM Decoding aRDS VS. Decoding hmRDS", ha="center")
        fig.text(-0.02, 0.5, "Prop. correct decoding aRDS", va="center", rotation=90)
        fig.text(0.5, -0.02, "Prop. correct decoding hmRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.35
        x_up = 0.81
        x_step = 0.05
        y_low = 0.35
        y_up = 0.86
        y_step = 0.05

        # MT with localizer
        ards = np.array(
            decode_MTloc_group_df.loc[
                (decode_MTloc_group_df.nVox == nVox_to_analyze)
                & (decode_MTloc_group_df.comp_pair == "n_c0,f_c0")
            ].acc_cv_mean,
            dtype=np.float32,
        )

        # get hmrds decoding performance
        hmrds = np.array(
            decode_MTloc_group_df.loc[
                (decode_MTloc_group_df.nVox == nVox_to_analyze)
                & (decode_MTloc_group_df.comp_pair == "n_c50,f_c50")
            ].acc_cv_mean,
            dtype=np.float32,
        )
        axes[0].plot(hmrds, ards, ".", color="gray", ms=30)

        # plot mean with errobar
        x = np.mean(hmrds)
        y = np.mean(ards)
        x_err = sem(hmrds)
        y_err = sem(ards)

        axes[0].errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            elinewidth=4,
            fmt="o",
            c="black",
            ms=15,
        )

        # MT without localizer
        ards = np.array(
            decode_MTnoloc_group_df.loc[
                (decode_MTnoloc_group_df.nVox == nVox_to_analyze)
                & (decode_MTnoloc_group_df.comp_pair == "n_c0,f_c0")
            ].acc_cv_mean,
            dtype=np.float32,
        )

        # get hmrds decoding performance
        hmrds = np.array(
            decode_MTnoloc_group_df.loc[
                (decode_MTnoloc_group_df.nVox == nVox_to_analyze)
                & (decode_MTnoloc_group_df.comp_pair == "n_c50,f_c50")
            ].acc_cv_mean,
            dtype=np.float32,
        )
        axes[1].plot(hmrds, ards, ".", color="gray", ms=30)

        # plot mean with errobar
        x = np.mean(hmrds)
        y = np.mean(ards)
        x_err = sem(hmrds)
        y_err = sem(ards)

        axes[1].errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            elinewidth=4,
            fmt="o",
            c="black",
            ms=15,
        )

        ## plot shuffled baseline
        axes[0].plot([x_low, x_up], [baseline_ards, baseline_ards], "r--", linewidth=4)
        axes[0].plot(
            [baseline_hmrds, baseline_hmrds], [y_low, y_up], "r--", linewidth=4
        )
        # draw diagonal line
        axes[0].plot([x_low - x_step, x_up], [x_low - x_step, x_up], "r--", linewidth=4)

        axes[1].plot([x_low, x_up], [baseline_ards, baseline_ards], "r--", linewidth=4)
        axes[1].plot(
            [baseline_hmrds, baseline_hmrds], [y_low, y_up], "r--", linewidth=4
        )
        # draw diagonal line
        axes[1].plot([x_low - x_step, x_up], [x_low - x_step, x_up], "r--", linewidth=4)

        ## linear fit ards vs hmrds
        # reg = LinearRegression().fit(hmrds.reshape(-1, 1), ards.reshape(-1, 1))

        # slope
        # slope = reg.coef_[0][0]
        # y_predict = reg.predict(hmrds.reshape(-1, 1))
        # r2 = explained_variance_score(ards, y_predict)

        # # plot linear fit
        # x = np.array([x_low, x_up])
        # y = reg.predict(x.reshape(-1, 1))
        # axes[id_row, id_col].plot(x, y, "b--", linewidth=4)

        # axes[id_row, id_col].text(
        #     0.7,
        #     0.31,
        #     "slope={}\nR2={}".format(str(np.round(slope, 2)), str(np.round(r2, 2))),
        # )

        axes[0].set_title("hMT+ with localizer")
        axes[1].set_title("hMT+ without localizer")

        axes[0].set_xlabel("")
        axes[0].set_xlim(x_low, x_up)
        axes[0].set_xticks(np.arange(x_low, x_up, x_step))
        axes[0].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))

        axes[1].set_xlabel("")
        axes[1].set_xlim(x_low, x_up)
        axes[1].set_xticks(np.arange(x_low, x_up, x_step))
        axes[1].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))

        axes[0].set_ylabel("")
        axes[0].set_ylim(y_low, y_up)
        axes[0].set_yticks(np.arange(y_low, y_up, y_step))
        axes[0].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes[1].set_ylabel("")
        axes[1].set_ylim(y_low, y_up)
        axes[1].set_yticks(np.arange(y_low, y_up, y_step))
        axes[1].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove the top and right frames
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes[0].xaxis.set_ticks_position("bottom")
        axes[0].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")

        if save_flag:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotScatter_decode_ards_vs_decode_hmrds_MT.pdf",
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
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=6, palette="deep")

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
                f"{self.plot_dir}/PlotScatter_decode_ards_vs_decode_hmrds_allROIs.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotScatter_decode_ards_vs_decode_hmrds_dprime(
        self,
        decode_all_df,
        nVox_to_analyze,
        save_flag,
    ):
        """
         after converted into dprime

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

        ## plot All group
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (22, 12)
        n_row = 2
        n_col = 4
        fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig1.text(
            0.5,
            1.03,
            "d_prime aRDS VS. d_prime hmRDS nVox=%s" % str(nVox_to_analyze),
            ha="center",
        )
        fig1.text(-0.02, 0.5, "d_prime aRDS", va="center", rotation=90)
        fig1.text(0.5, -0.02, "d_prime hmRDS", ha="center")
        fig1.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = -0.5
        x_up = 1.51
        x_step = 0.5
        y_low = -1.0
        y_up = 1.51
        y_step = 0.5

        # local linear fit variables, each roi
        m_all = np.empty(len(self.ROIs), dtype=np.float32)  # slope
        r2_all = np.empty(len(self.ROIs), dtype=np.float32)  # r2
        p_all = np.empty(len(self.ROIs), dtype=np.float32)  # p val
        for roi in range(len(self.ROIs)):
            # for roi in range(len(plot_mvpa_decode.ROIs)):
            id_row = roi // n_col
            id_col = roi % n_col

            ards = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c0,f_c0")
                ].acc,
                dtype=np.float32,
            )
            dprime_ards = 2 * erfinv(2 * ards - 1)

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c50,f_c50")
                ].acc,
                dtype=np.float32,
            )
            dprime_hmrds = 2 * erfinv(2 * hmrds - 1)

            x = dprime_hmrds
            y = dprime_ards

            # local linear fit on mean
            model = sm.OLS(y, sm.add_constant(x))
            results = model.fit()
            b, m = results.params
            p_val = results.pvalues[1]  # p-val for slope
            m_all[roi] = m
            r2_all[roi] = results.rsquared
            p_all[roi] = p_val

            axes[id_row, id_col].plot(
                x,
                y,
                ".",
                color="gray",
                ms=30,
            )

            # plot the local linear fit on median
            x_fit = np.linspace(x_low - 0.2, x_up)
            y_fit = b + m * x_fit
            axes[id_row, id_col].plot(x_fit, y_fit, "r--", linewidth=3)

            axes[id_row, id_col].set_title(self.ROI_plotname[roi])
            axes[id_row, id_col].set_aspect("equal")

            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xlim(x_low - 0.2, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_xticklabels(
                np.round(np.arange(x_low, x_up, x_step), 2)
            )

            axes[id_row, id_col].set_ylabel("")
            axes[id_row, id_col].set_ylim(y_low - 0.2, y_up)
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

        ########
        # plot bar the slope and r2
        ########
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (3, 4)
        n_row = 1
        n_col = 1
        fig2, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)
        fig2.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        y_low = 0
        y_up = 9
        y_step = 1
        x_low = -0.25
        x_up = 0.76
        x_step = 0.25
        y = np.arange(1, len(self.ROIs) + 1, 1)
        # axes.barh(y, m_all, fill=False, edgecolor="black", linewidth=3, align="center")
        (
            markerline,
            stemline,
            baseline,
        ) = axes.stem(
            y,
            m_all,
            orientation="horizontal",
            linefmt="black",
            markerfmt="D",
            basefmt="k",
        )

        plt.setp(stemline, linewidth=3)
        plt.setp(markerline, markersize=7)
        plt.setp(baseline, linewidth=3)

        axes.set_xlabel("Slope")
        axes.set_xlim(x_low, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2), rotation=45)

        axes.set_ylabel("")
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(y)
        axes.set_yticklabels(self.ROI_plotname)

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # plot bar r2
        fig3, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)
        fig3.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        y_low = 0
        y_up = 9
        y_step = 1
        x_low = 0.0
        x_up = 0.76
        x_step = 0.25
        # axes.barh(y, r2_all, fill=False, edgecolor="black", linewidth=3, align="center")
        (
            markerline,
            stemline,
            baseline,
        ) = axes.stem(
            y,
            r2_all,
            orientation="horizontal",
            linefmt="black",
            markerfmt="D",
            basefmt="k",
        )

        plt.setp(stemline, linewidth=3)
        plt.setp(markerline, markersize=7)
        plt.setp(baseline, linewidth=3)

        axes.set_ylabel("")
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(y)
        axes.set_yticklabels(self.ROI_plotname)

        axes.set_xlabel("R2")
        axes.set_xlim(x_low, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2), rotation=45)

        # remove the top and right frames
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        ## create pandas dataframe to save into excel
        temp = {
            "ROI": self.ROIs,
            "slope": m_all,
            "R2": r2_all,
            "p_val": p_all,
        }
        dprime_df = pd.DataFrame(temp)

        if save_flag == 1:
            fig1.savefig(
                f"{self.plot_dir}/PlotScatter_decode_ards_vs_decode_hmrds_dprime.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            fig2.savefig(
                f"{self.plot_dir}/PlotStem_decode_ards_vs_decode_hmrds_dprime_slope.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            fig3.savefig(
                f"{self.plot_dir}/PlotStem_decode_ards_vs_decode_hmrds_dprime_r2.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            # save statistic to csv
            dprime_df.to_csv(
                f"{self.stat_dir}/stat_decode_ards_vs_decode_hmrds_dprime.csv",
                index=False,
            )

    def plotScatter_decode_ards_vs_decode_hmrds_dprime_all_roi(
        self,
        decode_all_df,
        nVox_to_analyze,
        save_flag,
    ):
        """
         after converted into dprime

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

        ## plot All group
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 8)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(
            0.5,
            1.03,
            "d_prime aRDS VS. d_prime hmRDS nVox=%s" % str(nVox_to_analyze),
            ha="center",
        )
        fig.text(-0.04, 0.5, "d_prime aRDS", va="center", rotation=90)
        fig.text(0.5, -0.02, "d_prime hmRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.0
        x_up = 0.71
        x_step = 0.1
        y_low = 0.0
        y_up = 0.51
        y_step = 0.1

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        dprime_ards_all = np.empty((len(self.ROIs), self.n_sbjID), dtype=np.float32)
        dprime_hmrds_all = np.empty((len(self.ROIs), self.n_sbjID), dtype=np.float32)
        for roi in range(len(self.ROIs)):
            # for roi in range(len(plot_mvpa_decode.ROIs)):

            ards = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c0,f_c0")
                ].acc,
                dtype=np.float32,
            )
            dprime_ards = 2 * erfinv(2 * ards - 1)
            dprime_ards_all[roi] = dprime_ards

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c50,f_c50")
                ].acc,
                dtype=np.float32,
            )
            dprime_hmrds = 2 * erfinv(2 * hmrds - 1)
            dprime_hmrds_all[roi] = dprime_hmrds

            x = np.mean(dprime_hmrds)
            y = np.mean(dprime_ards)
            x_err = sem(dprime_hmrds)
            y_err = sem(dprime_ards)

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

        # global linear fit on mean
        y = dprime_ards_all.flatten()
        x = dprime_hmrds_all.flatten()

        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()
        b, m = results.params
        p_val = results.pvalues[1]  # p-val for slope

        # plot the global linear fit on median
        x_fit = np.linspace(x_low, x_up)
        y_fit = b + m * x_fit
        axes.plot(x_fit, y_fit, "r--", linewidth=3)

        # print linear fit
        fig.text(0.0, -0.1, "mean fit", ha="left", color="r")
        fig.text(0.25, -0.1, f"y = {m:.2f}x{b:.2f}", ha="left", color="r")
        fig.text(0.65, -0.1, f"R2: {results.rsquared:.3f}", ha="left", color="r")
        fig.text(0.9, -0.1, f"p_val: {p_val:.3f}", ha="left", color="r")

        axes.set_aspect("equal")

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

        if save_flag == 1:
            fig.savefig(
                f"{self.plot_dir}/PlotScatter_decode_ards_vs_decode_hmrds_dprime_allROIs.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotScatter_decode_ards_vs_decode_crds_dprime_all_roi(
        self,
        decode_all_df,
        nVox_to_analyze,
        save_flag,
    ):
        """
        plot decode ards vs decode crds after converted into dprime for all roi in one plot

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

        ## plot All group
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 8)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(
            0.5,
            1.03,
            "d_prime aRDS VS. d_prime cRDS nVox=%s" % str(nVox_to_analyze),
            ha="center",
        )
        fig.text(-0.04, 0.5, "d_prime aRDS", va="center", rotation=90)
        fig.text(0.5, -0.02, "d_prime cRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.4
        x_up = 1.1
        x_step = 0.1
        y_low = 0.0
        y_up = 0.51
        y_step = 0.1

        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]

        dprime_ratio = np.empty(len(self.ROIs), dtype=np.float32)
        for roi in range(len(self.ROIs)):
            # for roi in range(len(plot_mvpa_decode.ROIs)):

            ards = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c0,f_c0")
                ].acc,
                dtype=np.float32,
            )
            dprime_ards = 2 * erfinv(2 * ards - 1)

            # get hmrds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c100,f_c100")
                ].acc,
                dtype=np.float32,
            )
            dprime_hmrds = 2 * erfinv(2 * hmrds - 1)

            x = np.mean(dprime_hmrds)
            y = np.mean(dprime_ards)
            x_err = sem(dprime_hmrds)
            y_err = sem(dprime_ards)
            dprime_ratio[roi] = y / (x + y)

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
        # axes.plot([x_low, x_up], [x_low, x_up], "r--", linewidth=4)

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

        print(dprime_ratio)

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotScatter_decode_ards_vs_decode_crds_dprime_allROIs.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotScatter_decode_ards_vs_decode_crds_all_roi(
        self,
        decode_all_df,
        permuteDecode_all_df,
        nVox_to_analyze,
        save_flag,
        alpha=0.05,
    ):
        """
        plot decode ards vs decode crds for all roi in one plot

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

            ## find shuffled-baseline for crds
            permute_df = permute_group_df.loc[
                (permute_group_df.roi == roi)
                & (permute_group_df.nVox == nVox_to_analyze)
                & (permute_group_df.comp_pair == "n_c100,f_c100")
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
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=6, palette="deep")

        figsize = (15, 15)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)

        fig.text(
            0.5,
            1.03,
            "SVM Decoding aRDS VS. Decoding cRDS nVox=%s" % str(nVox_to_analyze),
            ha="center",
        )
        fig.text(-0.04, 0.5, "Pred. accuracy decoding aRDS", va="center", rotation=90)
        fig.text(0.5, -0.02, "Pred. accuracy decoding cRDS", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        x_low = 0.6
        x_up = 0.81
        x_step = 0.05
        y_low = 0.5
        y_up = 0.66
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

            # get crds decoding performance
            hmrds = np.array(
                decode_group_df.loc[
                    (decode_group_df.nVox == nVox_to_analyze)
                    & (decode_group_df.roi == roi)
                    & (decode_group_df.comp_pair == "n_c100,f_c100")
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
        axes.plot([x_low - 0.01, x_up], [x_low - 0.01, x_up], "r--", linewidth=4)
        # draw shuffled-chance level
        y_chance = np.max(baseline_list_ards)
        axes.plot([x_low - 0.01, x_up], [y_chance, y_chance], "r--", linewidth=4)
        x_chance = np.max(baseline_list_hmrds)
        axes.plot([x_chance, x_chance], [y_low - 0.01, y_up], "r--", linewidth=4)

        axes.set_xlabel("")
        axes.set_xlim(x_low - 0.01, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))

        axes.set_ylabel("")
        axes.set_ylim(y_low - 0.01, y_up)
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
                "../../../Plots/CMM/MVPA/PlotScatter_decode_ards_vs_decode_crds_allROIs.pdf",
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
            "../../../Plots/CMM/MVPA/PlotScatter_xDecode_ards_vs_decode_hmrds_allROIs.pdf",
            dpi=600,
            bbox_inches="tight",
        )
