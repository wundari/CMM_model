"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/MVPA/PlotMVPA_Decode.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/
Created Date: 2025-06-16
Author: Bayu G. Wundari
-----
Last Modified: 2025-06-16
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
                "../../../Plots/CMM/MVPA/PlotBox_decode_AVG_%s.pdf"
                % (str(nVox_to_analyze)),
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_decode_avg(
        self, decode_all_df, permuteDecode_all_df, save_flag, alpha=0.05
    ):

        nVox_all = decode_all_df.nVox.unique()
        nVox_all_permute = permuteDecode_all_df.nVox.unique()

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
        x_step = 200

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
                (len(self.comp_pair_all), len(nVox_all_permute)), dtype=np.float32
            )
            for c in range(len(self.comp_pair_all)):
                comp_pair = self.comp_pair_all[c]
                comp_pair_str = (
                    self.conds[comp_pair[0]] + "," + self.conds[comp_pair[1]]
                )

                for v in range(len(nVox_all_permute)):
                    nVox = nVox_all_permute[v]
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

    def plotBox_decode_at_voxPercent(
        self,
        decode_score_allSbj,
        decode_permute_allSbj_df,
        nVox_percentage_list,
        voxPercent_idx,
        save_flag,
        alpha=0.05,
    ):
        """
        box plot the average decoding accuracy at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

        Parameters
        ----------
        decode_score_allSbj : [n_sbjID,
                    n_ROIs,
                    len(comp_pair_all),
                    len(nVox_percentage_list)]
            roi order = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "hMT+"]

            dataframe containing the decoding performance for each participant
            using fixed percentage of voxels

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

        voxPercent_to_analyze = nVox_percentage_list[voxPercent_idx]

        # average permutation distribution across sbjID
        permute_group_df = (
            decode_permute_allSbj_df.groupby(
                ["roi", "roi_str", "voxPercent", "permute_i", "comp_pair"]
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
                df = decode_score_allSbj[:, roi, i, voxPercent_idx]
                data.append(df)

                ## add permutation distribution data to the plot
                permute_df = permute_group_df.loc[
                    (permute_group_df.roi == roi)
                    & (permute_group_df.voxPercent == voxPercent_to_analyze)
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
            "SVM Decoding, AVG, voxPercent=%s" % (str(voxPercent_to_analyze)),
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
        roi_title = ["V1", "V2", "V3", "V3A", "V3B", "hV4", "V7", "hMT+"]
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
            prob_thresh = np.mean(baseline_list)
            axes[id_row, id_col].plot(
                [pos_roi[0] - bar_width, pos_roi[-1] + bar_width],
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
            axes[id_row, id_col].set_title(roi_title[roi])

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                "../../../Plots/CMM/MVPA/PlotBox_decode_voxPercent_AVG_%s.pdf"
                % (str(voxPercent_to_analyze)),
                dpi=600,
                bbox_inches="tight",
            )