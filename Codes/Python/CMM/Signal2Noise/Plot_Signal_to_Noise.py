# %%
"""
working dir: 
    Codes/Python/CMM

"""
# %%
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pingouin as pg
import os
from Common.Common import PlotGeneral


# %%
class Plot_Signal2Noise(PlotGeneral):

    def __init__(self):
        super().__init__()

        self.plot_dir = os.getcwd()[:-16] + "Plots/CMM/S2N"
        self.stat_dir = os.getcwd()[:-16] + "Data/S2N"
        self.dpi = 600

    def plotBar_s2n_fmri(self, s2n_all_sbj, nVox_to_analyze, save_flag):
        """
        plot bar signal to noise

        Parameters
        ----------
        s2n_all_sbj: [n_sbjID, n_ROIs, n_rds] np.array

                        [n_sbjID, n_ROIs, 0]: s2n ards
                        [n_sbjID, n_ROIs, 1]: s2n hmrds
                        [n_sbjID, n_ROIs, 2]: s2n crds

        nVox_to_analyze : int
            the number of voxels used for analysis.
            ex: 250

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        Returns
        -------
        None.

        """

        # average across participants
        s2n_avg = np.mean(s2n_all_sbj, axis=0)

        s2n_avg_ards = s2n_avg[:, 0]
        s2n_avg_hmrds = s2n_avg[:, 1]
        s2n_avg_crds = s2n_avg[:, 2]

        # standard error across participants
        s2n_sem = sem(s2n_all_sbj, axis=0)

        s2n_sem_ards = s2n_sem[:, 0]
        s2n_sem_hmrds = s2n_sem[:, 1]
        s2n_sem_crds = s2n_sem[:, 2]

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

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
        roi_dorsal = ["V3A", "V3B", "V7", "hMT+"]
        roi_ventral = ["hV4", "", "", ""]

        bar_width = 6
        step = 24
        pos_init = step * 8
        pos_ards = np.arange(0, pos_init, step)
        d = bar_width
        pos_hmrds = np.arange(d, pos_init + d, step)
        d = 2 * bar_width
        pos_crds = np.arange(d, pos_init + d, step)

        color_bar = ["green", "orange", "magenta"]  # crds, hmrds, ards

        x_low = -6
        x_up = 90
        y_low = 0.0
        y_up = 2.1
        y_step = 0.5

        ## early cortex
        # plot ards
        temp = s2n_avg_ards[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_ards[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_ards[: len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            capsize=3,
        )

        # plot hmrds
        temp = s2n_avg_hmrds[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_hmrds[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_hmrds[: len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            capsize=3,
        )

        # plot crds
        temp = s2n_avg_crds[id_early]
        temp[-1] = 0  # dummy element
        temp_sem = s2n_sem_crds[id_early]
        temp_sem[-1] = 0  # dummy element
        axes[0].bar(
            pos_crds[: len(id_early)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
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
            pos_crds[: len(id_early)]
            + pos_hmrds[: len(id_early)]
            + pos_ards[: len(id_early)]
        ) / 3
        axes[0].set_xticks(pos_roi)
        axes[0].set_xticklabels(roi_early)

        ## dorsal cortex
        # plot ards
        axes[1].bar(
            pos_ards[: len(id_dorsal)],
            s2n_avg_ards[id_dorsal],
            yerr=s2n_sem_ards[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            capsize=3,
        )

        # plot hmrds
        axes[1].bar(
            pos_hmrds[: len(id_dorsal)],
            s2n_avg_hmrds[id_dorsal],
            yerr=s2n_sem_hmrds[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            capsize=3,
        )

        # plot crds
        axes[1].bar(
            pos_crds[: len(id_dorsal)],
            s2n_avg_crds[id_dorsal],
            yerr=s2n_sem_crds[id_dorsal],
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
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
            pos_crds[: len(id_dorsal)]
            + pos_hmrds[: len(id_dorsal)]
            + pos_ards[: len(id_dorsal)]
        ) / 3
        axes[1].set_xticks(pos_roi)
        axes[1].set_xticklabels(roi_dorsal)

        axes[1].set_title("Dorsal areas")

        ## ventral cortex
        # plot ards
        temp = s2n_avg_ards[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_ards[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_ards[0 : len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[2],
            capsize=3,
        )

        # plot hmrds_crossed
        temp = s2n_avg_hmrds[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_hmrds[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_hmrds[: len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[1],
            capsize=3,
        )
        # plot crds
        temp = s2n_avg_crds[id_ventral]
        temp[1:] = 0  # dummy element
        temp_sem = s2n_sem_crds[id_ventral]
        temp_sem[1:] = 0  # dummy element
        axes[2].bar(
            pos_crds[: len(id_ventral)],
            temp,
            yerr=temp_sem,
            width=bar_width,
            linewidth=2,
            color=color_bar[0],
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
            pos_ards[: len(id_ventral)]
            + pos_hmrds[: len(id_ventral)]
            + pos_crds[: len(id_ventral)]
        ) / 3
        axes[2].set_xticks(pos_roi)
        axes[2].set_xticklabels(roi_ventral)

        # plt.ylabel("GLM beta", labelpad=15)

        axes[2].legend(
            [
                "aRDS",
                "hmRDS",
                "cRDS",
            ],
            fontsize=12,
            bbox_to_anchor=(0.5, 0.1),
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

    def plotBox_s2n_fmri_at_nVox(
        self,
        s2n_all_sbj,
        nVox_to_analyze,
        save_flag,
    ):
        """
        box plot the average signal to noise ratio at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

        Parameters
        ----------
        s2n_all_sbj: [n_sbjID, n_ROIs, n_rds] np.array

                        [n_sbjID, n_ROIs, 0]: s2n ards
                        [n_sbjID, n_ROIs, 1]: s2n hmrds
                        [n_sbjID, n_ROIs, 2]: s2n crds

        nVox_to_analyze: int
                the number of voxels used for analysis

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        None.

        """

        # create pandas dataframe for two-way anova analysis
        n_rds = 3  # ards, hmrds, crds
        n_rows = self.n_sbjID * self.n_ROIs * n_rds
        df_array = np.zeros((n_rows, 4), dtype=np.float32)  # [roi, rds, sbj, s2n]
        for roi in range(len(self.ROIs)):
            for rds in range(n_rds):
                id_start = (roi * n_rds * self.n_sbjID) + (rds * self.n_sbjID)
                id_end = id_start + self.n_sbjID

                df_array[id_start:id_end, 0] = roi  # roi
                df_array[id_start:id_end, 1] = rds  # rds
                df_array[id_start:id_end, 2] = np.arange(self.n_sbjID)  # sbjID
                df_array[id_start:id_end, 3] = s2n_all_sbj[:, roi, rds]  # s2n

        # create pandas df
        df = pd.DataFrame(df_array, columns=["roi", "rds", "sbj", "s2n"])

        # # perform two-way ANOVA with pingouin
        aov2 = pg.anova(data=df, dv="s2n", between=["roi", "rds"])

        # post-hoc test with Mann-Whitney U stat
        # within: roi, between: rds
        pt = pg.pairwise_tests(
            data=df,
            dv="s2n",
            within="roi",
            between="rds",
            subject="sbj",
            parametric=False,
        )
        # within: rds, between: roi
        pt2 = pg.pairwise_tests(
            data=df,
            dv="s2n",
            within="rds",
            between="roi",
            subject="sbj",
            parametric=False,
        )

        # # perform two-way ANOVA with statsmodel
        # model = ols("s2n ~ C(roi) + C(rds) + C(roi):C(rds)", data=df).fit()
        # sm.stats.anova_lm(model, typ=2)

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
            "Signal to noise ratio, nVox=%s" % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(
            -0.03,
            0.5,
            "Signal-to-noise ratio (task vs. fixation)",
            va="center",
            rotation=90,
        )
        fig.text(0.5, -0.03, "Dot correlation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.4, hspace=0.2)

        bar_width = 2
        pos_roi = [0, 2 * bar_width, 4 * bar_width]

        # upper and lower boxplot y-axis
        yBox_low = 0.0
        yBox_up = 2.8
        y_step = 0.5

        boxprops = dict(
            linewidth=3, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=3, color="black")
        meanprops = dict(
            marker="x", markersize=17, markerfacecolor="blue", markeredgecolor="blue"
        )
        whiskerprops = dict(linewidth=3)
        capprops = dict(linewidth=3)

        for roi in range(len(self.ROI_plotname)):
            id_row = roi // n_col
            id_col = roi % n_col

            data = s2n_all_sbj[:, roi]

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
            for i in range(3):  # iterate over rds types: ards, hmrds, crds
                y = data[:, i]
                jitter = np.random.normal(0, 0.05, size=len(y))
                x = pos_roi[i] + jitter
                axes[id_row, id_col].plot(x, y, ".", color="gray", markersize=12)

            axes[id_row, id_col].set_ylim(yBox_low, yBox_up)
            axes[id_row, id_col].set_yticks(np.arange(yBox_low, yBox_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(yBox_low, yBox_up, y_step), 2)
            )

            axes[id_row, id_col].set_xlim(
                pos_roi[0] - bar_width, pos_roi[-1] + bar_width
            )
            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xticks(pos_roi)
            axes[id_row, id_col].set_xticklabels([-1.0, 0.0, 1.0])

            # set title
            axes[id_row, id_col].set_title(self.ROI_plotname[roi])

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                f"{self.plot_dir}/PlotBox_s2n_{nVox_to_analyze}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            # save 2two-way anova to csv
            aov2.to_csv(f"{self.stat_dir}/s2n_fmri_anova2way.csv", index=False)

            # save Mann-Whitney U stat to csv
            pt.to_csv(
                f"{self.stat_dir}/s2n_fmri_pairwise_tests_within_roi_between_rds.csv",
                index=False,
            )
            pt2.to_csv(
                f"{self.stat_dir}/s2n_fmri_pairwise_tests_within_rds_between_roi.csv",
                index=False,
            )

    def plotBox_signalchange_fmri_at_nVox(
        self,
        signalchange_all_sbj,
        nVox_to_analyze,
        save_flag,
    ):
        """
        box plot the average signal signal change at nVox_to_analyze
            for all ROIs, aRDS, hmrds, cRDS

        Parameters
        ----------
        signalchange_all_sbj: [n_sbjID, n_ROIs, n_rds] np.array

                        [n_sbjID, n_ROIs, 0]: s2n ards
                        [n_sbjID, n_ROIs, 1]: s2n hmrds
                        [n_sbjID, n_ROIs, 2]: s2n crds

        nVox_to_analyze: int
                the number of voxels used for analysis

        save_flag: scalar
            whether to save the resulted plot or not (0: no; 1: yes)

        alpha : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        None.

        """

        # create pandas dataframe for two-way anova analysis
        n_rds = 3  # ards, hmrds, crds
        n_rows = self.n_sbjID * self.n_ROIs * n_rds
        df_array = np.zeros((n_rows, 4), dtype=np.float32)  # [roi, rds, sbj, s2n]
        for roi in range(len(self.ROIs)):
            for rds in range(n_rds):
                id_start = (roi * n_rds * self.n_sbjID) + (rds * self.n_sbjID)
                id_end = id_start + self.n_sbjID

                df_array[id_start:id_end, 0] = roi  # roi
                df_array[id_start:id_end, 1] = rds  # rds
                df_array[id_start:id_end, 2] = np.arange(self.n_sbjID)  # sbjID
                df_array[id_start:id_end, 3] = signalchange_all_sbj[:, roi, rds]  # s2n

        # create pandas df
        df = pd.DataFrame(df_array, columns=["roi", "rds", "sbj", "s2n"])

        # # perform two-way ANOVA with pingouin
        aov2 = pg.anova(data=df, dv="s2n", between=["roi", "rds"])

        # post-hoc test with Mann-Whitney U stat
        # within: roi, between: rds
        pt = pg.pairwise_tests(
            data=df,
            dv="s2n",
            within="roi",
            between="rds",
            subject="sbj",
            parametric=False,
        )
        # within: rds, between: roi
        pt2 = pg.pairwise_tests(
            data=df,
            dv="s2n",
            within="rds",
            between="roi",
            subject="sbj",
            parametric=False,
        )

        # # perform two-way ANOVA with statsmodel
        # model = ols("s2n ~ C(roi) + C(rds) + C(roi):C(rds)", data=df).fit()
        # sm.stats.anova_lm(model, typ=2)

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
            "Percent signal change from fixation baseline, nVox=%s"
            % (str(nVox_to_analyze)),
            ha="center",
        )
        fig.text(
            -0.03,
            0.5,
            "Percent signal change",
            va="center",
            rotation=90,
        )
        fig.text(0.5, -0.03, "Dot correlation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.4, hspace=0.2)

        bar_width = 2
        pos_roi = [0, 2 * bar_width, 4 * bar_width]

        # upper and lower boxplot y-axis
        yBox_low = 0.0
        yBox_up = 2.2
        y_step = 0.5

        boxprops = dict(
            linewidth=3, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=3, color="black")
        meanprops = dict(
            marker="x", markersize=17, markerfacecolor="blue", markeredgecolor="blue"
        )
        whiskerprops = dict(linewidth=3)
        capprops = dict(linewidth=3)

        for roi in range(len(self.ROI_plotname)):
            id_row = roi // n_col
            id_col = roi % n_col

            data = signalchange_all_sbj[:, roi]

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
            for i in range(3):  # iterate over rds types: ards, hmrds, crds
                y = data[:, i]
                jitter = np.random.normal(0, 0.05, size=len(y))
                x = pos_roi[i] + jitter
                axes[id_row, id_col].plot(x, y, ".", color="gray", markersize=12)

            axes[id_row, id_col].set_ylim(yBox_low, yBox_up)
            axes[id_row, id_col].set_yticks(np.arange(yBox_low, yBox_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(yBox_low, yBox_up, y_step), 2)
            )

            axes[id_row, id_col].set_xlim(
                pos_roi[0] - bar_width, pos_roi[-1] + bar_width
            )
            axes[id_row, id_col].set_xlabel("")
            axes[id_row, id_col].set_xticks(pos_roi)
            axes[id_row, id_col].set_xticklabels([-1.0, 0.0, 1.0])

            # set title
            axes[id_row, id_col].set_title(self.ROI_plotname[roi])

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        if save_flag == 1:
            fig.savefig(
                f"{self.plot_dir}/PlotBox_signalchange_{nVox_to_analyze}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            # save 2two-way anova to csv
            aov2.to_csv(f"{self.stat_dir}/signalchange_fmri_anova2way.csv", index=False)

            # save Mann-Whitney U stat to csv
            pt.to_csv(
                f"{self.stat_dir}/signalchange_fmri_pairwise_tests_within_roi_between_rds.csv",
                index=False,
            )
            pt2.to_csv(
                f"{self.stat_dir}/signalchange_fmri_pairwise_tests_within_rds_between_roi.csv",
                index=False,
            )
