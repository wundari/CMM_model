# %%
"""
working dir: 
    /NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM

"""
# %%
import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from timeit import default_timer as timer

from Common.Common import General


# %%
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
                             voxResp_crds_crossed, voxResp_crds_uncrossed]
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
        n_rds = 3  # rds_types : ards, hmrds, crds

        # load vtc
        vtc = self.load_vtc(sbj)

        # load stimulus timing parameters
        vtc_stimID = sio.loadmat(
            "../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        )["paramIdFull"]

        # label vtc
        vtc_labeled = self.label_vtc(vtc, vtc_stimID)

        # label rds_type
        vtc_labeled["rds"] = "na"  # fix
        vtc_labeled["rds_id"] = 0  # fix
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c0") | (vtc_labeled.cond == "f_c0")), "rds"
        ] = "ards"
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c0") | (vtc_labeled.cond == "f_c0")), "rds_id"
        ] = 1
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c50") | (vtc_labeled.cond == "f_c50")), "rds"
        ] = "hmrds"
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c50") | (vtc_labeled.cond == "f_c50")), "rds_id"
        ] = 2
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c100") | (vtc_labeled.cond == "f_c100")), "rds"
        ] = "crds"
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c100") | (vtc_labeled.cond == "f_c100")), "rds_id"
        ] = 3

        s2n_all_roi = np.zeros((self.n_ROIs, nRuns, n_rds), dtype=np.float32)

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
            # average across all conditions in each run
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

                # group by [vox, rds_id, rds], average across timepoints
                vtc_group = (
                    vtc_run.groupby(["vox", "rds_id", "rds"])
                    .vtc_norm.agg(["mean"])
                    .reset_index()
                )
                # vtc_group = (
                #     vtc_run.groupby(["vox", "rds_id", "rds"])
                #     .vtc_value.agg(["mean"])
                #     .reset_index()
                # )
                vtc_group = vtc_group.rename(
                    columns={"mean": "vtc_avg", "var": "vtc_var"}
                )

                # transform vtc_group.vtc_avg into matrix [n_rds + 1, nVox]
                y_avg = np.array(
                    vtc_group.pivot_table(
                        index="rds_id", columns="vox", values="vtc_avg"
                    ),
                    dtype=np.float32,
                )

                # sort y_avg in descendeding order based on t_test
                y_sort = y_avg[:, id_sort]  # [n_rds + 1, nVox], including fixation

                # select nVox_to_analyze voxels
                y_sel = y_sort[
                    :, :nVox_to_analyze
                ]  # [n_rds + 1, nVox], including fixation

                # get fixation responses
                # y_fix = np.tile(
                #     y_sel[0], (n_rds, 1)
                # )  # [n_rds, nVox], excluding fixation
                y_fix = y_sel[0]

                # compute the response difference between stimulus and fixation
                y_diff = y_sel[1:] - y_fix  # [n_rds, nVox], excluding fixation

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
                y_std = np.tile(temp[:nVox_to_analyze], (n_rds, 1))

                # compute s2n with respect to fixation for each voxel
                s2n_val = y_diff / y_std  # [n_rds, nVox], excluding fixation

                # average across these voxels
                s2n_all_roi[roi, run] = np.mean(
                    s2n_val, axis=1
                )  # [nROIs, nRuns, n_rds], exclude fixation

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
        n_rds = 3  # rds_types: ards, hmrds, crds
        s2n_all_sbj = np.zeros((self.n_sbjID, self.n_ROIs, n_rds), dtype=np.float32)
        for sbj in range(self.n_sbjID):
            s2n = s2n_list[0][sbj]  # [nROIs, nRuns, n_rds]

            # average across run
            s2n_avg = np.mean(s2n, axis=1)  # [nROIs, n_rds]

            s2n_all_sbj[sbj] = s2n_avg  # [n_sbjID, nROIs, n_rds]

        return s2n_all_sbj

    def _signalchange_fmri_sbj(self, t_stat_all_sbjID, nVox_to_analyze, sbj):
        """
        compute percent signal change for a single participant

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
        n_rds = 3  # rds_types : ards, hmrds, crds

        # load vtc
        vtc = self.load_vtc(sbj)

        # load stimulus timing parameters
        vtc_stimID = sio.loadmat(
            "../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
        )["paramIdFull"]

        # label vtc
        vtc_labeled = self.label_vtc(vtc, vtc_stimID)

        # label rds_type
        vtc_labeled["rds"] = "na"  # fix
        vtc_labeled["rds_id"] = 0  # fix
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c0") | (vtc_labeled.cond == "f_c0")), "rds"
        ] = "ards"
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c0") | (vtc_labeled.cond == "f_c0")), "rds_id"
        ] = 1
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c50") | (vtc_labeled.cond == "f_c50")), "rds"
        ] = "hmrds"
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c50") | (vtc_labeled.cond == "f_c50")), "rds_id"
        ] = 2
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c100") | (vtc_labeled.cond == "f_c100")), "rds"
        ] = "crds"
        vtc_labeled.loc[
            ((vtc_labeled.cond == "n_c100") | (vtc_labeled.cond == "f_c100")), "rds_id"
        ] = 3

        signalchange_all_roi = np.zeros((self.n_ROIs, nRuns, n_rds), dtype=np.float32)

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
            # average across all conditions in each run
            # avg = vtc_roi.groupby(["roi", "run", "vox"])["vtc_value"].transform("mean")
            # # normalize
            # temp = (vtc_roi.vtc_value - avg) / avg * 100
            # vtc_roi = vtc_roi.assign(vtc_norm=temp)

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

                # group by [vox, rds_id, rds], average across timepoints
                vtc_group = (
                    vtc_run.groupby(["vox", "rds_id", "rds"])
                    .vtc_value.agg(["mean"])
                    .reset_index()
                )
                # vtc_group = (
                #     vtc_run.groupby(["vox", "rds_id", "rds"])
                #     .vtc_value.agg(["mean"])
                #     .reset_index()
                # )
                vtc_group = vtc_group.rename(columns={"mean": "vtc_avg"})

                # transform vtc_group.vtc_avg into matrix [n_rds + 1, nVox]
                y_avg = np.array(
                    vtc_group.pivot_table(
                        index="rds_id", columns="vox", values="vtc_avg"
                    ),
                    dtype=np.float32,
                )

                # sort y_avg in descendeding order based on t_test
                y_sort = y_avg[:, id_sort]  # [n_rds + 1, nVox], including fixation

                # select nVox_to_analyze voxels
                y_sel = y_sort[
                    :, :nVox_to_analyze
                ]  # [n_rds + 1, nVox], including fixation

                # get fixation responses
                # y_fix = np.tile(
                #     y_sel[0], (n_rds, 1)
                # )  # [n_rds, nVox], excluding fixation
                y_fix = y_sel[0]

                # compute the response difference between stimulus and fixation
                y_diff = y_sel[1:] - y_fix  # [n_rds, nVox], excluding fixation

                ## compute standard deviation of response distribution for each
                # voxel across all conditions (including fixation).
                # Thus, the response distribution of each voxel is a collection
                # of responses of that voxel all timepoint in a run.
                # vtc_group = (
                #     vtc_run.groupby(["vox"]).vtc_norm.agg([np.std]).reset_index()
                # )
                # vtc_group = vtc_group.rename(columns={"std": "vtc_std"})
                # y_std = np.array(vtc_group.vtc_std)
                # temp = y_std[id_sort]
                # y_std = np.tile(temp[:nVox_to_analyze], (n_rds, 1))

                # compute s2n with respect to fixation for each voxel
                s2n_val = (
                    y_diff / y_fix.mean() * 100
                )  # [n_rds, nVox], excluding fixation

                # average across these voxels
                signalchange_all_roi[roi, run] = np.mean(
                    s2n_val, axis=1
                )  # [nROIs, nRuns, n_rds], exclude fixation

        return signalchange_all_roi

    def compute_signalchange_fmri_all_sbj(self, t_stat_all_sbjID, nVox_to_analyze):
        """
        compute signal change for all particpants

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

        signalchange_list = []

        t_start = timer()
        signalchange_list.append(
            Parallel(n_jobs=10)(
                delayed(self._signalchange_fmri_sbj)(
                    t_stat_all_sbjID, nVox_to_analyze, sbj
                )
                for sbj in range(self.n_sbjID)
            )
        )
        t_end = timer()
        print(t_end - t_start)

        # extract s2n_list
        n_rds = 3  # rds_types: ards, hmrds, crds
        signalchange_all_sbj = np.zeros(
            (self.n_sbjID, self.n_ROIs, n_rds), dtype=np.float32
        )
        for sbj in range(self.n_sbjID):
            s2n = signalchange_list[0][sbj]  # [nROIs, nRuns, n_rds]

            # average across run
            s2n_avg = np.mean(s2n, axis=1)  # [nROIs, n_rds]

            signalchange_all_sbj[sbj] = s2n_avg  # [n_sbjID, nROIs, n_rds]

        return signalchange_all_sbj

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
