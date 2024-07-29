#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:58:54 2021

    cd /NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM
    
    representational similarity analysis on CMM

@author: cogni
"""



import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import gc
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut

# import sys
# sys.path.insert(1, "../common")
# import utils as util 

# sys.path.insert(1, "../../SVM_Standard")
from SVM import Lib_SVM_PermuteDecode as svm_std

from Common.Common import General

class RSA(General):
    
    def __init__(self):
        
        super().__init__()
    
        
    def _compute_rdm_dist(self, P_data,
                           sbjID,
                           nVox_to_analyze,
                           mtd="correlation"):
        '''
        inner function for compute_rdm_all_sbjID.
        
        generate rdm by measuring the distance between voxel patterns of one
        condition and another voxel patterns from another condition

        Parameters
        ----------
        P_data : TYPE
            DESCRIPTION.
        sbjID : TYPE
            DESCRIPTION.
        nVox_to_analyze : TYPE
            DESCRIPTION.
        mtd : TYPE, optional
            DESCRIPTION. The default is "correlation".

        Returns
        -------
        rdm_all_roi.

        '''
    
        P_temp = P_data.copy()
            
        conds =  P_temp.cond.unique() # there is no fixation cond here
        n_conds = len(conds) # number of conditions
        nRuns = len(P_temp.run.unique()) # number of scan runs
        
        if mtd=="correlation":
            c = 0.5
        else:
            c = 1.0
        
        rdm_all_roi = np.zeros((len(self.ROIs), n_conds, n_conds))
        print('Computing %s RDM for %s' %(mtd, sbjID), end=' ')    
        for roi in range(len(self.ROIs)):
            # filter dataset according to sbjID, roi, and nVox and exclude fixation
            P_roi = P_temp.loc[(P_temp.roi==roi+1) &
                               (P_temp.vox.isin(range(1, nVox_to_analyze+1))) &
                               (P_temp.cond!=1)]
            
            # average voxVal across rep
            P_roi_cond = (P_roi.groupby(["cond", "vox", "run"])
                                .voxVal
                                .agg(["mean"])
                                .reset_index())
            P_roi_cond = P_roi_cond.rename(columns={"mean":"avg"})
            
            # collect data for each condition and run
            df = P_roi_cond.pivot_table(index="run",
                                          columns=["cond", "vox"],
                                          values="avg")
            
            # coordinate: [nRuns, nConds, nVox]
            df2 = np.reshape(np.array(df), (nRuns, n_conds, nVox_to_analyze))
            
            # compute distance for each conditions, excluding fixation
            rdm_roi = np.zeros((n_conds, n_conds))
            for cond1 in range(n_conds): # loop excluding fixation
                for cond2 in range(cond1+1, n_conds):
    #                print(cond1, cond2)
                    
                    # x = df2[:, cond1, :] # [n_run, nVox]
                    # y = df2[:, cond2, :] # [n_run, nVox]                    
                    # dist = np.diag(cdist(x, y, mtd))
                    # rdm_roi[cond1, cond2] = c*np.mean(dist)
                    
                    
                    x = df2[:, cond1, :] # [n_run, nVox]
                    y = df2[:, cond2, :] # [n_run, nVox]
                    # average across n_run
                    x_mean = np.mean(x, axis=0)[np.newaxis, :] # [1, nVox]
                    y_mean = np.mean(y, axis=0)[np.newaxis, :] # [1, nVox]
                    # calculate the distance
                    dist = cdist(x_mean, y_mean, mtd)
                    rdm_roi[cond1, cond2] = c*np.mean(dist)
                    
        
            # copy upper triangle to lower triangle        
            i_lower = np.tril_indices(n_conds, -1)
            rdm_roi[i_lower] = rdm_roi.T[i_lower]  
            
            rdm_all_roi[roi, :, :] = rdm_roi
            print('.', end=' ')
        
        print('')
        
        return rdm_all_roi
        
    def compute_rdm_all_sbjID(self,
                              nVox_to_analyze,
                              mtd="correlation"):
        '''
        generate rdm for each sbjID

        Parameters
        ----------
        nVox_to_analyze : scalar
            the number of voxels used for the analysis.
            
        mtd : TYPE, optional
            DESCRIPTION. The default is "correlation".

        Returns
        -------
        None.

        '''
                        
        # nVox = 300
        
        # mtd = "correlation"
        rdm_all = np.zeros((len(self.sbjID_all), len(self.ROIs), 6, 6))
        
        for sbj in range(len(self.sbjID_all)):
            # load P_data
            sbjID = self.sbjID_all[sbj]
            nRuns = self.nRuns_all[sbj]
            P_data = self.load_P_data(sbjID, nRuns)
            
            # process P_data
            P_data = self.label_P_data(P_data)
            
            # normalize P_data
            P_data = self.normalize_P_data(P_data) # fixation is excluded here
            
            # compute rdm and store it according to partiality
            rdm_allROI = self._compute_rdm_dist(P_data,
                                                  sbjID,
                                                  nVox_to_analyze,
                                                  mtd)
            
            rdm_all[sbj] = rdm_allROI
            
        self.rdm_fmri_all = rdm_all
                
        
    def wrapper_compute_rdm_cmm(self, CMM, DispCol,
                                mu_stim_crossed, std_stim_crossed,
                                mu_stim_uncrossed, std_stim_uncrossed):
        '''
        

        Parameters
        ----------
        CMM : TYPE
            DESCRIPTION.
        DispCol : TYPE
            DESCRIPTION.
        mu_stim_crossed : TYPE
            DESCRIPTION.
        std_stim_crossed : TYPE
            DESCRIPTION.
        mu_stim_uncrossed : TYPE
            DESCRIPTION.
        std_stim_uncrossed : TYPE
            DESCRIPTION.

        Returns
        -------
        rdm_corr_bootsrap : [len(f_batch), n_bootstrap, 6, 6] np.array
            DESCRIPTION.
        rdm_match_bootsrap : [len(f_batch), n_bootstrap, 6, 6] np.array
            DESCRIPTION.

        '''
            
        
        #[len(f_batch), n_bootstrap, 6, 6] np.array
        rdm_corr_bootsrap = np.zeros((len(CMM.f_batch), CMM.n_bootstrap, 6, 6),
                                     dtype=np.float32)
        rdm_match_bootsrap = np.zeros((len(CMM.f_batch), CMM.n_bootstrap, 6, 6),
                                     dtype=np.float32)
        
        for i in range(CMM.n_epoch):
            
            # generate hypothetical disparity tuning column
            dispTuning_corr_hypothetical_bootstrap = \
                CMM.create_dispTuning_corr_bootstrap(DispCol, i)
                
            dispTuning_match_hypothetical_bootstrap = \
                CMM.create_dispTuning_match_bootstrap(DispCol, i)
            
            # compute rdm
            id_start = i*CMM.n_miniBootstrap
            id_end = id_start + CMM.n_miniBootstrap
            rdm_corr_bootsrap[:,id_start:id_end], rdm_match_bootsrap[:,id_start:id_end] = \
                    CMM.compute_rdm_corr(dispTuning_corr_hypothetical_bootstrap,
                                         dispTuning_match_hypothetical_bootstrap,
                                         mu_stim_crossed, std_stim_crossed,
                                         mu_stim_uncrossed, std_stim_uncrossed)
                    
        del dispTuning_corr_hypothetical_bootstrap, dispTuning_match_hypothetical_bootstrap
        gc.collect()
                    
        return rdm_corr_bootsrap, rdm_match_bootsrap
                            
        
    def compute_diss_fmri_cmm(self, CMM,
                              rdm_corr_bootsrap, rdm_match_bootsrap):
        '''
        Compute the dissimilarity between fmri response patterns and model patterns
    
        Args:
            rdm_all ([sbjID_all, ROIs, 6, 6] np.array): rdm for each sbj in sbjID_all
                                        and ROI. 
                                        Obtained from Lib_RSA_bw18_005_3.fxCompute_rdm_dist
                                    
            CMM : object, obtained from correlation-match model (CMM) class
            
                rdm_corr_bootsrap ([len(f_batch), n_bootstrap, 6, 6] np.array): rdm for correlation model.
                                Obtained from self.wrapper_compute_rdm
                                
                rdm_match_bootsrap ([len(f_batch), n_bootstrap, 6, 6] np.array): rdm for mathc model.
                                Obtained from self.wrapper_compute_rdm
    
        Returns:
            diss_corr_all ([ROIs, sbjID_all] np.array): dissimilarity between fmri 
                        response pattern and correlation model in sbjID_all.
                                        
            diss_match_all ([ROIs, sbjID_all] np.array): dissimilarity between fmri 
                        response pattern and match model in sbjID_all.
                                    
        '''
        
        # average across n_bootstrap
        rdm_corr_avg = np.mean(rdm_corr_bootsrap, axis=1) # [len(f_batch), 6, 6]
        rdm_match_avg = np.mean(rdm_match_bootsrap, axis=1) # [len(f_batch), 6, 6]
        
        
        diss_corr = np.zeros((len(self.ROIs), len(CMM.f_batch), 
                              len(self.sbjID_all), 6), 
                             dtype=np.float32)
        diss_match = np.zeros((len(self.ROIs), len(CMM.f_batch), 
                               len(self.sbjID_all), 6), 
                              dtype=np.float32)
        
        for roi in range(len(self.ROIs)):
            
            for i_f in range(len(CMM.f_batch)):
                
                print("compute dissimilarity, ROI:{}, f:{}"
                      .format(self.ROIs[roi], str(CMM.f_batch[i_f])))
                
                for sbj in range(len(self.rdm_fmri_all)):
                    
                    # loop over rows
                    for i in range(6):
                        x = self.rdm_fmri_all[sbj, roi, i]                
                        y = rdm_corr_avg[i_f, i]
                        
                        # delete diagonal element
                        x = np.delete(x, i)
                        y = np.delete(y, i)
                        
                        # compute dissimilariy between data and correlation model
                        diss_corr[roi, i_f, sbj, i] = 0.5*(1 - stats.pearsonr(x, y)[0])
                        # diss_corr[roi, i_f, sbj, i] = 0.5*(1 - stats.spearmanr(x, y)[0])
                        # diss_corr[roi, i_f, sbj, i] = 0.5*(1 - stats.kendalltau(x, y)[0])
                        
                        # compute dissimilarity between data and match model
                        y = rdm_match_avg[i_f, i]
                        y = np.delete(y, i)
                        diss_match[roi, i_f, sbj, i] = 0.5*(1 - stats.pearsonr(x, y)[0])
                        # diss_match[roi, i_f, sbj, i] = 0.5*(1 - stats.spearmanr(x, y)[0])
                        # diss_match[roi, i_f, sbj, i] = 0.5*(1 - stats.kendalltau(x, y)[0])
                        
        ## avg across rows
        diss_corr_all = np.mean(diss_corr, axis=3)
        diss_match_all = np.mean(diss_match, axis=3)
                    
        return diss_corr_all, diss_match_all
    
    

    def compute_diss_fmri_cmm2(self, CMM,
                               rdm_corr_bootsrap, rdm_match_bootsrap):
        '''
        Compute the dissimilarity between fmri response patterns and model patterns
    
        Args:
            rdm_all ([sbjID_all, ROIs, 6, 6] np.array): rdm for each sbj in sbjID_all
                                        and ROI. 
                                        Obtained from Lib_RSA_bw18_005_3.fxCompute_rdm_dist
                                    
            CMM : object, obtained from correlation-match model (CMM) class
            
                CMM.rdm_corr ([len(f_batch), n_bootstrap, 6, 6] np.array): rdm for correlation model.
                                Obtained from run_corr_match_model
                                
                CMM.rdm_match ([len(f_batch), n_bootstrap, 6, 6] np.array): rdm for mathc model.
                                Obtained from run_corr_match_model
    
        Returns:
            diss_corr_all ([ROIs, sbjID_all] np.array): dissimilarity between fmri 
                        response pattern and correlation model in sbjID_all.
                                        
            diss_match_all ([ROIs, sbjID_all] np.array): dissimilarity between fmri 
                        response pattern and match model in sbjID_all.
                                    
        '''
        
        # average across n_bootstrap
        rdm_corr_avg = np.mean(rdm_corr_bootsrap, axis=1) # [len(f_batch), 6, 6]
        rdm_match_avg = np.mean(rdm_match_bootsrap, axis=1) # [len(f_batch), 6, 6]        
        
        diss_corr = np.zeros((len(self.ROIs), len(CMM.f_batch), 
                              len(self.sbjID_all)), 
                             dtype=np.float32)
        diss_match = np.zeros((len(self.ROIs), len(CMM.f_batch), 
                               len(self.sbjID_all)), 
                              dtype=np.float32)
        
        for roi in range(len(self.ROIs)):
            
            for i_f in range(len(CMM.f_batch)):
                
                print("compute dissimilarity, ROI:{}, f:{}"
                      .format(self.ROIs[roi], str(CMM.f_batch[i_f])))
                
                for sbj in range(len(self.rdm_fmri_all)):
                
                    x = self.rdm_fmri_all[sbj, roi][np.triu_indices(6, k=1)]                
                    y = rdm_corr_avg[i_f][np.triu_indices(6, k=1)]
                                        
                    # compute dissimilariy between data and correlation model
                    diss_corr[roi, i_f, sbj] = 0.5*(1 - stats.pearsonr(x, y)[0])
                    # diss_corr[roi, i_f, sbj] = 0.5*(1 - stats.spearmanr(x, y)[0])
                    # diss_corr[roi, i_f, sbj] = 0.5*(1 - stats.kendalltau(x, y)[0])
                    
                    # compute dissimilarity between data and match model
                    y = rdm_match_avg[i_f][np.triu_indices(6, k=1)]
                    diss_match[roi, i_f, sbj] = 0.5*(1 - stats.pearsonr(x, y)[0])
                    # diss_match[roi, i_f, sbj] = 0.5*(1 - stats.spearmanr(x, y)[0])
                    # diss_match[roi, i_f, sbj] = 0.5*(1 - stats.kendalltau(x, y)[0])
                    
        
        return diss_corr, diss_match
    
    
    
            
    def _compute_kendalltau_up_single_roi(self, rdm_fmri_all,
                                          sbjID_bootsrap,
                                          n_bootstrap,
                                          roi):
        '''
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

        '''
        
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
    
    def _compute_kendalltau_low_single_roi(self, rdm_fmri_all,
                                          sbjID_bootsrap,
                                          n_bootstrap,
                                          sbj, roi):
        '''
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

        '''
        
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
    
    def compute_noiseCeiling(self, rdm_fmri_all,
                             n_bootstrap):
        '''
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

        '''
        
        n_sbj, n_roi, _, _ = rdm_fmri_all.shape
        
        # calculating upper bound
        print("compute upper bound noise ceiling")
        sbjID_bootsrap = np.arange(n_sbj)
        temp = []
        temp.append(Parallel(n_jobs=8)
                 (delayed(self._compute_kendalltau_up_single_roi)
                  (rdm_fmri_all, sbjID_bootsrap, n_bootstrap, roi)
                  for roi in range(n_roi)))
        
        ## unpack
        kendalltau_up = np.zeros((n_roi, n_bootstrap), dtype=np.float32)
        for roi in range(n_roi):
            kendalltau_up[roi] = temp[0][roi]
            
        
        ## calculating lower bound
        kendalltau_low = np.zeros((n_roi, n_bootstrap, n_sbj), dtype=np.float32)
        sbjID_all = np.arange(n_sbj)
        
        for sbj in range(n_sbj):            
            print("compute lower bound noise ceiling, sbj_out: {}/{}"
                  .format(str(sbj+1), str(n_sbj)))
            
            sbjID_bootsrap = sbjID_all[sbjID_all!=sbj]            
            
            temp = []
            temp.append(Parallel(n_jobs=8)
                        (delayed(self._compute_kendalltau_low_single_roi)
                         (rdm_fmri_all, sbjID_bootsrap, n_bootstrap, sbj, roi)
                         for roi in range(n_roi)))
            
            # unpack
            for roi in range(n_roi):
                kendalltau_low[roi, :, sbj] = temp[0][roi]
                    
        return kendalltau_low, kendalltau_up
    
        
    def compute_rdm_svm(self, useVoxVal,
                        nVox_to_analyze):
        '''
        compute rdm based on svm

        Parameters
        ----------
        useVoxVal : string
            ex: "voxVal_noBaseline".
        nVox_to_analyze : scalar
            the number of voxels used for analysis.
            ex: 250

        Returns
        -------
        rdm_svm_avg : [nROIs, 6, 6] np.array
            svm-based rdm.

        '''
        
            
        rdm_svm = np.zeros((len(self.ROIs), len(self.sbjID_all), 6, 6), 
                           dtype=np.float32)
        
        # rdm_svm = np.zeros((8, 23, 6, 6), 
        #                     dtype=np.float32)
        
        for i in range(6):
            for j in range(i+1, 6):
                    
                comp_pair = [i+2, j+2]
        
                
                # start decoding
                # [nVox, roi, fold_id, acc, acc_mean, acc_sem, sbjID, roi_str, comp_pair]
                score = svm_std.fxDecode_allSbjID(self.sbjID_all,
                                                  self.sbj_nRuns,
                                                  self.ROIs,
                                                  comp_pair,
                                                  useVoxVal, [nVox_to_analyze],
                                                  n_jobs=len(self.sbjID_all))
                
                # average across fold_od
                score_avg = (score.groupby(["nVox", "roi", "sbjID", "comp_pair"])
                                  .acc.agg(["mean"])
                                  .reset_index())
                score_avg = score_avg.rename(columns={"mean":"acc_avg"})
                
                # get acc for each roi
                for roi in range(len(self.ROIs)):
                    rdm_svm[roi, :, i, j] = score_avg.loc[score_avg.roi==roi+1].acc_avg
        

        # average across sbjID
        rdm_svm_avg = np.mean(rdm_svm, axis=1)
        
        # copy upper to lower triangle elements
        i_lower = np.tril_indices(6, k=-1)
        for roi in range(8):
            temp = rdm_svm_avg[roi]
            temp[i_lower] = temp.T[i_lower]
            # fill diagonal with 0.5
            np.fill_diagonal(temp, 0.5)
            
            rdm_svm_avg[roi] = temp
                
                    
        return rdm_svm_avg
                    
                    
                
    def plotHeat_rdm_svm(self, rdm_svm_avg):
        
        
        conds = ['aRDS_cross','aRDS_uncross',
                 'hmRDS_cross','hmRDS_uncross',
                 'cRDS_cross','cRDS_uncross']         
        
        
        plt.style.use('seaborn-colorblind')
        sns.set()
        sns.set(context='paper',
                style='white',
                font_scale=2,
                palette='deep')
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['axes.labelweight'] = 'bold'
        
        # estimate v_min and v_max for cbar
        v_min = 0.5
        v_max = 0.8
        
        figsize = (10, 7)
        n_row = 3
        n_col = 3
        
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col,
                                figsize=figsize,
                                sharex=True, sharey=True)
            
        fig.text(0.5, 1.1,
                 "SVM-based RDM",
                 ha="center")
        fig.text(-0.2, 0.5,
                 "Conditions",
                 va="center",
                 rotation=90)
        fig.text(0.5, -0.3,
                 "Conditions",
                 ha="center")
        
        fig.tight_layout()
        
        plt.subplots_adjust(wspace=0.2,
                            hspace=0.3)
        
        cmap = "jet"
        for roi in range(len(self.ROIs)):
            id_row = np.int(roi/n_row)
            id_col = roi%n_col         
            

            sns.heatmap(rdm_svm_avg[roi],
                        cmap=cmap,
                        vmin=v_min,
                        vmax=v_max,
                        xticklabels=conds,
                        yticklabels=conds,
                        ax=axes[id_row, id_col])
            axes[id_row, id_col].set_title(self.ROIs[roi],
                                           pad=10)
            

        
        fig.savefig("../../../../Plots/Dissertation/theory/Disparity_column_model/PlotHeat_cmm_noRF_rdm_svm.pdf",
                    dpi=500,
                    bbox_inches="tight")
        
        # fig.savefig("../../../../Plots/Dissertation/theory/Disparity_column_model/PlotHeat_cmm_noRF_rdm_svm.png",
        #             dpi=500,
        #             bbox_inches="tight")
                
        
        
    def plotScatter_diss_corr_match_avg(self, diss_corr_all, diss_match_all):
        '''
        scatter plot the dissimilarity between rdm_fmri and rdm_cmm in 
        corr-match dissimmilarity space.
        
        this plot is the average across all frequency channels

        Parameters
        ----------
        diss_corr_all : TYPE
            DESCRIPTION.
        diss_match_all : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        
        # avg across spatial freq
        diss_corr_avg = diss_corr_all.mean(axis=1)
        diss_match_avg = diss_match_all.mean(axis=1)
    
    
        ## start plotting
        
        markers = ["s", "o", ">", "^", "<", "v", "X", "D"]  
        
        
        figsize = (7, 7)
        n_row = 1
        n_col = 1
        
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col,
                                 figsize=figsize,
                                 sharex=True, sharey=True)
        
        fig.text(0.5, 1.04,
                 "Correlation Dissimilarity VS Match Dissimilarity, AVG, #Voxels={}" 
                 .format(str(self.nVox)),
                 fontsize=18,
                 ha="center")
        fig.text(-0.06, 0.5,
                 "Correlation dissimilarity, 0.5(1 - r)",
                 fontsize=16,
                 va="center",
                 rotation=90)
        fig.text(0.5, -0.06,
                 "Match dissimilarity, 0.5(1 - r)",
                 fontsize=16,
                 ha="center")
        
        fig.tight_layout()
        
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        
        
        for i in range(len(self.ROIs)):
            x = np.mean(diss_match_avg[i])
            y = np.mean(diss_corr_avg[i])
            x_err = sem(diss_match_avg[i])
            y_err = sem(diss_corr_avg[i])
            axes.errorbar(x, y,
                         xerr=x_err, yerr=y_err,
                         elinewidth=2,
                         fmt=markers[i],
                         c="black",
                         ms=12)
            
            
        axes.legend(self.ROIs, fontsize=10)
        
        # plot diagonal line
        y_low = 0.35
        y_up = 0.65
        step = 0.05
        
        axes.plot([y_low, y_up], [y_low, y_up], 
                     "r--",
                     linewidth=3)
        
        axes.set_title("All",
                          fontsize=16)

        axes.set_xticks(np.round(np.arange(y_low, y_up, step), 2))
        axes.set_xticklabels(np.round(np.arange(y_low, y_up, step), 2),
                            fontsize=16)
        axes.set_yticks(np.round(np.arange(y_low, y_up, step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, step), 2),
                            fontsize=16)
        
        axes.text(0.5, 0.35,
                     "Correlation dominant",
                     fontsize=18,
                     color="gray",
                     weight="bold",
                     alpha=0.8,
                     ha="center")
        axes.text(0.5, 0.65,
                     "Match dominant",
                     fontsize=18,
                     color="gray",
                     weight="bold",
                     alpha=0.8,
                     ha="center")
        
        # fig.savefig("../../../../Plots/RSA/PlotScatter_corrDiss_vs_matchDiss_AVG.pdf",
        #             dp=500,
        #             bbox_inches="tight")
        
    def plotHeat_rdm_freq(self, rdm_corr_bootsrap, rdm_match_bootsrap,
                          f_batch, i_f):
        '''
        heatmap plot rdm at given spatial frequency

        Parameters
        ----------
        rdm_corr_bootsrap : TYPE
            DESCRIPTION.
        rdm_match_bootsrap : TYPE
            DESCRIPTION.
        f_batch : TYPE
            DESCRIPTION.
        i_f : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # average across bootstrap and get data at frequency f
        rdm_corr_f = np.mean(rdm_corr_bootsrap, axis=1)[i_f]
        rdm_match_f = np.mean(rdm_match_bootsrap, axis=1)[i_f]
        
        conds_plot = ['aRDS_cross','aRDS_uncross',
                           'hmRDS_cross','hmRDS_uncross',
                           'cRDS_cross','cRDS_uncross'] # conditions for plotting
        
        
        ## plot heatmap
        figsize = (12, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col,
                                figsize=figsize,
                                sharex=True, sharey=True)
        
        fig.text(0.5, 1.05,
                 "RDM_Correlation VS RDM_Match, f={}, #voxel={}"
                 .format(str(f_batch[i_f]), str(self.nVox)),
                 ha="center",
                 fontsize=16)
        fig.text(-0.12, 0.5,
                 "Conditions",
                 va="center",
                 rotation=90,
                 fontsize=14)
        fig.text(0.5, -0.25,
                 "Conditions",
                 ha="center",
                 fontsize=14)
        
        fig.tight_layout()
        
        plt.subplots_adjust(wspace=0.2,
                            hspace=0.3)
        
        sns.heatmap(rdm_corr_f,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=1.0,
                    xticklabels=conds_plot,
                    yticklabels=conds_plot,
                    ax=axes[0])
        axes[0].set_title("Correlation computation")
        
        sns.heatmap(rdm_match_f,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=1.0,
                    xticklabels=conds_plot,
                    yticklabels=conds_plot,
                    ax=axes[1])
        axes[1].set_title("Match computation")
        
        
        # fig.savefig("../../../../Plots/Voxel_encoding/PlotHeat_rdm_f{}.pdf"
        #             .format(str(f_batch[i_f])),
        #             dp=500,
        #             bbox_inches="tight")
        
        
    def plotHeat_rdm_avg(self, rdm_corr_bootsrap, rdm_match_bootsrap):
        
        # average across bootstrap and frequency
        rdm_corr_avg = np.mean(np.mean(rdm_corr_bootsrap, axis=1), 
                               axis=0)
        rdm_match_avg = np.mean(np.mean(rdm_match_bootsrap, axis=1),
                                axis=0)
        
        conds_plot = ['aRDS_cross','aRDS_uncross',
                           'hmRDS_cross','hmRDS_uncross',
                           'cRDS_cross','cRDS_uncross'] # conditions for plotting
        
        ## plot heatmap
        figsize = (12, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col,
                                figsize=figsize,
                                sharex=True, sharey=True)
        
        fig.text(0.5, 1.05,
                 "RDM_Correlation VS RDM_Match, AVG, #voxel={}"
                 .format(str(self.nVox)),
                 ha="center",
                 fontsize=16)
        fig.text(-0.12, 0.5,
                 "Conditions",
                 va="center",
                 rotation=90,
                 fontsize=14)
        fig.text(0.5, -0.25,
                 "Conditions",
                 ha="center",
                 fontsize=14)
        
        fig.tight_layout()
        
        plt.subplots_adjust(wspace=0.2,
                            hspace=0.3)
        
        sns.heatmap(rdm_corr_avg,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=1.0,
                    xticklabels=conds_plot,
                    yticklabels=conds_plot,
                    ax=axes[0])
        axes[0].set_title("Correlation computation")
        
        sns.heatmap(rdm_match_avg,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=1.0,
                    xticklabels=conds_plot,
                    yticklabels=conds_plot,
                    ax=axes[1])
        axes[1].set_title("Match computation")
        
        
        # fig.savefig("../../../../Plots/Voxel_encoding/PlotHeat_rdm_avg.pdf",
        #             dp=500,
        #             bbox_inches="tight")