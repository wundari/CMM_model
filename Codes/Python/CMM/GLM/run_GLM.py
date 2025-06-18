#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:09:00 2020

    run standard GLM analysis

    working dir:
    cd /NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM

@author: cogni
"""

# %% load necessary modules
import GLM.glm_v2 as glm_v2

import numpy as np
import pandas as pd
import scipy.io as sio

# %%
glm = glm_v2.GLM()

# %%
## compute beta glm
mtd_normalization = 2
beta_all_sbjID, beta_avg_all_sbjID = glm.compute_glm_all_sbjID(mtd_normalization)

glm.plotBar_beta(beta_all_sbjID)

## diagnose glm
sbj = 0
sbjID = glm.sbjID_all[sbj]
# load vtc
vtc = glm.load_vtc(sbj)

# load stimulus timing parameters
vtc_stimID = sio.loadmat(
    "../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
)["paramIdFull"]

# label vtc
vtc_labeled = glm.label_vtc(vtc, vtc_stimID)

roi = 0
run = 0
vox = 0
a = vtc_labeled.loc[
    (vtc_labeled.roi == roi) & (vtc_labeled.run == run) & (vtc_labeled.vox == vox)
]

glm.diagnose_glm(beta_all_sbjID, sbj, roi, vox, run, mtd_normalization)


## compute t_stat
t_stat_all_sbjID = glm.compute_t_stat_all_sbjID(beta_all_sbjID, mtd_normalization)

# %% load beta and t values
beta_all_sbjID = np.load("../../../../Data/MVPA/beta_all_sbjID.npy", allow_pickle=True)
t_stat_all_sbjID = np.load(
    "../../../../Data/MVPA/t_stat_all_sbjID.npy", allow_pickle=True
)
# %%

## compute percent signal change
nVox_to_analyze = 250
signal_change_all_sbj = glm.compute_signal_change_all_sbj(
    t_stat_all_sbjID, nVox_to_analyze
)
signal_change_avg = np.mean(signal_change_all_sbj, axis=0)

# %%
# sort beta according to t-stat in descending order
nVox_to_analyze = 250
beta_sort_all_sbjID = glm.sort_beta_all_sbjID(
    beta_all_sbjID, t_stat_all_sbjID, nVox_to_analyze
)

sbj = 0
sbjID = glm.sbjID_all[sbj]
# load vtc data
col_names = ["run", "roi", "timepoint", "vox", "vtc_value"]
# vtc_mat = sio.loadmat("../../../../Data/VTC_extract_smoothed/vtc_{}.mat".format(sbjID))[
#     "vtc_extract2"
# ]
vtc_mat = sio.loadmat("../../../../Data/VTC_extract_smoothed/vtc_{}_V1.mat".format(sbjID))[
    "vtc_extract"
]
# convert to pandas dataframe
col_names = ["run", "timepoint", "vox"]
vtc = pd.DataFrame(data=vtc_mat, columns=col_names)

# load stimulus id data associated with vtc
# [block_id, run]
vtc_stimID = sio.loadmat(
    "../../../Data/VTC_stimID/paramStimFull_bw18_005_16Sec_{}".format(sbjID)
)["paramIdFull"]

# label vtc data
vtc_labeled = glm.label_vtc(vtc, vtc_stimID)

# create design matrix
designMatrix_conv = glm.create_designMatrix(vtc_stimID, sbj)


beta_sbj = beta_all_sbjID[sbj]

t_axis = np.arange(16)
t_peak = 4
t_undershoot = 6
c_undershoot = 0.35
# gamma1  = gamma.pdf(t_axis, t_peak)
# gamma2 = gamma.pdf(t_axis, t_undershoot)
# hrf = gamma1 - 0.2*gamma2
# plt.plot(hrf/np.max(hrf))

hrf = glm.create_hrf2gamma(t_axis, c_undershoot, t_peak, t_undershoot)
plt.plot(hrf)


rtc = glm.load_rtc_file(sbj, run)

x = np.arange(0, 1, 1e-4)
y = x**x
plt.plot(x, y)


import scipy.io as sio

vtc_stimID = sio.loadmat("vtc_SF")["vtc_extract"]
col_names = ["run", "roi", "timepoint", "vox", "vtc_value"]
vtc = pd.read_csv("vtc_SF.csv", names=col_names)

roi = 6
vox = 1
run = 1
vtc_run = np.array(
    vtc.loc[
        (vtc.roi == roi + 1) & (vtc.vox == vox + 1) & (vtc.run == run + 1)
    ].vtc_value
)

vtc_run2 = vtc_stimID[run, roi, :, vox]

plt.plot(vtc_run, vtc_run2)
