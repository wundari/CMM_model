"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/MVPA/run_MVPA_XDecode.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/
Created Date: 2022-06-30 16:43:05
Author: Bayu G. Wundari
-----
Last Modified: 2022-06-30 18:57:54
Modified By: Bayu G. Wundari
-----
Copyright (c) 2022 National Institute of Information and Communications Technology (NICT)

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
cross-decoding analysis
"""

# %% load necessary modules
import numpy as np
import pandas as pd

from GLM.GLM_v2 import GLM as GLM
from MVPA_XDecode import MVPA_XDecode
from MVPA.PlotMVPA_XDecode import PlotMVPA_XDecode

# %% GLM analysis
glm = GLM()

## compute beta glm
mtd_normalization = 2  # percent signal change with respect to average across timepoints
beta_all_sbjID, beta_avg_all_sbjID = glm.compute_glm_all_sbjID(mtd_normalization)
# save files
# np.save("../../../Data/MVPA/beta_all_sbjID.npy", beta_all_sbjID)
# np.save("../../../Data/MVPA/beta_avg_all_sbjID.npy", beta_avg_all_sbjID)

## compute t_stat
t_stat_all_sbjID = glm.compute_t_stat_all_sbjID(beta_all_sbjID, mtd_normalization)
# np.save("../../../Data/MVPA/t_stat_all_sbjID.npy", t_stat_all_sbjID)

# %%start mvpa cross-decoding
# read beta and t_stat data
beta_all_sbjID = np.load("../../../Data/MVPA/beta_all_sbjID.npy", allow_pickle=True)
t_stat_all_sbjID = np.load("../../../Data/MVPA/t_stat_all_sbjID.npy", allow_pickle=True)
mvpa_xDecode = MVPA_XDecode(beta_all_sbjID, t_stat_all_sbjID)

# %% two ways cross-decoding crds vs. ards
rds_train_test = "crds_ards"
comp_pair_train = [5, 6]  # crds
comp_pair_test = [1, 2]  # ards
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_crds_vs_ards_df = mvpa_xDecode.XDecode_allSbjID_twoway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_crds_vs_ards_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_crds_ards_shift_zscore_vtc.pkl"
)

## two ways cross-decoding crds vs. hmrds
rds_train_test = "crds_hmrds"
comp_pair_train = [5, 6]  # crds
comp_pair_test = [3, 4]  # hmrds
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_crds_vs_hmrds_df = mvpa_xDecode.XDecode_allSbjID_twoway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_crds_vs_hmrds_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_crds_hmrds_shift_zscore_vtc.pkl"
)

## two ways cross-decoding hmrds vs. ards
rds_train_test = "hmrds_ards"
comp_pair_train = [3, 4]  # crds
comp_pair_test = [1, 2]  # ards
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_hmrds_vs_ards_df = mvpa_xDecode.XDecode_allSbjID_twoway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_hmrds_vs_ards_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_hmrds_ards_shift_zscore_vtc.pkl"
)

## one way cross-decoding crds vs. ards
rds_train_test = "crds_ards"
comp_pair_train = [5, 6]  # crds
comp_pair_test = [1, 2]  # ards
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_crds_vs_ards_df = mvpa_xDecode.XDecode_allSbjID_oneway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_crds_vs_ards_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_oneway_crds_ards_shift_zscore_vtc.pkl"
)
## one way cross-decoding ards vs. crds
rds_train_test = "ards_crds"
comp_pair_train = [1, 2]  # ards
comp_pair_test = [5, 6]  # crds
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_crds_vs_ards_df = mvpa_xDecode.XDecode_allSbjID_oneway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_crds_vs_ards_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_oneway_ards_crds_shift_zscore_vtc.pkl"
)

## one way cross-decoding crds vs. hmrds
rds_train_test = "crds_hmrds"
comp_pair_train = [5, 6]  # crds
comp_pair_test = [3, 4]  # hmrds
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_crds_vs_hmrds_df = mvpa_xDecode.XDecode_allSbjID_oneway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_crds_vs_hmrds_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_oneway_crds_hmrds_shift_zscore_vtc.pkl"
)

## one way cross-decoding hmrds vs. crds
rds_train_test = "hmrds_crds"
comp_pair_train = [3, 4]  # hmrds
comp_pair_test = [5, 6]  # crds
flip_label = 1  # -1 -> flip test depth label to make prediction acc. above chance
# nVox_list = np.arange(25, 325, 25)
nVox_list = np.arange(225, 325, 25)
xDecode_crds_vs_hmrds_df = mvpa_xDecode.XDecode_allSbjID_oneway(
    comp_pair_train, comp_pair_test, nVox_list, flip_label, rds_train_test
)
# save file
xDecode_crds_vs_hmrds_df.to_pickle(
    "../../../Data/MVPA/mvpa_xDecode_oneway_hmrds_crds_shift_zscore_vtc.pkl"
)

# %% cross-decoding with permutation test
n_permute = 10000
permuteXDecode_all_df = mvpa_xDecode.permuteXDecode_allSbjID(
    comp_pair_train, comp_pair_test, nVox_list, n_permute, rds_train_test
)
# xDecode_all_df.to_pickle(
#     "../../../../Data/MVPA/mvpa_xDecode_{}_shift_zscore_vtc.pkl"
#     .format(rds_train_test)
# )
# permuteXDecode_all_df.to_pickle(
#     "../../../../Data/MVPA/mvpa_xDecode_{}_shift_zscore_vtc_permute_{}.pkl"
#     .format(rds_train_test, n_permute)
# )

xDecode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_xDecode_{}_shift_zscore_vtc.pkl".format(rds_train_test)
)
permuteXDecode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_xDecode_twoway_{}_shift_zscore_vtc_permute_{}.pkl".format(
        rds_train_test, n_permute
    )
)

#####################
## statistical testing

# crds vs ards
rds_train_test = "crds_ards"
n_bootstrap = 10000
nVox_to_analyze = 250
alpha = 0.05
stat_at_vox_df = mvpa_xDecode.compute_stat_xDecode_permute_bootstrap(
    xDecode_crds_vs_ards_df,
    permuteXDecode_all_df,
    nVox_to_analyze,
    n_bootstrap,
    rds_train_test,
    alpha,
)
# save to excel
stat_at_vox_df.to_excel(
    "../../../Data/MVPA/stat_xDecode_crds_vs_ards_shift_zscore_vtc_{}.xlsx".format(
        nVox_to_analyze
    )
)

# hmrds vs ards
rds_train_test = "hmrds_ards"
stat_at_vox_df = mvpa_xDecode.compute_stat_xDecode_permute_bootstrap(
    xDecode_hmrds_vs_ards_df,
    permuteXDecode_all_df,
    nVox_to_analyze,
    n_bootstrap,
    rds_train_test,
    alpha,
)
# save to excel
stat_at_vox_df.to_excel(
    "../../../Data/MVPA/stat_xDecode_hmrds_vs_ards_shift_zscore_vtc_{}.xlsx".format(
        nVox_to_analyze
    )
)


stat_allVox = []
for v in range(len(nVox_list)):
    nVox_to_analyze = nVox_list[v]

    temp = mvpa_xDecode.compute_stat_xDecode_permute_bootstrap(
        xDecode_crds_vs_ards_df,
        permuteXDecode_all_df,
        nVox_to_analyze,
        n_bootstrap,
        rds_train_test,
        alpha,
    )

    stat_allVox.append(temp)

stat_allVox_df = pd.concat(stat_allVox)
stat_allVox_df.to_excel(
    "../../../Data/MVPA/stat_xDecode_crds_vs_ards_shift_zscore_vtc_allVox.xlsx"
)

##################
# %% plotting

plot_mvpa_xDecode = PlotMVPA_XDecode()
nVox_to_analyze = 250
save_flag = 0

# %% read file
xDecode_crds_vs_ards_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_xDecode_crds_ards_shift_zscore_vtc.pkl"
)
xDecode_crds_vs_hmrds_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_xDecode_crds_hmrds_shift_zscore_vtc.pkl"
)

# xDecode_crds_vs_ards_df = pd.read_pickle(
#     "../../../../Data/MVPA/mvpa_xDecode_oneway_crds_ards_shift_zscore_vtc.pkl"
# )
# xDecode_crds_vs_hmrds_df = pd.read_pickle(
#     "../../../../Data/MVPA/mvpa_xDecode_oneway_crds_hmrds_shift_zscore_vtc.pkl"
# )

# xDecode_crds_vs_ards_df = pd.read_pickle(
#     "../../../Data/MVPA/mvpa_xDecode_oneway_ards_crds_shift_zscore_vtc.pkl"
# )
# xDecode_crds_vs_hmrds_df = pd.read_pickle(
#     "../../../Data/MVPA/mvpa_xDecode_oneway_hmrds_crds_shift_zscore_vtc.pkl"
# )

# %% crds vs ards
rds_train_test = "crds_ards"
# plotBox
plot_mvpa_xDecode.plotBox_xDecode_at_nVox(
    xDecode_crds_vs_ards_df,
    permuteXDecode_all_df,
    nVox_to_analyze,
    rds_train_test,
    save_flag,
)

## disparity bias
plot_mvpa_xDecode.plotBox_xDecode_disp_bias_at_nVox(
    xDecode_crds_vs_ards_df, nVox_to_analyze, rds_train_test, save_flag
)

plot_mvpa_xDecode.plotLine_xDdecode_avg(
    xDecode_crds_vs_ards_df, permuteXDecode_all_df, rds_train_test, save_flag
)

# %% crds vs hmrds
rds_train_test = "crds_hmrds"
# plotBox
plot_mvpa_xDecode.plotBox_xDecode_at_nVox(
    xDecode_crds_vs_hmrds_df,
    permuteXDecode_all_df,
    nVox_to_analyze,
    rds_train_test,
    save_flag,
)

## disparity bias
plot_mvpa_xDecode.plotBox_xDecode_disp_bias_at_nVox(
    xDecode_crds_vs_hmrds_df, nVox_to_analyze, rds_train_test, save_flag
)

# plot_mvpa_xDecode.plotLine_xDdecode_avg(
#     xDecode_crds_vs_ards_df, permuteXDecode_all_df, rds_train_test, save_flag
# )

# %% hmrds vs ards
n_permute = 1000
rds_train_test = "hmrds_ards"
save_flag = 1
# read data
xDecode_hmrds_vs_ards_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_xDecode_twoway_hmrds_ards_shift_zscore_vtc.pkl"
)
permuteXDecode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_xDecode_twoway_{}_shift_zscore_vtc_permute_{}.pkl".format(
        rds_train_test, n_permute
    )
)

# plotBox
plot_mvpa_xDecode.plotBox_xDecode_hmrds_ards_at_nVox(
    xDecode_hmrds_vs_ards_df,
    permuteXDecode_all_df,
    nVox_to_analyze,
    save_flag,
)

## disparity bias
# plot_mvpa_xDecode.plotBox_xDecode_disp_bias_at_nVox(
#     xDecode_hmrds_vs_ards_df, nVox_to_analyze, rds_train_test, save_flag
# )

plot_mvpa_xDecode.plotLine_xDdecode_avg(
    xDecode_hmrds_vs_ards_df, permuteXDecode_all_df, rds_train_test, save_flag
)


# %% xDecode all comparisons
decode_all_df = pd.read_pickle("../../../../Data/MVPA/mvpa_decode_shift_zscore_vtc.pkl")
plot_mvpa_xDecode.plotBox_xDecode_at_nVox_all_comparison(
    decode_all_df,
    xDecode_crds_vs_ards_df,
    xDecode_crds_vs_hmrds_df,
    permuteXDecode_all_df,
    nVox_to_analyze,
    save_flag,
)
