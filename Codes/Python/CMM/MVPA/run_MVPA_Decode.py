"""
working dir: bw18_005_2/Codes/Python/CMM/
Created Date: 2022-06-20 22:57:31
Author: Bayu G. Wundari
-----
Last Modified: 2022-06-29 12:01:36
Modified By: Bayu G. Wundari


-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
decoding analysis
"""

# %%
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from GLM.GLM_v2 import GLM as GLM
from MVPA.MVPA_Decode import MVPA_Decode
from MVPA.PlotMVPA_Decode import PlotMVPA_Decode


# %%
glm = GLM()

## compute beta glm
mtd_normalization = 2  # percent signal change with respect to average across timepoints
beta_all_sbjID, beta_avg_all_sbjID = glm.compute_glm_all_sbjID(mtd_normalization)

## compute t_stat
t_stat_all_sbjID = glm.compute_t_stat_all_sbjID(beta_all_sbjID, mtd_normalization)

# create shift and normalize vtc
# nVox_list = np.arange(25, 401, 25)
# for v in range(len(nVox_list)):
# nVox_to_analyze = nVox_list[v]
# glm.shift_normalize_save_vtc_all_sbjID(t_stat_all_sbjID, nVox_to_analyze)

## start mvpa
mvpa_decode = MVPA_Decode(beta_all_sbjID, t_stat_all_sbjID)

## create permute_df for many nVox
n_permute = 10000
nVox_list = np.arange(25, 325, 25)
# nVox_list = [250]

decode_all_list = []
permute_all_list = []
for c in range(len(mvpa_decode.comp_pair_all)):

    comp_pair = np.array(mvpa_decode.comp_pair_all[c]).astype(np.int32)

    print(
        "generate decode_df for comp_pair: %s VS %s"
        % (mvpa_decode.conds[comp_pair[0]], mvpa_decode.conds[comp_pair[1]])
    )

    t_start = timer()
    decode_df = mvpa_decode.decode_allSbj(comp_pair, nVox_list)
    decode_all_list.append(decode_df.copy())
    t_end = timer()
    print(t_end - t_start)

    print(
        "generate permute_df for comp_pair: %s VS %s"
        % (mvpa_decode.conds[comp_pair[0]], mvpa_decode.conds[comp_pair[1]])
    )

    t_start = timer()
    permute_df = mvpa_decode.permuteDecode_allSbj(comp_pair, nVox_list, n_permute)
    t_end = timer()
    print(t_end - t_start)

    permute_all_list.append(permute_df.copy())


# concatenate dataframe
decode_all_df = pd.concat(decode_all_list, ignore_index=True)
permuteDecode_all_df = pd.concat(permute_all_list, ignore_index=True)

## save data
# decode_all_df.to_pickle("../../../../Data/MVPA/mvpa_decode_shift_zscore_vtc.pkl")
# permuteDecode_all_df.to_pickle(
#     "../../../../Data/MVPA/mvpa_decode_shift_zscore_vtc_permute_{}.pkl"
#     .format(n_permute)
# )

# %% read data
n_permute = 10000
decode_all_df = pd.read_pickle("../../../Data/MVPA/mvpa_decode_shift_zscore_vtc.pkl")
permuteDecode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_decode_shift_zscore_vtc_permute_{}.pkl".format(n_permute)
)

#####################
# %% statistical testing

n_bootstrap = 10000
nVox_to_analyze = 250
alpha = 0.05
stat_at_vox_df = mvpa_decode.compute_stat_decode_permute_bootstrap(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, n_bootstrap, alpha
)
# save to excel
stat_at_vox_df.to_excel(
    "../../../Data/MVPA/stat_decode_shift_zscore_vtc_{}.xlsx".format(nVox_to_analyze)
)

stat_allVox = []
for v in range(len(nVox_list)):

    nVox_to_analyze = nVox_list[v]

    temp = mvpa_decode.compute_stat_decode_permute_bootstrap(
        decode_all_df, permuteDecode_all_df, nVox_to_analyze, n_bootstrap
    )

    stat_allVox.append(temp)

stat_allVox_df = pd.concat(stat_allVox)
stat_allVox_df.to_excel("../../../Data/MVPA/stat_decode_shift_zscore_vtc_allVox.xlsx")

##################
# %% plotting

plot_mvpa_decode = PlotMVPA_Decode()
nVox_to_analyze = 250
save_flag = 1

# %% plotBox
plot_mvpa_decode.plotBox_decode_at_nVox(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)
plot_mvpa_decode.plotBox_decode_at_nVox_v2(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)
plot_mvpa_decode.plotBox_decode_and_dprime_at_nVox(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)

# MT with_localizer vs without_localizer
plot_mvpa_decode.plotBox_decode_at_nVox_MT(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)

## plotLine decoding
plot_mvpa_decode.plotLine_decode_avg(decode_all_df, permuteDecode_all_df, save_flag)

## plotScatter ards vs hmrds
plot_mvpa_decode.plotScatter_decode_ards_vs_decode_hmrds_all_roi(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)

# plotScatter ards vs hmrds in dprime, each roi
plot_mvpa_decode.plotScatter_decode_ards_vs_decode_hmrds_dprime(
    decode_all_df, nVox_to_analyze, save_flag
)

# plotScatter ards vs hmrds in dprime combined all rois
plot_mvpa_decode.plotScatter_decode_ards_vs_decode_hmrds_dprime_all_roi(
    decode_all_df, nVox_to_analyze, save_flag
)

## plotScatter ards vs crds
plot_mvpa_decode.plotScatter_decode_ards_vs_decode_crds_all_roi(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)

# plotScatter ards vs crds in dprime
plot_mvpa_decode.plotScatter_decode_ards_vs_decode_crds_dprime_all_roi(
    decode_all_df, nVox_to_analyze, save_flag
)

# %%
plot_mvpa_decode.plotScatter_decode_ards_vs_decode_hmrds_MT(
    decode_all_df,
    permuteDecode_all_df,
    nVox_to_analyze,
    save_flag,
)
# %% analysis in MT with localizer
decode_MTloc_df = decode_all_df[
    (decode_all_df.roi_str == "MT")
    & (decode_all_df.sbjID.isin(plot_mvpa_decode.sbjID_with_MTlocalizer))
]
permuteDecode_MT_df = permuteDecode_all_df[(permuteDecode_all_df.roi_str == "MT")]
# obtain data for MT that does not use localizer
decode_MTnoloc_df = decode_all_df[
    (decode_all_df.roi_str == "MT")
    & ~(decode_all_df.sbjID.isin(plot_mvpa_decode.sbjID_with_MTlocalizer))
]

# averate decode_MTloc_df across fold_id for each sbjID
decode_MTloc_group_df = (
    decode_MTloc_df.groupby(["sbjID", "roi", "roi_str", "nVox", "comp_pair"])
    .acc.agg(["mean"])
    .reset_index()
)
decode_MTloc_group_df = decode_MTloc_group_df.rename(columns={"mean": "acc_cv_mean"})

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
