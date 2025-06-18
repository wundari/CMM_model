# %%
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from GLM.glm_v2 import GLM as GLM

# from GLM_v2 import GLM
from MVPA.MVPA_Decode import MVPA_Decode
from MVPA.PlotMVPA_Decode import PlotMVPA_Decode


# %%
# glm = GLM()

## compute beta glm
# mtd_normalization = 2  # percent signal change with respect to average across timepoints
# beta_all_sbjID, beta_avg_all_sbjID = glm.compute_glm_all_sbjID(mtd_normalization)

# compute t_stat
# t_stat_all_sbjID = glm.compute_t_stat_all_sbjID(beta_all_sbjID, mtd_normalization)

# load beta and t values
beta_all_sbjID = np.load("../../../Data/MVPA/beta_all_sbjID.npy", allow_pickle=True)
t_stat_all_sbjID = np.load("../../../Data/MVPA/t_stat_all_sbjID.npy", allow_pickle=True)

# %% start mvpa
mvpa_decode = MVPA_Decode(beta_all_sbjID, t_stat_all_sbjID)

# %%
# get a list of max number of voxels in each roi for each participant
# [n_sbjID, n_ROIs]
nVox_max_all = mvpa_decode.count_nVox_max()
nVox_percentage_list = np.array([0.1, 0.2, 0.5, 0.75, 1], dtype=np.float32)

decode_score_allSbj = np.empty(
    (
        mvpa_decode.n_sbjID,
        mvpa_decode.n_ROIs,
        len(mvpa_decode.comp_pair_all),
        len(nVox_percentage_list),
    ),
    dtype=np.float32,
)
decode_permute_allSbj = []

n_permute = 1000
nVox_max = 3183
for sbj in range(mvpa_decode.n_sbjID):
    sbjID = mvpa_decode.sbjID_all[sbj]
    t_stat_sbj = mvpa_decode.t_stat_all_sbjID[sbj]

    # load vtc_norm, vtc data that has been shifted backward 2TR and z-scored
    vtc_norm = mvpa_decode.load_vtc_normalized(sbjID, nVox_max)

    for roi in range(mvpa_decode.n_ROIs):
        nVox_list = np.floor((nVox_percentage_list * nVox_max_all[sbj, roi])).astype(
            np.int32
        )

        for c in range(len(mvpa_decode.comp_pair_all)):

            comp_pair = np.array(mvpa_decode.comp_pair_all[c]).astype(np.int32)

            print(
                "generate decode_df for comp_pair: %s VS %s"
                % (mvpa_decode.conds[comp_pair[0]], mvpa_decode.conds[comp_pair[1]])
            )

            decode_score_allVox = mvpa_decode.decode_roi_allVox_percentage(
                sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_list, roi
            )

            decode_score_allSbj[sbj, roi, c] = decode_score_allVox.mean(axis=1)

            # permute decoding
            decode_permute = mvpa_decode.permuteDecode_roi_allVox_percentage(
                sbj, vtc_norm, t_stat_sbj, comp_pair, nVox_list, roi, n_permute
            )
            decode_permute_allSbj.append(decode_permute)

decode_score_allSbj.mean(axis=0)
# concetenate all df
decode_permute_allSbj_df = pd.concat(decode_permute_allSbj, ignore_index=True)

# add voxel percentage column
decode_permute_allSbj_df["voxPercent"] = 0
for sbj in range(mvpa_decode.n_sbjID):
    sbjID = mvpa_decode.sbjID_all[sbj]
    for roi in range(mvpa_decode.n_ROIs):
        nVox_sbj_roi = nVox_max_all[sbj, roi]
        voxPercentList = np.floor(nVox_sbj_roi * nVox_percentage_list).astype(np.int32)
        for v in range(len(voxPercentList)):
            decode_permute_allSbj_df.loc[
                (decode_permute_allSbj_df.sbjID == sbjID)
                & (decode_permute_allSbj_df.roi == roi)
                & (decode_permute_allSbj_df.nVox == voxPercentList[v]),
                "voxPercent",
            ] = nVox_percentage_list[v]

# save
# np.save("../../../Data/MVPA/decode_vox_fixed_percentage.npy", decode_score_allSbj)
# decode_permute_allSbj_df.to_pickle(
#     "../../../Data/MVPA/decode_vox_fixed_percentage_permute_{}.pkl".format(n_permute)
# )


# %% statistical testing
n_bootstrap = 10000
alpha = 0.05

stat_at_vox_df = mvpa_decode.compute_stat_decode_vox_fixed_percent_permute_bootstrap(
    decode_score_allSbj,
    decode_permute_allSbj_df,
    nVox_percentage_list,
    n_bootstrap,
    alpha,
)

# stat_at_vox_df.to_excel(
#     "../../../Data/MVPA/stat_decode_voxPercent_bootstrap{}.xlsx".format(n_bootstrap)
# )

# %% plot
plot_mvpa_decode = PlotMVPA_Decode()

# read data
n_permute = 10000
decode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_decode_shift_zscore_vtc_vox25_1000.pkl"
)
permuteDecode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_decode_shift_zscore_vtc_permute_{}.pkl".format(n_permute)
)

save_flag = 0
plot_mvpa_decode.plotLine_decode_avg(
    decode_all_df, permuteDecode_all_df, save_flag, alpha=0.05
)

# %% plotBox

for voxPercent_idx in range(len(nVox_percentage_list)):
    plot_mvpa_decode.plotBox_decode_at_voxPercent(
        decode_score_allSbj,
        decode_permute_allSbj_df,
        nVox_percentage_list,
        voxPercent_idx,
        save_flag,
    )

# %%
