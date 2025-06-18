#%%

# Script for plotting Supplemtary Figure 1
# working dir:
#     Codes/Python/CMM

# %%
import numpy as np
from GLM.glm_v2 import GLM as GLM
from MVPA.PlotMVPA_Decode import PlotMVPA_Decode

# %% plotBox, Supplementary Figure 1
plot_mvpa_decode = PlotMVPA_Decode()

# read data
decode_score_allSbj = np.load("../../../Data/MVPA/decode_vox_fixed_percentage.npy", allow_pickle=True)
decode_permute_allSbj_df = np.load("../../../Data/MVPA/decode_vox_fixed_percentage_permute_1000.pkl", allow_pickle=True)

nVox_percentage_list = np.array([0.1, 0.2, 0.5, 0.75, 1], dtype=np.float32)

save_flag = 0
for voxPercent_idx in range(len(nVox_percentage_list)):
    plot_mvpa_decode.plotBox_decode_at_voxPercent(
        decode_score_allSbj,
        decode_permute_allSbj_df,
        nVox_percentage_list,
        voxPercent_idx,
        save_flag,
    )