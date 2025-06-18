"""
Script for making Figure 2 and Supplementary Fgure 2 in the paper
# working dir:
#     Codes/Python/CMM
"""

# %%
import numpy as np
import pandas as pd
from MVPA.PlotMVPA_Decode import PlotMVPA_Decode

# %% read data
n_permute = 10000
decode_all_df = pd.read_pickle("../../../Data/MVPA/mvpa_decode_shift_zscore_vtc.pkl")
permuteDecode_all_df = pd.read_pickle(
    "../../../Data/MVPA/mvpa_decode_shift_zscore_vtc_permute_{}.pkl".format(n_permute)
)

# %% statistical test
from MVPA.MVPA_Decode import MVPA_Decode

n_bootstrap = 10000
nVox_to_analyze = 250
alpha = 0.05

beta_all_sbjID = np.load("../../../Data/MVPA/beta_all_sbjID.npy", allow_pickle=True)
t_stat_all_sbjID = np.load("../../../Data/MVPA/t_stat_all_sbjID.npy", allow_pickle=True)
mvpa_decode = MVPA_Decode(beta_all_sbjID, t_stat_all_sbjID)

stat_at_vox_df = mvpa_decode.compute_stat_decode_permute_bootstrap(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, n_bootstrap, alpha
)
# save to excel
# stat_at_vox_df.to_excel(
#     "../../../Data/MVPA/stat_decode_shift_zscore_vtc_{}.xlsx".format(nVox_to_analyze)
# )

# %% plotting
plot_mvpa_decode = PlotMVPA_Decode()
nVox_to_analyze = 250
save_flag = 0  # 0: not save, 1: save

# %% plotBox, Figure 2
plot_mvpa_decode.plotBox_decode_at_nVox(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
)

## plotLine decoding, Supplementary Figure 2
decode_all_df = pd.read_pickle("../../../Data/MVPA/mvpa_decode_shift_zscore_vtc_vox25_1000.pkl")
plot_mvpa_decode.plotLine_decode_avg(decode_all_df, permuteDecode_all_df, save_flag)

