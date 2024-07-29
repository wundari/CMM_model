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
decoding analysis for making figures in the paper
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
# %% plotting

plot_mvpa_decode = PlotMVPA_Decode()
nVox_to_analyze = 250
save_flag = 1  # 0: not save, 1: save

# %% plotBox

plot_mvpa_decode.plotBox_decode_at_nVox_v2(
    decode_all_df, permuteDecode_all_df, nVox_to_analyze, save_flag
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
stat_at_vox_df.to_excel(
    "../../../Data/MVPA/stat_decode_shift_zscore_vtc_{}.xlsx".format(nVox_to_analyze)
)
