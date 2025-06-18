# %%
# Script for plotting Supplementary Figure 3
# working dir:
#     Codes/Python/CMM

# %%
import numpy as np
from Signal2Noise.Plot_Signal_to_Noise import Plot_Signal2Noise

# %% plot box, Supplementary Figure 3
plot_s2n = Plot_Signal2Noise()
save_flag = 0 # 1: save figure; 0: not save
nVox_to_analyze = 250
signalchange_all_sbj = np.load("../../../Data/S2N/signalchange_fmri_all_sbj.npy")
plot_s2n.plotBox_signalchange_fmri_at_nVox(
    signalchange_all_sbj, nVox_to_analyze, save_flag
)
