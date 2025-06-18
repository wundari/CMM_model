# %%
# working dir:
#     Codes/Python/CMM

# %%
import numpy as np

import Signal2Noise.Signal_to_Noise as S2N
from Signal2Noise.Plot_Signal_to_Noise import Plot_Signal2Noise

# %%
sawtooth_noise_std_list = [0.01, 0.05, 0.15]
s2n = S2N.Signal2Noise(sawtooth_noise_std_list)
# %%
# read beta and t_stat data
beta_all_sbjID = np.load("../../../Data/MVPA/beta_all_sbjID.npy", allow_pickle=True)
t_stat_all_sbjID = np.load("../../../Data/MVPA/t_stat_all_sbjID.npy", allow_pickle=True)

# %%
nVox_to_analyze = 250
# s2n_all_sbj = s2n.compute_s2n_fmri_all_sbj(t_stat_all_sbjID, nVox_to_analyze)
# signalchange_all_sbj = s2n.compute_signalchange_fmri_all_sbj(
#     t_stat_all_sbjID, nVox_to_analyze
# )

# save data
# np.save("../../../Data/S2N/s2n_fmri_all_sbj.npy", s2n_all_sbj)
# np.save("../../../Data/S2N/signalchange_fmri_all_sbj.npy", signalchange_all_sbj)

# %% plot
# load s2n_all_sbj
s2n_all_sbj = np.load("../../../Data/S2N/s2n_fmri_all_sbj.npy")
signalchange_all_sbj = np.load("../../../Data/S2N/signalchange_fmri_all_sbj.npy")
plot_s2n = Plot_Signal2Noise()
save_flag = 0
nVox_to_analyze = 250

plot_s2n.plotBar_s2n_fmri(s2n_all_sbj, nVox_to_analyze, save_flag)

# %% plot box
plot_s2n.plotBox_s2n_fmri_at_nVox(s2n_all_sbj, nVox_to_analyze, save_flag)
plot_s2n.plotBox_signalchange_fmri_at_nVox(
    signalchange_all_sbj, nVox_to_analyze, save_flag
) # Supplementary Figure 3
