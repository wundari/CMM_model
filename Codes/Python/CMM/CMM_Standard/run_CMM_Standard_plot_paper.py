"""
script for plotting Figure 3 in papers
# working dir:
#     Codes/Python/CMM
"""

# %% load necessary modules
import CMM_Standard.CMM_Standard as CMM
from DisparityColumn.DisparityColumn import DisparityColumn as DispCol

import numpy as np

# %%
# define rds object
# the rds only has 2 disparities: crossed and uncrossed,
# [n_trial, crossed_uncrossed, size_rds, size_rds] = [n_trial, 2, size_rds, size_rds]
n_trial_total = 510  # the number of rds images (trial) used per bootstrap
n_trial_batch = 17  # n_trial_batch, has to be multiple of 8
n_trial_epoch = np.int32(n_trial_total / n_trial_batch)

n_bootstrap = 1000  # 5

rDot = 0.045
dotDens = 0.5
size_rds_bg_deg = 2.5  # 120 pix
size_rds_ct_deg = 1.25
deg_per_pix = 0.02
n_rds = 10240  # ori
rds = CMM.RDS(n_rds, rDot, dotDens, size_rds_bg_deg, size_rds_ct_deg, deg_per_pix)

# load rds
rds_type_list = ["ards", "hmrds", "crds"]
rds_type = "ards"
rds.load_rds(rds_type)
# resize rds
new_dim = (96, 96)
rds.resize_rds(new_dim)

# generate disparity column distribution
nVox_to_analyze = 250  # the number of voxels used for the analysis
n_bootstrap_dispCol = n_bootstrap  # similar to scan run (maybe)
noise_dispCol_sigma_list = np.array(
    [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32
)
sawtooth_noise_std = noise_dispCol_sigma_list[2]
dispCol = DispCol(sawtooth_noise_std, n_bootstrap_dispCol, nVox_to_analyze)
# generate disparity column map
# [n_bootstrap, nVox, neurons_per_vox]
dispColMap_bootstrap = dispCol.create_dispColMap_vox_bootstrap()

# define rf object
f_batch = np.array([1.0, 2.0, 4.0, 8.0, 16.0]).astype(np.float32)

# instantiate CMM
mtd = "euclidean"  # measured distance method for computing RDM
cmm = CMM.Simulate_CMM(
    rds,
    f_batch,
    dispColMap_bootstrap,
    nVox_to_analyze,
    n_trial_epoch,
    n_trial_batch,
    rds_type,
    mtd,
)
# %%
w_cmm_best, loss_min, id_best = cmm.compute_w_cmm_best(noise_dispCol_sigma_list)

# %%
plot_cmm = CMM.Plot_CMM(mtd)
save_flag = 0  # 0: not save, 1: save

# plot scatter the weight
plot_cmm.plotScatter_w_corr_vs_w_match(w_cmm_best, save_flag)

# plot box the weight ratio
plot_cmm.plotBox_w_cmm_ratio(w_cmm_best, save_flag)

plot_cmm.plotBar_goodness_of_fit(
    noise_dispCol_sigma_list, w_cmm_best, id_best, save_flag
)

# plot rdm_fmri and the rdm_fitted
plot_cmm.plotHeat_rdm_cmm_fit(noise_dispCol_sigma_list, w_cmm_best, id_best, save_flag)

# %%
