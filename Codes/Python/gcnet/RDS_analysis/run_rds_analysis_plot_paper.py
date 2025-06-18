# %%
# Script for ploting Figure 4b right panel
# working dir: Codes/Python/gcnet
#%%

from RDS_analysis.rds_analysis import RDSAnalysis
import numpy as np

# %%
# network parameters
params_network = {
    "h_bg": 256,
    "w_bg": 512,
    "maxdisp": 192,
}

# training parameters
sceneflow_type = "monkaa"  # dataset on which GC-Net was trained
epoch_to_load = 7
iter_to_load = 22601
target_disp = 10
target_dotDens = 0.25
c_disp_shift = 1.5
flip_input = 0  # set to 1 if flip the input (right2left), use right disparity image as ground truth
batch_size = 2

params_train = {
    "sceneflow_type": sceneflow_type,
    "learning_rate": 6e-4,
    "eval_iter": 25,
    "eval_interval": 200,
    "batch_size": batch_size,
    "c_disp_shift": c_disp_shift,
    "n_epoch": 10,
    "load_state": True,
    "epoch_to_load": epoch_to_load,
    "iter_to_load": iter_to_load,
    "train_or_eval_mode": "eval",
    "flip_input": flip_input,
    "compile_mode": "default",  # "reduce-overhead",
}

# rds parameters
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 64,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": np.round(0.1 * np.arange(1, 10), 1),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
}

rdsa = RDSAnalysis(params_network, params_train, params_rds)

# %% plot cross-decoding performance
# plot performance at a target dot density
save_flag = 0 # 1: save fig, 0: don't save fig
dotDens = 0.3 # target dot density
rdsa.plotLine_xDecode_at_dotDens(dotDens, save_flag)

# %%
