# %%
from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct
import numpy as np

# %% prepare GC-net
# network parameters
params_network = {
    "h_bg": 256,
    "w_bg": 512,
    "maxdisp": 192,
}

# training parameters
sceneflow_type = "monkaa"  # dataset on which GC-Net was trained
epoch_to_load = 7
iter_to_load = 22201  # 22601  #
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
n_bootstrap = 1000
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 64,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 1,  # 1: use pedestal to ensure rds disparity > 0
    "n_bootstrap": n_bootstrap,
}

rdsl = RDS_LayerAct(params_network, params_train, params_rds)

# %% compute layer activation for a specific dot density and dot match
dotDens = 0.25
dotMatch = 0.0
rdsl.compute_layer_act_rds(dotMatch, dotDens, rdsl.background_flag)

# compute layer activation for all dot density
rdsl.compute_layer_act_rds_all(rdsl.background_flag)

# %% single bootstrap
split_train = 0.75
# for dotDens in rdsl.dotDens_list:
dotDens = 0.9
rdsl.xDecode(dotDens, split_train, rdsl.n_bootstrap)

# %% plot
save_flag = 1

# all dot density in one image
rdsl.plotLine_xDecode_across_layers(save_flag)

# each dot density
for dotDens in rdsl.dotDens_list:
    rdsl.plotLine_xDecode_across_layers_at_dotDens(dotDens, save_flag)

rdsl.plotHeat_xDecode(save_flag)

# %%
