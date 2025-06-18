#%%

# Script for plotting Supplementary Figure 5
# working dir: Codes/Python/gcnet
# %% load necessary modules
from engine.engine_base import Engine
from wavelet.wavelet_analysis import WaveletAnalysis, WaveletParameters

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

seed_number = 1618  # 12321
torch.manual_seed(seed_number)
np.random.seed(seed_number)


# initialize random seed number for dataloader
def seed_worker(worker_id):
    worker_seed = seed_number  # torch.initial_seed()  % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed_number)

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% set up wavelet instance
wp = WaveletParameters()
wa = WaveletAnalysis(wp)

# %% set up train_loader

# network parameters
params_network = {
    "h_bg": 256,
    "w_bg": 512,
    "maxdisp": 192,
}

# training parameters
c_disp_shift = 0  # 1.5
flip_input = 0  # set to 1 if flip the input (right2left), use right disparity image as ground truth
params_train = {
    "sceneflow_type": "monkaa",
    "learning_rate": 6e-4,
    "eval_iter": 25,
    "eval_interval": 200,
    "batch_size": 8,
    "c_disp_shift": c_disp_shift,
    "n_epoch": 10,
    "load_state": False,
    "epoch_to_load": None,
    "iter_to_load": None,
    "train_or_eval_mode": "train",
    "flip_input": flip_input,
    "compile_mode": "reduce-overhead",
}

train_engine = Engine(params_network, params_train)
train_loader, val_loader = train_engine.prepare_dataset()


# %%create gabor filter kernels
gabor_real, gabor_imag = (
    wa.gabor()
)  # [n_freq_channels, n_theta_channels, kernel_size, kernel_size]

# %% create gabor filters

f = 1
t = 5
gab = gabor_real[f, t]

# check filter

fig, axis = plt.subplots(1, 2, figsize=(10, 5))
axis[0].imshow(gab, cmap="gray")
axis[1].plot(np.arange(wa.kernel_size), gab[5])

# %% compute wavelet power
# wavelet_power = wa.compute_wavelet_power(train_loader, gabor_real, gabor_imag)

# # save
# torch.save(
#     wavelet_power,
#     f"results/sceneflow/{params_train['sceneflow_type']}/wavelet_power_{params_train['sceneflow_type']}.pt",
# )

# %% plot wavelet power

# load pre-computed wavelet power
wavelet_power = torch.load(f"results/sceneflow/{params_train['sceneflow_type']}/wavelet_power_{params_train['sceneflow_type']}.pt")

# average across images
wavelet_power_avg = torch.mean(wavelet_power, dim=0)

save_flag = 0
wa.plotHeat_wavelet_power(wavelet_power_avg, params_train["sceneflow_type"], save_flag)
