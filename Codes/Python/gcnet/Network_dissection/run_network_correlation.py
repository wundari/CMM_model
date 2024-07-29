# %%
# %% load necessary modules
from Common.network_correlation import gauss2d
from Network_dissection.network_dissection import NetworkDissection
from Visualization.network_visualization import NetworkVisualization

import numpy as np
import matplotlib.pyplot as plt

# %% compute correlation between left and right preferred inputs
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
c_disp_shift = 1.5
flip_input = 0  # set to 1 if flip the input (right2left), use right disparity image as ground truth
batch_size = 2
params_train = {
    "sceneflow_type": sceneflow_type,
    "c_disp_shift": c_disp_shift,
    "epoch_to_load": epoch_to_load,
    "iter_to_load": iter_to_load,
    "flip_input": flip_input,
    "target_disp": target_disp,
    "batch_size": batch_size,
}
nd = NetworkDissection(params_network, params_train)
nv = NetworkVisualization(params_train)

# %% set up filter parameter
filter_mean = 1
filter_size = 11

# %% compute binocular correlation of preferred inputs
# for center neuron in each feature and disparity channel
nd.neuron_pref_input_corr(filter_mean, filter_size)


#%% plot binocular correlation
save_flag = 0
nv.plotBox_neuron_pref_input_interocular_corr(save_flag)

# %% compute binocular correlation of preferred inputs
# for each feature and disparity channel
nd.channel_correlation(filter_mean, filter_size)

# plot binocular correlation
nv.plotBox_channel_correlation(save_flag)

# %% layer
filt = gauss2d(filter_mean, filter_size)
img_left_layers = np.load(f"{nd.layer_dir}/img_left_deep_dream.npy")
img_right_layers = np.load(f"{nd.layer_dir}/img_right_deep_dream.npy")
# convert to grayscale
img_left_layer = img_left_layers.mean(axis=-1)
img_right_layer = img_right_layers.mean(axis=-1)

corr_layer = nd.layer_correlation(filter_mean, filter_size)

plt.plot(corr_layer)

# %%
