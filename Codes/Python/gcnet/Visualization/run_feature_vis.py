# %% load necessary modules

from Network_dissection.network_dissection import NetworkDissection
from utils.utils import *
from timeit import default_timer as timer
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch

# import sys
import os

# from Optimization.loss import NeuronActivation
from Visualization.network_visualization import NetworkVisualization

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


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
iter_to_load = 22601  # 22201
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

# %% optimize single neuron
feature_channel_index = 4
disp_channel_index = 40  # disparity channel index
neuron_row_index = 64  # neuron coordinate
neuron_col_index = 128  # neuron coordinate
# target_layer = list(nd.model.layer19[0]) + list(nd.model.layer20[0])
target_layer = nd.model.layer20[0]
n_iters = 128
img_left_optimized, img_right_optimized, loss_history = nd.optimize_neuron_activation(
    target_layer,
    feature_channel_index,
    disp_channel_index,
    neuron_row_index,
    neuron_col_index,
    n_iters,
)

# visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_optimized.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_optimized.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_history.cpu().numpy())
axes[1, 0].set_title("Loss")


# %%
for i in range(1, len(nd.target_list)):
    target_layer_name = nd.target_name[i]
    target_layer = nd.target_list[i]
    print(target_layer_name)
    (
        img_left_layer,
        img_right_layer,
        loss_history,
    ) = nd.optimize_all_neurons_in_single_layer(target_layer)

    # save
    np.save(f"{nd.neuron_dir}/img_left_{target_layer_name}_conv.npy", img_left_layer)
    np.save(f"{nd.neuron_dir}/img_right_{target_layer_name}_conv.npy", img_right_layer)
    np.save(f"{nd.neuron_dir}/loss_history_{target_layer_name}_conv.npy", loss_history)

# %% visualize neuron activation
target_layer_name = "layer19"
img_left_layer = np.load(f"{nd.neuron_dir}/img_left_{target_layer_name}_conv.npy")
img_right_layer = np.load(f"{nd.neuron_dir}/img_right_{target_layer_name}_conv.npy")

n_img_row = 1
nv.visualize_neuron_activation_in_single_disp_channel(
    img_left_layer, img_right_layer, target_layer_name, n_img_row
)
rgb_flag = False
nv.visualize_neuron_activation_in_single_disp_channel(
    img_left_layer, img_right_layer, target_layer_name, n_img_row, rgb_flag
)

# %% visualize neuropn activation max, near-zero, and min interocular correlation
save_flag = 1
nv.visualize_neuron_activation_mzm_interocular_corr(save_flag)

# %% Optimize single channel activation
target_layer = nd.model.layer35a
feature_channel_index = 2
disp_channel_index = 5  # disparity channel index
n_iters = 128
img_left_optimized, img_right_optimized, loss_history = nd.optimize_channel_activation(
    target_layer, feature_channel_index, disp_channel_index, n_iters, 0.05
)

# %% visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_optimized.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_optimized.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_history.cpu().numpy())
axes[1, 0].set_title("Loss")

# %% Optimize single channel activation all
target_layer = model.layer37
target_layer_name = "layer37"
n_iters = 128
(
    img_left_optimized,
    img_right_optimized,
    loss_history,
) = nd.optimize_channel_activation_all(target_layer, target_layer_name, n_iters)

# save
channel_dir = nd.network_diss_dir + "/channel_optimization"
if not os.path.exists(channel_dir):
    os.mkdir(channel_dir)

np.save(f"{channel_dir}/img_left_channel_{target_layer_name}.npy", img_left_optimized)
np.save(f"{channel_dir}/img_right_channel_{target_layer_name}.npy", img_right_optimized)
np.save(f"{channel_dir}/loss_channel_{target_layer_name}.npy", loss_history)
############################################################

# %% visualize channel activation
target_layer_name = "layer19"
img_left_layer = np.load(f"{channel_dir}/img_left_channel_{target_layer_name}.npy")
img_right_layer = np.load(f"{channel_dir}/img_right_channel_{target_layer_name}.npy")

n_img_row = 16
# plot rgb
nv.visualize_channel_activation_in_single_disp_channel(
    img_left_layer, img_right_layer, target_layer_name, n_img_row
)

# plot heatmap
rgb_flag = False
nv.visualize_channel_activation_in_single_disp_channel(
    img_left_layer, img_right_layer, target_layer_name, n_img_row, rgb_flag
)

# %% Optimize single feature channel activation
target_layer = model.layer35a
feature_channel_index = 10
n_iters = 128
(
    img_left_optimized,
    img_right_optimized,
    loss_history,
) = nd.optimize_feature_channel_activation(
    target_layer,
    feature_channel_index,
    n_iters,
)

# %% visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_optimized.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_optimized.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_history.cpu().numpy())
axes[1, 0].set_title("Loss")


# %% optimize all feature channels in all layers
n_iters = 128
feature_channel_dir = nd.network_diss_dir + "/feature_channel_activation"
if not os.path.exists(feature_channel_dir):
    os.mkdir(feature_channel_dir)

for t in range(len(target_name)):
    target_layer = target_list[t]
    target_layer_name = target_name[t]
    print(f"compute feature channel activation in {target_layer_name}")

    (
        img_left_layer,
        img_right_layer,
        loss_history_layer,
    ) = nd.optimize_feature_channel_activation_all(
        target_layer, target_layer_name, n_iters
    )

    np.save(
        f"{feature_channel_dir}/img_left_feature_channel_{target_name[t]}.npy",
        img_left_layer,
    )
    np.save(
        f"{feature_channel_dir}/img_right_feature_channel_{target_name[t]}.npy",
        img_right_layer,
    )
    np.save(
        f"{feature_channel_dir}/loss_feature_channel_{target_name[t]}.npy",
        loss_history_layer,
    )

# %%
for t in range(len(target_name)):
    # t = 11
    target_layer_name = target_name[t]
    img_left_layer = np.load(
        f"{feature_channel_dir}/img_left_feature_channel_{target_layer_name}.npy"
    )
    img_right_layer = np.load(
        f"{feature_channel_dir}/img_right_feature_channel_{target_layer_name}.npy"
    )
    n_img_row = 8
    # plot rgb
    nv.visualize_feature_channel_activation(
        img_left_layer, img_right_layer, target_layer_name, n_img_row
    )

    # plot heatmap
    rgb_flag = False
    nv.visualize_feature_channel_activation(
        img_left_layer, img_right_layer, target_layer_name, n_img_row, rgb_flag
    )

# %% Optimize single disparity channel activation
target_layer = model.layer37
disp_channel_index = 5
n_iters = 128
(
    img_left_optimized,
    img_right_optimized,
    loss_history,
) = nd.optimize_disparity_channel_activation(
    target_layer,
    disp_channel_index,
    n_iters,
)

# %% visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_optimized.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_optimized.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_history.cpu().numpy())
axes[1, 0].set_title("Loss")

# %% optimize all disparity channel activation
target_layer = model.layer37
n_iters = 128
(
    img_left_layer,
    img_right_layer,
    loss_history_layer,
) = nd.optimize_disparity_channel_activation_all(target_layer, n_iters)

np.save("img_left_disp_channel_layer37.npy", img_left_layer)
np.save("img_right_disp_channel_layer37.npy", img_right_layer)
np.save("loss_disp_channel_layer37.npy", loss_history_layer)

# %%

for t in range(len(target_list)):
    print(f"optimize layer: {target_name[t]}")
    (
        img_left_layer,
        img_right_layer,
        loss_history_layer,
    ) = nd.optimize_disparity_channel_activation_all(target_list[t], n_iters)

    np.save(f"img_left_disp_channel_{target_name[t]}.npy", img_left_layer)
    np.save(f"img_right_disp_channel_{target_name[t]}.npy", img_right_layer)
    np.save(f"loss_disp_channel_{target_name[t]}.npy", loss_history_layer)


# %%
target_layer_name = "layer37"  # target_name[t]
img_left_layer = np.load(f"img_left_disp_channel_{target_layer_name}.npy")
img_right_layer = np.load(f"img_right_disp_channel_{target_layer_name}.npy")
n_img_row = 32
# plot rgb
nv.visualize_disp_channel_activation(
    img_left_layer, img_right_layer, target_layer_name, n_img_row
)

# plot heatmap
rgb_flag = False
nv.visualize_disp_channel_activation(
    img_left_layer, img_right_layer, target_layer_name, n_img_row, rgb_flag
)


# %% Optimize layer activation (deep dream)
target_layer = model.layer37
n_iters = 128
(
    img_left_optimized,
    img_right_optimized,
    loss_history,
) = nd.optimize_layer_activation(
    target_layer,
    n_iters,
)

# %% deep dream all layers
n_iters = 128
img_left_layer, img_right_layer, loss_history_layer = nd.optimize_layer_activation_all(
    n_iters
)

# save file
np.save(f"{nd.layer_dir}/img_left_deep_dream.npy", img_left_layer)
np.save(f"{nd.layer_dir}/img_right_deep_dream.npy", img_right_layer)
np.save(f"{nd.layer_dir}/loss_deep_dream.npy", loss_history_layer)
# %% visualize deep dream

img_left_layer = np.load(f"{nd.layer_dir}/img_left_deep_dream.npy")
img_right_layer = np.load(f"{nd.layer_dir}/img_right_deep_dream.npy")
n_img_row = 6

# plot rgb
nv.visualize_deep_dream_activation(img_left_layer, img_right_layer, n_img_row)

# plot heatmap
rgb_flag = False
nv.visualize_deep_dream_activation(img_left_layer, img_right_layer, n_img_row, rgb_flag)
# %%
