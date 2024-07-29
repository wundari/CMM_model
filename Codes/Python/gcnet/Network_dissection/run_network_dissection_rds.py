# %% load necessary modules

from Network_dissection.network_dissection import NetworkDissection
from Visualization.network_visualization import NetworkVisualization
from RDS.DataHandler_RDS import *
from utils.output_hook import ModuleOutputsHook

from utils.utils import *
from timeit import default_timer as timer
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch._dynamo

import sys
import os

sys.path.append("captum")
# sys.path.insert(0, "captum")
import captum.optim as optimviz

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


# reproducibility
import random

seed_number = 3407  # 12321
torch.manual_seed(seed_number)
np.random.seed(seed_number)


# initialize random seed number for dataloader
def seed_worker(worker_id):
    worker_seed = seed_number  # torch.initial_seed()  % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed_number)


# %% prepare GC-net
sceneflow_type = "monkaa"  # dataset on which GC-Net was trained
epoch_to_load = 7
iter_to_load = 22201
disp_mag = 10
c_disp_shift = 1.5
nd = NetworkDissection(
    sceneflow_type, epoch_to_load, iter_to_load, disp_mag, c_disp_shift
)
nv = NetworkVisualization(
    sceneflow_type, epoch_to_load, iter_to_load, disp_mag, c_disp_shift
)

model = nd.model

target_name = [
    "layer19",
    "layer20",
    "layer21",
    "layer22",
    "layer23",
    "layer24",
    "layer25",
    "layer26",
    "layer27",
    "layer28",
    "layer29",
    "layer30",
    "layer31",
    "layer32",
    "layer33a",
    "layer34a",
    "layer35a",
    "layer36a",
    "layer37",
]

target_list = [
    model.layer19,
    model.layer20,
    model.layer21,
    model.layer22,
    model.layer23,
    model.layer24,
    model.layer25,
    model.layer26,
    model.layer27,
    model.layer28,
    model.layer29,
    model.layer30,
    model.layer31,
    model.layer32,
    model.layer33a,
    model.layer34a,
    model.layer35a,
    model.layer36a,
    model.layer37,
]

# %% set up rds
# rds params
n_rds_each_disp = 50  # 500 # n_rds for each disparity magnitude in disp_ct_pix
dotDens_list = [0.3]  # 0.1 * np.arange(1, 10)
rds_type = ["ards", "hmrds", "crds"]
# ards: 0, crds: 1, hmrds: 0.5, urds: -1
dotMatch_list = [0.0, 0.5, 1.0]
disp_mag = 10
disp_ct_pix_list = [disp_mag, -disp_mag]  # disparity magnitude (near, far)

# transform rds to tensor and in range [0, 1]
transform_data = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
)

dotMatch_ct = 0.0
dotDens = 0.3
background_flag = 1  # with cRDS background
rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
    dotMatch_ct, dotDens, disp_ct_pix_list, n_rds_each_disp, background_flag
)

rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
rds_loader = DataLoader(
    rds_data,
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g,
)

# %% compute activation in response to rds
inputs_left, inputs_right, disps = next(iter(rds_loader))
# move to gpu
input_left = inputs_left.pin_memory().to(device, non_blocking=True)
input_right = inputs_right.pin_memory().to(device, non_blocking=True)

# %% define hook
# target = [model.layer19[0], model.layer20[0]]
target = model.layer19
hook = ModuleOutputsHook(target)

# compute model's output
logits = model(input_left, input_right)

# consume_outputs return the captured values and resets the hook's state
# compute module output
module_outputs = hook.consume_outputs()
activations = module_outputs[target[0]]  # get conv2d output
hook.remove_hooks()

# %% plot distribution of neuron activations
feature_channel_index = 0
disp_channel_index = 0
neuron_activation = (
    activations[0, feature_channel_index, disp_channel_index].detach().cpu().numpy()
)

plt.hist(neuron_activation.ravel())

# %% find neurons that fire max and min

# max-activated neuron
neuron_coordinate = np.where(neuron_activation == neuron_activation.max())

# visualize max-activated neuron
img_left_max, img_right_max, loss_max = nd.optimize_neuron_activation(
    target,
    feature_channel_index,
    disp_channel_index,
    neuron_coordinate[0][0],
    neuron_coordinate[1][0],
)

# visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_max.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_max.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_max.cpu().numpy())
axes[1, 0].set_title("Loss")
# %% min-activated neuron
neuron_coordinate = np.where(neuron_activation == neuron_activation.min())

# visualize max-activated neuron
img_left_min, img_right_min, loss_min = nd.optimize_neuron_activation(
    target,
    feature_channel_index,
    disp_channel_index,
    neuron_coordinate[0][0],
    neuron_coordinate[1][0],
)

# visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_min.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_min.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_min.cpu().numpy())
axes[1, 0].set_title("Loss")

# %% which channel that strongly response to ards?
channel_activation = activations[0].mean(axis=(2, 3)).detach().cpu().numpy()
channel_coordinate = np.where(channel_activation == channel_activation.max())

# visualize max-activated channel
img_left_max, img_right_max, loss_max = nd.optimize_channel_activation(
    target,
    channel_coordinate[0][0],
    channel_coordinate[1][0],
)

# visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_max.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_max.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_max.cpu().numpy())
axes[1, 0].set_title("Loss")
# %%
channel_coordinate = np.where(channel_activation == channel_activation.min())

# visualize max-activated channel
img_left_min, img_right_min, loss_min = nd.optimize_channel_activation(
    target,
    channel_coordinate[0][0],
    channel_coordinate[1][0],
)

# visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_min.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_min.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_min.cpu().numpy())
axes[1, 0].set_title("Loss")
