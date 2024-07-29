
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import torch._dynamo

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

from typing import Optional, assert_type
import numpy as np
from tqdm import tqdm
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("captum")
import captum.optim as optimviz

from data_handler.DataHandler_SceneFlow_v2 import *
from RDS.DataHandler_RDS import *
from engine.model import *
from Optimization.loss import NeuronActivation
from utils.utils import *
from utils.output_hook import ModuleOutputsHook
from utils.typing import LossFunction

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

    # print out seed number for each worker
    # np_seed = np.random.get_state()[1][0]
    # py_seed = random.getstate()[1][0]

    # print(f"{worker_id} seed pytorch: {worker_seed}\n")
    # print(f"{worker_id} seed numpy: {np_seed}\n")
    # print(f"{worker_id} seed python: {py_seed}\n")


g = torch.Generator()
g.manual_seed(seed_number)
# %%
    def optimize_feature_channel_activation(
        self,
        target: nn.Module,
        feature_channel_index: int,
        disp_channel_index: int,
        neuron_row_index: int,
        neuron_col_index: int,
        batch_index: Optional[int] = None,
        n_iters: int = 100,
        lr: float = 0.025,
    ):
        # define loss function
        loss_fn = FeatureChannelActivation(
            target,
            feature_channel_index,
            disp_channel_index
        )

        # define noise image for optimization
        input_param_left = optimviz.images.NaturalImage((self.h_bg, self.w_bg))
        input_param_right = optimviz.images.NaturalImage((self.h_bg, self.w_bg))

        # move to gpu
        self.input_param_left = input_param_left.to(device)
        self.input_param_right = input_param_right.to(device)

        # define optimizer
        # lr = 0.025
        # optimizer = torch.optim.Adam(obj.parameters(), lr=lr)
        params = list(self.input_param_left.parameters()) + list(
            self.input_param_right.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        # start optimizing
        # n_iters = 128
        loss_history = self.optimize(loss_fn, optimizer, n_iters)

        # fetch the optimized inputs and swap their axis to [h, w, rgb_channels]
        img_left_optimized = self.input_param_left()[0].permute(1, 2, 0).detach()
        img_right_optimized = self.input_param_right()[0].permute(1, 2, 0).detach()

        return img_left_optimized, img_right_optimized, loss_history

    def optimize(
        self,
        loss_fn: LossFunction,
        optimizer: Optional[torch.optim.Optimizer],
        n_iters: int,
    ):
        # define hook
        if isinstance(loss_fn.target, list):
            hook = ModuleOutputsHook(loss_fn.target)
        else:
            hook = ModuleOutputsHook([loss_fn.target])

        pbar = tqdm(total=n_iters, unit="step")
        history = []
        step = 0
        try:
            while step < n_iters:
                optimizer.zero_grad()

                # get input images
                image_left_t = self.input_param_left()
                image_right_t = self.input_param_right()
                # image_left_t = nd.input_param_left()
                # image_right_t = nd.input_param_right()

                # transform input
                image_left_t = self.transform(image_left_t)
                image_right_t = self.transform(image_right_t)
                # image_left_t = nd.transform(image_left_t)
                # image_right_t = nd.transform(image_right_t)

                # compute model's output.
                logits = self.model(image_left_t, image_right_t)
                # logits = nd.model(image_left_t, image_right_t)

                # consume_outputs return the captured values and resets the hook's state
                # compute module output
                module_outputs = hook.consume_outputs()
                # activations = module_outputs[loss_fn.target]

                # compute loss value.
                # essentially this is the activation of the target neuron
                loss_value = self.default_loss_summarize(loss_fn(module_outputs))
                # loss_value = self.default_loss_summarize(
                #     activations[
                #         0,  # batch index
                #         channel_index,
                #         disp_channel_index,
                #         neuron_row_idx,
                #         neuron_col_idx,
                #     ]
                # )
                history.append(loss_value.clone().detach())

                # update parameters
                loss_value.backward()
                optimizer.step()

                # update progress bar
                pbar.set_postfix(
                    {"Objective": f"{history[-1].mean():.1f}"}, refresh=False
                )

                if step < n_iters:
                    pbar.update()
                else:
                    pbar.close()

                step += 1
        finally:
            hook.remove_hooks()

        return torch.stack(history)

    def optimize_all_neurons_in_single_layer(
        self,
        target: nn.Module,
        n_iters: int = 100,
    ):
        # get layer activation to get layer dimension
        # define hook
        if isinstance(target, list):
            hook = ModuleOutputsHook(target)
        else:
            hook = ModuleOutputsHook([target])
            # hook = ModuleOutputsHook([model.layer36a])

        # define noise image for optimization
        image_left_t = optimviz.images.NaturalImage((self.h_bg, self.w_bg)).to(device)
        image_right_t = optimviz.images.NaturalImage((self.h_bg, self.w_bg)).to(device)
        # image_left_t = optimviz.images.NaturalImage((nd.h_bg, nd.w_bg)).to(device)
        # image_right_t = optimviz.images.NaturalImage((nd.h_bg, nd.w_bg)).to(device)

        # compute model's output.
        logits = self.model(image_left_t(), image_right_t())
        # logits = nd.model(image_left_t(), image_right_t())

        # consume_outputs return the captured values and resets the hook's state
        # compute module output
        module_outputs = hook.consume_outputs()
        activations = module_outputs[target]
        # activations = module_outputs[model.layer36a]

        hook.remove_hooks()

        # get dimension
        _, n_feature_channels, n_disp_channels, h, w = activations.shape
        # coordinate of target neurons
        neuron_coordinates = [h // 2, w // 2]
        img_left_layer = torch.zeros(
            (
                n_feature_channels,
                n_disp_channels,
                self.h_bg,
                self.w_bg,
                3,
            ),
            dtype=torch.float32,
        )
        img_right_layer = torch.zeros(
            (
                n_feature_channels,
                n_disp_channels,
                self.h_bg,
                self.w_bg,
                3,
            ),
            dtype=torch.float32,
        )
        loss_history_layer = torch.zeros(
            (
                n_feature_channels,
                n_disp_channels,
                n_iters,
            ),
            dtype=torch.float32,
        )

        for c in range(n_feature_channels):
            for d in range(n_disp_channels):
                neuron_row_index = neuron_coordinates[0]
                neuron_col_index = neuron_coordinates[1]
                print(
                    f"optimize neuron activation at "
                    + f"feature_channel_idx: {c}/{n_feature_channels}, "
                    + f"disp_channel_idx: {d}/{n_disp_channels}, "
                    + f"neuron_coordinate: ({neuron_row_index}, {neuron_col_index})"
                )
                (
                    img_left_optimized,
                    img_right_optimized,
                    loss,
                ) = self.optimize_neuron_activation(
                    target,
                    c,  # feature_channel_index,
                    d,  # disp_channel_index,
                    neuron_row_index,  # neuron_row_index
                    neuron_col_index,  # neuron_col_index
                    n_iters,
                )

                img_left_layer[c, d] = img_left_optimized
                img_right_layer[c, d] = img_right_optimized
                loss_history_layer[c, d] = loss

        return img_left_layer, img_right_layer, loss_history_layer