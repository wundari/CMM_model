# %% load necessary modules
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
import torch.nn.functional as F

import os

import imageio.v2 as imageio
from python_pfm import *

# disparity shift
MAX_SHIFT = 0


# %% dataset
def generate_image_paths(train_list, val_list, flip_input=1):
    """
    set up image folders

    Args:
        train_list (list): a list containing dataset group used for training.
            ex: train_list = ["driving"]

        val_list (list): a list containing dataset group used for validation.
            ex: val_list = ["flying", "monkaa"]

        flip_input (int, optional): whether flip the input or not:
            0 -> no flip.
                Use left disparity image as the ground truth
            1 -> flip input images.
                Use right disparity image as the ground truth

            Defaults to 1.

    Returns:
        _type_: _description_
    """

    parent_folder = "/media/wundari/S990Pro2_4TB"
    ###############################
    ## generate training dataset ##
    ###############################
    dir_driving = f"{parent_folder}/Dataset/SceneFlow_complete/Driving/frames_cleanpass"
    dir_monkaa = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/frames_cleanpass"
    dir_flying = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/frames_cleanpass/TRAIN"
    dir_list = {"driving": dir_driving, "flying": dir_flying, "monkaa": dir_monkaa}

    paths = []
    for dir_name in train_list:
        dir = dir_list[dir_name]

        for root, dirs, files in os.walk(dir):
            for file in files:
                paths.append(os.path.join(root, file))

    file_train_left = []
    file_train_right = []
    for i in range(len(paths)):
        if flip_input:
            if paths[i].find("right") > -1:
                file_train_left.append(paths[i])
            elif paths[i].find("left") > -1:
                file_train_right.append(paths[i])
        else:
            if paths[i].find("left") > -1:
                file_train_left.append(paths[i])
            elif paths[i].find("right") > -1:
                file_train_right.append(paths[i])

    # load disparity images
    dir_driving = f"{parent_folder}/Dataset/SceneFlow_complete/Driving/disparity"
    dir_monkaa = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/disparity"
    dir_flying = (
        f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/disparity/TRAIN"
    )
    dir_list = {"driving": dir_driving, "flying": dir_flying, "monkaa": dir_monkaa}

    paths = []
    for dir_name in train_list:
        dir = dir_list[dir_name]

        for root, dirs, files in os.walk(dir):
            for file in files:
                paths.append(os.path.join(root, file))

    file_train_disp = []
    for i in range(len(paths)):
        if flip_input:  # use right disparity image as ground truth
            if paths[i].find("right") > -1:
                file_train_disp.append(paths[i])
        else:
            if paths[i].find("left") > -1:
                file_train_disp.append(paths[i])

    # sort file
    file_train_left.sort()
    file_train_right.sort()
    file_train_disp.sort()

    # %%
    #################################
    ## generate validation dataset ##
    #################################
    dir_driving = f"{parent_folder}/Dataset/SceneFlow_complete/Driving/frames_cleanpass"
    dir_monkaa = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/frames_cleanpass"
    # dir_flying = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/frames_cleanpass/TEST"
    dir_flying = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/frames_cleanpass/TRAIN"
    dir_list = {"driving": dir_driving, "flying": dir_flying, "monkaa": dir_monkaa}

    paths = []
    for dir_name in val_list:
        dir = dir_list[dir_name]

        for root, dirs, files in os.walk(dir):
            for file in files:
                paths.append(os.path.join(root, file))

    file_val_left = []
    file_val_right = []
    for i in range(len(paths)):
        if flip_input:
            # flip the inputs
            if paths[i].find("right") > -1:
                file_val_left.append(paths[i])
            elif paths[i].find("left") > -1:
                file_val_right.append(paths[i])
        else:
            if paths[i].find("left") > -1:
                file_val_left.append(paths[i])
            elif paths[i].find("right") > -1:
                file_val_right.append(paths[i])

    # load disparity images
    dir_driving = f"{parent_folder}/Dataset/SceneFlow_complete/Driving/disparity"
    dir_monkaa = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/disparity"
    # dir_flying = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/disparity/TEST"
    dir_flying = (
        f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/disparity/TRAIN"
    )
    dir_list = {"driving": dir_driving, "flying": dir_flying, "monkaa": dir_monkaa}

    paths = []
    for dir_name in val_list:
        dir = dir_list[dir_name]

        for root, dirs, files in os.walk(dir):
            for file in files:
                paths.append(os.path.join(root, file))

    file_val_disp = []
    for i in range(len(paths)):
        if flip_input:  # use right disparity image as ground truth
            if paths[i].find("right") > -1:
                file_val_disp.append(paths[i])

        else:
            if paths[i].find("left") > -1:
                file_val_disp.append(paths[i])

    # sort file
    file_val_left.sort()
    file_val_right.sort()
    file_val_disp.sort()

    return (
        file_train_left,
        file_train_right,
        file_train_disp,
        file_val_left,
        file_val_right,
        file_val_disp,
    )


# %% define patch dataset class
class DatasetTrain(Dataset):
    def __init__(
        self,
        file_train_left,
        file_train_right,
        file_train_disp,
        patch_size_h,
        patch_size_w,
        c_disp_shift,
        transform=None,
        flip_input=1,
    ):
        self.file_train_left = file_train_left
        self.file_train_right = file_train_right
        self.file_train_disp = file_train_disp
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.c_disp_shift = c_disp_shift
        self.transform = transform
        self.flip_input = flip_input

    def __len__(self):
        return len(self.file_train_disp)

    def __getitem__(self, idx):
        ## construct image patches

        # load image
        # get file name
        img_left_file = self.file_train_left[idx]
        img_right_file = self.file_train_right[idx]
        img_disp_file = self.file_train_disp[idx]

        # idx = 0
        # img_left_file = file_train_left[idx]
        # img_right_file = file_train_right[idx]
        # img_disp_file = file_train_disp[idx]

        ## read image
        # left image
        pic = imageio.imread(img_left_file) / 255.0
        img_left = np.array(
            pic.reshape(540, 960, 3), dtype=np.float32
        )  # [n_batch, h, w, rgb_channels]

        # right image
        pic = imageio.imread(img_right_file) / 255.0
        img_right = np.array(
            pic.reshape(540, 960, 3), dtype=np.float32
        )  # [n_batch, h, w, rgb_channels]

        ## load disparity image
        pic, scale = readPFM(img_disp_file)  # float32
        # shift disparity
        if self.c_disp_shift == 0:
            shift = 0  # np.random.randint(MAX_SHIFT)
            img_disp = np.array(np.clip(pic.reshape(540, 960) - shift, 0, 255)).astype(
                int
            )  # [n_batch, h, w]

            # generate random patch location
            row_start = np.random.randint(250)
            row_end = row_start + self.patch_size_h
            col_start = np.random.randint(400)  # for left disparity map
            # col_start = np.random.randint(150)  # for right disparity map
            col_end = col_start + self.patch_size_w

        else:
            shift = np.clip(self.c_disp_shift * np.median(pic), 0, 255).astype(int)
            # shift = np.clip(c_disp_shift * np.median(pic), 0, 255).astype(np.uint8)
            img_disp = np.array(
                np.clip(pic.reshape(540, 960) - shift, -128, 127)
            ).astype(
                np.int8
            )  # [n_batch, h, w]

            # generate random patch location
            row_start = np.random.randint(250)
            row_end = row_start + self.patch_size_h

            if self.flip_input:
                col_start = np.random.randint(150)  # for right disparity map
            else:
                col_start = np.random.randint(255, 400)  # for left disparity map
            col_end = col_start + self.patch_size_w

        ## create patch
        # left patch
        patch_left = img_left[row_start:row_end, col_start:col_end, :]

        if self.flip_input:
            ## shift the right image with respect to the right disparity map
            # right patch, shift to the right to compensate the shifted disparity map
            patch_right = img_right[
                row_start:row_end, col_start + shift : col_end + shift, :
            ]
        else:
            ## shift the right image with respect to the left disparity map
            # right patch, shift to the left to compensate the shifted disparity map
            patch_right = img_right[
                row_start:row_end, col_start - shift : col_end - shift, :
            ]

        # disparity patch, the patch location is the same as left patch
        patch_disp = img_disp[row_start:row_end, col_start:col_end]

        # transform image
        if self.transform is not None:
            # convert to tensor and normalize
            # transforms.ToTensor() change the axis into [rgb_channel, h, w]
            patch_left = self.transform(patch_left)  # [n_batch, rgb_channels, h, w]
            patch_right = self.transform(patch_right)  # [n_batch, rgb_channels, h, w]
            patch_disp = torch.tensor(patch_disp, dtype=torch.float32)  # [h, w]

        return patch_left, patch_right, patch_disp


class DatasetVal:
    def __init__(
        self,
        file_val_left,
        file_val_right,
        file_val_disp,
        patch_size_h,
        patch_size_w,
        c_disp_shift,
        transform=None,
        flip_input=1,
    ):
        self.file_val_left = file_val_left
        self.file_val_right = file_val_right
        self.file_val_disp = file_val_disp
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.c_disp_shift = c_disp_shift
        self.transform = transform
        self.flip_input = flip_input

    def __len__(self):
        return len(self.file_val_disp)

    def __getitem__(self, idx):
        # load image
        # get file name
        img_left_file = self.file_val_left[idx]
        img_right_file = self.file_val_right[idx]
        img_disp_file = self.file_val_disp[idx]

        ## read image
        # left image
        pic = imageio.imread(img_left_file) / 255.0
        img_left = np.array(
            pic.reshape(540, 960, 3), dtype=np.float32
        )  # [n_batch, h, w, rgb_channels]

        # right image
        pic = imageio.imread(img_right_file) / 255.0
        img_right = np.array(
            pic.reshape(540, 960, 3), dtype=np.float32
        )  # [n_batch, h, w, rgb_channels]

        ## load disparity image
        pic, scale = readPFM(img_disp_file)
        # shift disparity
        if self.c_disp_shift == 0:
            shift = 0  # np.random.randint(MAX_SHIFT)
            img_disp = np.array(np.clip(pic.reshape(540, 960) - shift, 0, 255)).astype(
                int
            )  # [n_batch, h, w]

            # generate random patch location
            row_start = np.random.randint(250)
            row_end = row_start + self.patch_size_h

            if self.flip_input:
                col_start = np.random.randint(150)  # for right disparity map
            else:
                col_start = np.random.randint(400)  # for left disparity map
            col_end = col_start + self.patch_size_w

        else:
            shift = np.clip(self.c_disp_shift * np.median(pic), 0, 255).astype(int)
            img_disp = np.array(
                np.clip(pic.reshape(540, 960) - shift, -128, 127)
            ).astype(
                np.int8
            )  # [n_batch, h, w]

            # generate random patch location
            row_start = np.random.randint(250)
            row_end = row_start + self.patch_size_h

            if self.flip_input:
                col_start = np.random.randint(150)  # for right disparity map
            else:
                col_start = np.random.randint(255, 400)  # for left disparity map
            col_end = col_start + self.patch_size_w

        ## create patch
        # left patch
        patch_left = img_left[row_start:row_end, col_start:col_end, :]

        if self.flip_input:
            # shift the right image with respect to the right disparity map
            # right patch, shift to the right to compensate the shifted disparity map
            patch_right = img_right[
                row_start:row_end, col_start + shift : col_end + shift, :
            ]
        else:
            ## shift the right image with respect to the left disparity map
            # right patch, shift to the left to compensate the shifted disparity map
            patch_right = img_right[
                row_start:row_end, col_start - shift : col_end - shift, :
            ]

        # disparity patch, the patch location is the same as left patch
        patch_disp = img_disp[row_start:row_end, col_start:col_end]

        # transform image
        if self.transform is not None:
            patch_left = self.transform(patch_left)  # [rgb_channels, h, w]
            patch_right = self.transform(patch_right)  # [rgb_channels, h, w]
            patch_disp = torch.tensor(patch_disp, dtype=torch.float32)  # [h, w]

        return patch_left, patch_right, patch_disp


class InnerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inner_product, patch_targets):
        # shift inner_product to avoid nan
        # max_val = inner_product.max(dim=1, keepdim=True)[0]
        # inner_product_shift = inner_product - max_val

        # calculate loss
        # patch_targets = input_disp.to(device)
        # loss = cross_entropy(inner_product_shift, patch_targets)
        loss = F.cross_entropy(input=inner_product, target=patch_targets)

        return loss


def compute_loss(model, input_val_left, input_val_right, input_val_disp):
    # n_batch_val = 128
    n_batch = len(input_val_left)
    # n_iters = len(input_val_left) // n_batch_val

    true_disp_val = torch.argmax(input_val_disp, dim=1).int()
    # loss_val = []
    # acc_val = []
    # for i in range(n_iters):

    # id_start = i * n_batch_val
    # id_end = id_start + n_batch_val
    inner_product = model(input_val_left, input_val_right)

    loss = F.cross_entropy(input=inner_product, target=input_val_disp)
    # loss_val.append(loss.item())

    ## calculate 3 pixel error for validation dataset
    predicted = torch.argmax(inner_product, dim=1).int()
    acc = torch.sum(torch.abs(predicted - true_disp_val) <= 3) / n_batch
    # acc_val.append(acc.item())

    return loss.item(), acc.item()


# @torch.no_grad()
# def estimate_loss(model, eval_iters):
#     """
#     helps estimate an arbitrarily accurate loss over either split using many batches

#     """
#     out = {}
#     model.eval()
#     for split in ["train", "val"]:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             x, y = get_batch(split)

#             with ctx:
#                 logits, loss = model(x, y)

#             losses[k] = loss.item()

#         out[split] = losses.mean()

#     model.train()

#     return out

# %%
