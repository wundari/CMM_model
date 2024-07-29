# %% load necessary modules
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# from torchvision import transforms as T
import torch.nn.functional as F

import os
import imageio.v2 as imageio
from python_pfm import *


# %% dataset
def generate_image_paths(train_list, val_list):
    # train_list = ["driving"]
    # val_list = ["flying", "monkaa"]

    ###############################
    ## generate training dataset ##
    ###############################
    dir_driving = "../Dataset/SceneFlow_complete/Driving/frames_cleanpass"
    dir_monkaa = "../Dataset/SceneFlow_complete/Monkaa/frames_cleanpass"
    dir_flying = "../Dataset/SceneFlow_complete/FlyingThings3D/frames_cleanpass/TRAIN"
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
        if paths[i].find("left") > -1:
            file_train_left.append(paths[i])
        elif paths[i].find("right") > -1:
            file_train_right.append(paths[i])

    # load disparity images
    dir_driving = "../Dataset/SceneFlow_complete/Driving/disparity"
    dir_monkaa = "../Dataset/SceneFlow_complete/Monkaa/disparity"
    dir_flying = "../Dataset/SceneFlow_complete/FlyingThings3D/disparity/TRAIN"
    dir_list = {"driving": dir_driving, "flying": dir_flying, "monkaa": dir_monkaa}

    paths = []
    for dir_name in train_list:
        dir = dir_list[dir_name]

        for root, dirs, files in os.walk(dir):
            for file in files:
                paths.append(os.path.join(root, file))

    file_train_disp = []
    for i in range(len(paths)):
        # if paths[i].find("left") > -1:
        #     file_train_disp.append(paths[i])

        if paths[i].find("right") > -1:
            file_train_disp.append(paths[i])

    # sort file
    file_train_left.sort()
    file_train_right.sort()
    file_train_disp.sort()

    # %%
    #################################
    ## generate validation dataset ##
    #################################
    dir_driving = "../Dataset/SceneFlow_complete/Driving/frames_cleanpass"
    dir_monkaa = "../Dataset/SceneFlow_complete/Monkaa/frames_cleanpass"
    # dir_flying = "../Dataset/SceneFlow_complete/FlyingThings3D/frames_cleanpass/TEST"
    dir_flying = "../Dataset/SceneFlow_complete/FlyingThings3D/frames_cleanpass/TRAIN"
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
        if paths[i].find("left") > -1:
            file_val_left.append(paths[i])
        elif paths[i].find("right") > -1:
            file_val_right.append(paths[i])

    # load disparity images
    dir_driving = "../Dataset/SceneFlow_complete/Driving/disparity"
    dir_monkaa = "../Dataset/SceneFlow_complete/Monkaa/disparity"
    # dir_flying = "../Dataset/SceneFlow_complete/FlyingThings3D/disparity/TEST"
    dir_flying = "../Dataset/SceneFlow_complete/FlyingThings3D/disparity/TRAIN"
    dir_list = {"driving": dir_driving, "flying": dir_flying, "monkaa": dir_monkaa}

    paths = []
    for dir_name in val_list:
        dir = dir_list[dir_name]

        for root, dirs, files in os.walk(dir):
            for file in files:
                paths.append(os.path.join(root, file))

    file_val_disp = []
    for i in range(len(paths)):
        # if paths[i].find("left") > -1:
        #     file_val_disp.append(paths[i])

        if paths[i].find("right") > -1:
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


def shift_image_disparity(
    file_train_left,
    file_train_right,
    file_train_disp,
    file_val_left,
    file_val_right,
    file_val_disp,
    shift_disparity,
):
    # %%
    idx = 0
    img_left_file = file_train_left[idx]
    img_right_file = file_train_right[idx]
    img_disp_file = file_train_disp[idx]

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
    img_disp = np.array(pic.reshape(540, 960), dtype=np.int16)  # [n_batch, h, w]

    ## create patch
    # %% generate random patch location
    patch_size = 37
    row_start = 275  # np.random.randint(250)
    row_end = row_start + patch_size
    col_start = 200  # np.random.randint(440)
    col_end = col_start + patch_size

    # left patch
    patch_left = img_left[row_start:row_end, col_start:col_end, :]

    # right patch
    shift = 12
    patch_right = img_right[row_start:row_end, col_start + shift : col_end + shift, :]

    # disparity patch, the patch location is the same as left patch
    patch_disp = img_disp[row_start:row_end, col_start:col_end] - shift

    # shift roght image
    patch_right_shift = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    for i in range(patch_size):
        id_row = row_start + i
        for j in range(patch_size):
            id_col = col_start + j + +shift + patch_disp[i, j]
            patch_right_shift[i, j] = img_right[id_row, id_col, :]

    fig, axes = plt.subplots(nrows=1, ncols=4)
    axes[0].imshow(patch_left)
    axes[0].axis("off")
    axes[1].imshow(patch_right)
    axes[1].axis("off")
    axes[2].imshow(patch_disp)
    axes[2].axis("off")
    axes[3].imshow(patch_right_shift)
    axes[3].axis("off")

    # %% transform image
    if self.transform is not None:
        # convert to tensor and normalize
        # transforms.ToTensor() change the axis into [rgb_channel, h, w]
        patch_left = self.transform(patch_left)  # [n_batch, rgb_channels, h, w]
        patch_right = self.transform(patch_right)  # [n_batch, rgb_channels, h, w]


# %% define patch dataset class
class DatasetTrain(Dataset):
    def __init__(
        self,
        file_train_left,
        file_train_right,
        file_train_disp,
        patch_size_h,
        patch_size_w,
        transform=None,
    ):
        self.file_train_left = file_train_left
        self.file_train_right = file_train_right
        self.file_train_disp = file_train_disp
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.transform = transform

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
        pic, scale = readPFM(img_disp_file)
        img_disp = np.array(pic.reshape(540, 960), dtype=np.uint8)  # [n_batch, h, w]

        ## create patch
        # generate random patch location
        row_start = np.random.randint(250)
        row_end = row_start + self.patch_size_h
        col_start = np.random.randint(440)
        col_end = col_start + self.patch_size_w

        # left patch
        patch_left = img_left[row_start:row_end, col_start:col_end, :]

        # right patch
        patch_right = img_right[row_start:row_end, col_start:col_end, :]

        # disparity patch, the patch location is the same as left patch
        patch_disp = img_disp[row_start:row_end, col_start:col_end]

        # transform image
        if self.transform is not None:
            # convert to tensor and normalize
            # transforms.ToTensor() change the axis into [rgb_channel, h, w]
            patch_left = self.transform(patch_left)  # [n_batch, rgb_channels, h, w]
            patch_right = self.transform(patch_right)  # [n_batch, rgb_channels, h, w]

        return patch_left, patch_right, patch_disp


class DatasetVal:
    def __init__(
        self,
        file_val_left,
        file_val_right,
        file_val_disp,
        patch_size_h,
        patch_size_w,
        transform=None,
    ):
        self.file_val_left = file_val_left
        self.file_val_right = file_val_right
        self.file_val_disp = file_val_disp
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.transform = transform

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
        img_disp = np.array(pic.reshape(540, 960), dtype=np.uint8)  # [n_batch, h, w]

        ## create patch
        # generate random patch location
        row_start = np.random.randint(250)
        row_end = row_start + self.patch_size_h
        col_start = np.random.randint(440)
        col_end = col_start + self.patch_size_w

        # left patch
        patch_left = img_left[row_start:row_end, col_start:col_end, :]

        # right patch
        patch_right = img_right[row_start:row_end, col_start:col_end, :]

        # disparity patch, the patch location is the same as left patch
        patch_disp = img_disp[row_start:row_end, col_start:col_end]

        # transform image
        if self.transform is not None:
            patch_left = self.transform(patch_left)  # [rgb_channels, h, w]
            patch_right = self.transform(patch_right)  # [rgb_channels, h, w]

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
