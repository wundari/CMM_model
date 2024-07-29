# %% load necessary modules
import numpy as np
import random
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torch import optim
import torch.nn.functional as F

import glob
import imageio.v2 as imageio


# %% dataset
def load_images(img_dir, disp_dir_name="disp_noc_0", split="training"):
    ## gather image paths
    file_list_left = glob.glob(img_dir + split + "/image_2/*")
    file_list_right = glob.glob(img_dir + split + "/image_3/*")
    file_list_disp = glob.glob(img_dir + "training/" + disp_dir_name + "/*")

    # sort
    file_list_left.sort()
    file_list_right.sort()
    file_list_disp.sort()

    ## store all images
    imgs_left = {}
    imgs_right = {}
    imgs_disp = {}

    for f in range(len(file_list_disp)):
        # get file name
        img_left_file = file_list_left[2 * f]  # even number, images with ending _10
        img_right_file = file_list_right[2 * f]
        img_disp_file = file_list_disp[f]

        ## read image
        # left image
        pic = imageio.imread(img_left_file) / 255.0
        # pic = (pic - pic.mean()) / pic.std()  # standardize image
        img_left = np.array(pic, dtype=np.float32)
        # img_left = np.array(
        #     np.moveaxis(pic.reshape(pic.shape[0], pic.shape[1], 3), 2, 0),
        #     dtype=np.float32,
        # )
        # right image
        pic = imageio.imread(img_right_file) / 255.0
        # pic = (pic - pic.mean()) / pic.std()  # standardize image
        img_right = np.array(pic, dtype=np.float32)
        # img_right = np.array(
        #     np.moveaxis(pic.reshape(pic.shape[0], pic.shape[1], 3), 2, 0),
        #     dtype=np.float32,
        # )

        # store images
        imgs_left[f] = img_left
        imgs_right[f] = img_right

        ###############################################################################
        ## load disparity image
        pic = imageio.imread(img_disp_file)
        img_disp = np.array(pic, dtype=np.uint8)

        # store images
        imgs_disp[f] = img_disp

    return imgs_left, imgs_right, imgs_disp


# %% define patch dataset class
class DatasetTrain(Dataset):
    def __init__(
        self,
        imgs_left,
        imgs_right,
        imgs_disp,
        patch_size_h,
        patch_size_w,
        transform=None,
    ):
        # self.img_dir = img_dir
        self.imgs_left = imgs_left
        self.imgs_right = imgs_right
        self.imgs_disp = imgs_disp
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.transform = transform

        ###############################################################################
        #### set up training dataset ####
        ##############################################################################
        ## load patch_location training dataset
        # [file_id, row_center, col_center_left, col_center_right]
        patch_loc = pd.read_csv(
            "data/patch_loc_train_h_{}_w_{}.csv".format(
                self.patch_size_h, self.patch_size_w
            )
        )
        # shuffle dataset
        patch_loc = patch_loc.sample(frac=1)
        self.patch_loc = patch_loc

        # create image patches for training dataset
        # here the dataset only contains the patch coordinate
        # n_train = len(self.patch_loc_train)
        patches = np.zeros(
            (len(self.patch_loc), 4), dtype=np.int32
        )  # use numpy for dict indexing
        for i in range(len(self.patch_loc)):
            # [file_id, row_center, col_center_left, col_center_right]
            # temp = patch_loc_train.values[i]
            patches[i] = self.patch_loc.values[i]

        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        ## load image
        # patches: [file_id, row_center, col_center_left, col_center_right]
        file_id = self.patches[idx, 0]

        img_left = self.imgs_left[file_id]
        img_right = self.imgs_right[file_id]
        img_disp = self.imgs_disp[file_id]

        ## create patch
        # left patch
        # row_start = kitti_patch_train.patch_train.iloc[0].row_start
        row_start = self.patches[idx, 1] - self.patch_size_h // 2
        row_end = row_start + self.patch_size_h

        col_left_start = self.patches[idx, 2] - self.patch_size_w // 2
        col_left_end = col_left_start + self.patch_size_w
        patch_left = img_left[row_start:row_end, col_left_start:col_left_end, :]

        # right patch
        # col_right_start = kitti_patch_train.patch_train.iloc[0].col_right_start
        # col_right_end = col_right_start + kitti_patch_train.patch_size + kitti_patch_train.disp_range
        col_right_start = self.patches[idx, 3] - self.patch_size_w // 2
        col_right_end = col_right_start + self.patch_size_w
        patch_right = img_right[row_start:row_end, col_right_start:col_right_end, :]

        # disparity patch, the patch location is the same as left patch
        col_disp_start = self.patches[idx, 2] - self.patch_size_w // 2
        col_disp_end = col_disp_start + self.patch_size_w
        patch_disp = img_disp[row_start:row_end, col_disp_start:col_disp_end]

        # convert into tensor
        if self.transform is not None:
            # convert to tensor and normalize
            # transforms.ToTensor() change the axis into [rgb_channel, h, w]
            patch_left = self.transform(patch_left)  # [n_batch, rgb_channels, h, w]
            patch_right = self.transform(patch_right)  # [n_batch, rgb_channels, h, w]

        return patch_left, patch_right, patch_disp


class DatasetVal:
    def __init__(
        self,
        imgs_left,
        imgs_right,
        imgs_disp,
        patch_size_h,
        patch_size_w,
        transform=None,
    ):
        self.imgs_left = imgs_left
        self.imgs_right = imgs_right
        self.imgs_disp = imgs_disp
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.transform = transform

        ###############################################################################
        #### set up validation dataset ####
        ##############################################################################
        ## load patch_location validation dataset
        patch_loc = pd.read_csv(
            "data/patch_loc_val_h_{}_w_{}.csv".format(
                self.patch_size_h, self.patch_size_w
            )
        )
        # shuffle dataset
        patch_loc = patch_loc.sample(frac=1)
        self.patch_loc = patch_loc

        # create image patches for training dataset
        # here the dataset only contains the patch coordinate
        # n_val = len(patch_loc_val)
        patches = np.zeros((len(self.patch_loc), 4), dtype=np.int32)
        for i in range(len(self.patch_loc)):
            # [file_id, row_center, col_center_left, col_center_right]
            temp = self.patch_loc.values[i]
            patches[i] = temp

        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        ## load image
        # patches_train: [file_id, row_center, col_center_left, col_center_right]
        file_id = self.patches[idx, 0]

        img_left = self.imgs_left[file_id]
        img_right = self.imgs_right[file_id]
        img_disp = self.imgs_disp[file_id]

        ## create patch
        # left patch
        # row_start = kitti_patch_train.patch_train.iloc[0].row_start
        row_start = self.patches[idx, 1] - self.patch_size_h // 2
        row_end = row_start + self.patch_size_h

        col_left_start = self.patches[idx, 2] - self.patch_size_w // 2
        col_left_end = col_left_start + self.patch_size_w
        patch_left = img_left[row_start:row_end, col_left_start:col_left_end, :]

        # right patch
        # col_right_start = kitti_patch_train.patch_train.iloc[0].col_right_start
        # col_right_end = col_right_start + kitti_patch_train.patch_size + kitti_patch_train.disp_range
        col_right_start = self.patches[idx, 3] - self.patch_size_w // 2
        col_right_end = col_right_start + self.patch_size_w
        patch_right = img_right[row_start:row_end, col_right_start:col_right_end, :]

        # disparity patch, the patch location is the same as left patch
        col_disp_start = self.patches[idx, 2] - self.patch_size_w // 2
        col_disp_end = col_disp_start + self.patch_size_w
        patch_disp = img_disp[row_start:row_end, col_disp_start:col_disp_end]

        # convert into tensor
        if self.transform is not None:
            # convert to tensor and normalize
            # transforms.ToTensor() change the axis into [rgb_channel, h, w]
            patch_left = self.transform(patch_left)  # [n_batch, rgb_channels, h, w]
            patch_right = self.transform(patch_right)  # [n_batch, rgb_channels, h, w]

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
