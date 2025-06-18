# %% load necessary modules
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch._dynamo

# import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from engine.model import GCNet
from data_handler.DataHandler_SceneFlow_v2 import (
    generate_image_paths,
    DatasetTrain,
    DatasetVal,
)

# reproducibility
import random

seed_number = 3407  # 12321
torch.manual_seed(seed_number)
np.random.seed(seed_number)

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)


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

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# %%
class Engine:
    """
    Engine is a parent class designed for setting up the model

    Attributes:
        w_bg (int): Width of the background.

        h_bg (int): Height of the background.

        n_bootstrap (int): Number of bootstrapping iterations.

        batch_size (int): Batch size for processing.

        maxdisp (int): Maximum disparity, should be a multiple of 32.

        loss_mul (list): List for storing loss multipliers.

        epoch_to_load (int): The epoch number at which the model is loaded.

        iter_to_load (int): The iteration number at which the model is loaded.

        disp_mag (int): Magnitude of the disparity.

        c_disp_shift (float): Multiplier for disparity shift.

        dataset_to_process (str): Type of the dataset to process, e.g.,
        "driving", "flying", "monkaa".

        transform (torch.nn.Sequential): Image transformation pipeline.

        load_model (function): Method for loading the model.

        save_dir (str): Directory for saving results.

        network_diss_dir (str): Directory for saving network dissection results.

    Methods:
        dataset_to_process: Getter and setter for `dataset_to_process` attribute.
        loss_mul: Getter and setter for `loss_mul` attribute.
        load_model: Getter and setter for loading and managing the model.
        get_activations_shape(target: nn.Module):
            Gets the shape of activations for a given model layer.
        default_loss_summarize(loss_value: torch.Tensor):
            Summarizes loss values for optimization.

    Args:
        sceneflow_type (str): Type of scene flow dataset to use.
        epoch_to_load (int): Epoch number for loading the model.
        iter_to_load (int): Iteration number for loading the model.
        disp_mag (int): Magnitude of disparity.
        c_disp_shift (float): Disparity shift multiplier.
    """

    def __init__(
        self,
        params_network: dict,
        params_train: dict,
    ) -> None:
        # network parameter
        self.w_bg = params_network["w_bg"]  # input width
        self.h_bg = params_network["h_bg"]  # input height
        self.maxdisp = params_network["maxdisp"]  # must be a multiple of 32

        # parameters for the model to be used
        # sceneflow_type: "driving", "flying", "monkaa"
        # for ex:
        # dataset names: "driving", "flying", "monkaa"
        # epoch_to_load (earlystop): driving: 4; flying: 4; monkaa: 5
        # iter_to_load (earlystop): driving: 6801; flying: 29201; monkaa: 17201
        self.epoch_to_load = params_train["epoch_to_load"]
        self.iter_to_load = params_train["iter_to_load"]
        self.batch_size = params_train["batch_size"]  # training batch size
        self.c_disp_shift = params_train["c_disp_shift"]  # disparity shift multiplier
        self.dataset_to_process = params_train["sceneflow_type"]
        self.learning_rate = params_train["learning_rate"]
        self.eval_iter = params_train["eval_iter"]
        self.eval_interval = params_train["eval_interval"]
        self.n_epoch = params_train["n_epoch"]
        # self.max_iter = params_train["max_iter"]
        self.load_state = params_train["load_state"]
        self.flip_input = params_train["flip_input"]
        self.train_or_eval_mode = params_train["train_or_eval_mode"]
        self.compile_mode = params_train["compile_mode"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

        # construct disparity indices for normalizing the probability of
        # cost volume across disparity axis (see eq. 1 in the Kendall 2017 paper)
        self.disp_indices = []

        # prepare saving directory
        if self.flip_input:
            self.save_dir = (
                "results/sceneflow/"
                + self.dataset_to_process
                + f"/shift_{self.c_disp_shift}_median_wrt_right"
            )
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        else:
            self.save_dir = (
                "results/sceneflow/"
                + self.dataset_to_process
                + f"/shift_{self.c_disp_shift}_median_wrt_left"
            )
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # checkpoint directory
        self.checkpoint_dir = f"{self.save_dir}/checkpoint"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        # pred_images directory
        self.pred_images_dir = f"{self.save_dir}/pred_images"
        if not os.path.exists(self.pred_images_dir):
            os.mkdir(f"{self.save_dir}/pred_images")

        # load model
        self.model = GCNet(self.h_bg, self.w_bg, self.maxdisp)

        if self.load_state:
            # load pre-trained gcnet
            checkpoint = torch.load(
                f"{self.checkpoint_dir}/"
                + f"gcnet_state_earlystop_epoch_{self.epoch_to_load}"
                + f"_iter_{self.iter_to_load}.pth"
            )
            # checkpoint = torch.load(
            #     f"{self.checkpoint_dir}/"
            #     + f"gcnet_state_epoch_{self.epoch_to_load}"
            #     + f"_iter_{self.iter_to_load}.pth"
            # )
            self.checkpoint = checkpoint
            # fix the keys of the state dictionary
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)

        self.model = torch.compile(
            self.model, mode=self.compile_mode
        )  # use compile_mode = "default" for layer analysis
        # self.model = torch.compile(self.model)
        self.model.to(self.device)

        if self.train_or_eval_mode == "train":
            self.model.train()  # training mode
            print(
                f"GC-Net was successfully loaded to {self.device}, "
                + f"{self.train_or_eval_mode} mode, "
                + f"compile mode: {self.compile_mode}\n"
            )
            print(
                f"DNN will be trained on {self.dataset_to_process} dataset "
                + f"with batch size {self.batch_size} for {self.n_epoch} epochs"
            )
        elif self.train_or_eval_mode == "eval":
            self.model.eval()  # evaluation mode
            print(
                f"GC-Net was successfully loaded to {self.device}, "
                + f"{self.train_or_eval_mode} mode, "
                + f"compile mode: {self.compile_mode}\n"
            )
            print(
                f"DNN was trained on {self.dataset_to_process} dataset "
                + f"with batch size {self.batch_size} for {self.n_epoch} epochs"
            )

    @property
    def disp_indices(self):
        return self._disp_indices

    @disp_indices.setter
    def disp_indices(self, disp_indices_list):
        """
        create disparity indices used for calculating the expected value
        of model's output to regress disparity (see eq. 1 in Kendall 2017 paper)
        """
        disp_indices_list = [
            d * torch.ones((1, self.h_bg, self.w_bg))
            for d in range(-self.maxdisp // 2, self.maxdisp // 2, 1)
        ]
        self._disp_indices = (
            torch.cat(disp_indices_list, 0)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )

    def predict(self, input_left, input_right):
        logits = self.model(
            input_left, input_right
        )  # [n_batch, max_disp, img_height, img_width]

        # regress disparity for each pixel by computing
        # the expected value of normalized cost volume across disparity dimension
        # (eq. 1 in Kendall 2017 paper)
        pred_disp = torch.sum(
            logits.mul(self.disp_indices), dim=1
        )  # [n_batch, img_height, img_width]

        return pred_disp

    @staticmethod
    def default_loss_summarize(loss_value: torch.Tensor) -> torch.Tensor:
        """
        Helper function to summarize tensor outputs from loss functions.

        default_loss_summarize applies `mean` to the loss tensor
        and negates it so that optimizing it maximizes the activations we
        are interested in.
        """
        lambda_reg = 0.0
        return -1 * (loss_value.mean() + lambda_reg * (loss_value**2).sum())

    def prepare_dataset(self):
        # prepare dataset
        train_list = [self.dataset_to_process]
        val_list = [self.dataset_to_process]
        # train_list = ["driving", "flying", "monkaa"]
        # val_list = ["driving", "flying", "monkaa"]
        (
            file_train_left,
            file_train_right,
            file_train_disp,
            file_val_left,
            file_val_right,
            file_val_disp,
        ) = generate_image_paths(train_list, val_list, self.flip_input)

        # data normalization notes: https://cs231n.github.io/neural-networks-2/
        # a = np.zeros((len(imgs_left), 540, 960, 3), dtype=np.float32)
        # for i in range(len(imgs_left)):
        #     a[i] = imgs_left[i]
        # DATA_MEANS = [a[:, :, :, 0].mean(), a[:, :, :, 1].mean(), a[:, :, :, 2].mean()]
        # DATA_STD = [a[:, :, :, 0].std(), a[:, :, :, 1].std(), a[:, :, :, 2].std()]
        # DATA_MEANS = np.array([0.32, 0.32, 0.28])
        # DATA_STD = np.array([0.28, 0.27, 0.25])
        DATA_MEANS = np.array([0.5, 0.5, 0.5])
        DATA_STD = np.array([0.5, 0.5, 0.5])

        transform_data = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
        )

        # transform_data = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]
        # )

        # get training dataset
        n_train = int(len(file_train_disp) * 0.8)
        train_id = np.random.choice(
            np.arange(len(file_train_disp)), n_train, replace=False
        )

        patch_data = DatasetTrain(
            [file_train_left[i] for i in train_id],
            [file_train_right[i] for i in train_id],
            [file_train_disp[i] for i in train_id],
            self.h_bg,
            self.w_bg,
            self.c_disp_shift,
            transform=transform_data,
            flip_input=self.flip_input,
        )

        train_loader = DataLoader(
            patch_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get validation dataset
        val_id = np.setdiff1d(np.arange(len(file_train_disp)), train_id)
        patch_data = DatasetVal(
            [file_val_left[i] for i in val_id],
            [file_val_right[i] for i in val_id],
            [file_val_disp[i] for i in val_id],
            self.h_bg,
            self.w_bg,
            self.c_disp_shift,
            transform=transform_data,
            flip_input=self.flip_input,
        )

        val_loader = DataLoader(
            patch_data,
            batch_size=2,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # check input
        inputs_left, inputs_right, disps = next(iter(val_loader))
        img_left = (inputs_left[0] * 128 + 127).to(torch.uint8)
        img_right = (inputs_right[0] * 128 + 127).to(torch.uint8)
        img_disp = disps[0]

        ## create patch
        # generate random patch location
        patch_size = 100
        row_start = np.random.randint(150)
        row_end = row_start + patch_size
        col_start = np.random.randint(200, 300)
        col_end = col_start + patch_size

        # left patch
        patch_left = img_left[:, row_start:row_end, col_start:col_end]

        # right patch
        patch_right = img_right[:, row_start:row_end, col_start:col_end]

        # disparity patch, the patch location is the same as left patch
        patch_disp = img_disp[row_start:row_end, col_start:col_end]

        # shift right image
        patch_right_shift = torch.zeros((3, patch_size, patch_size), dtype=torch.uint8)
        for i in range(patch_size):
            id_row = row_start + i
            for j in range(patch_size):
                if self.flip_input:
                    # shift with respect to right disparity image
                    id_col = col_start + j + patch_disp[i, j].to(torch.int)
                else:
                    # shift with respect to left disparity image
                    id_col = col_start + j - patch_disp[i, j].to(torch.int)

                patch_right_shift[:, i, j] = img_right[:, id_row, id_col]

        fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=3)
        ## draw left patch
        axes[0].imshow(img_left.permute(1, 2, 0))
        axes[0].set_title("Left patch")
        axes[0].axis("off")
        # draw box
        axes[0].plot([col_start, col_end], [row_start, row_start], "r-")
        axes[0].plot([col_start, col_start], [row_start, row_end], "r-")
        axes[0].plot([col_start, col_end], [row_end, row_end], "r-")
        axes[0].plot([col_end, col_end], [row_start, row_end], "r-")
        ## draw right patch
        axes[1].imshow(img_right.permute(1, 2, 0))
        axes[1].set_title("Right patch")
        axes[1].axis("off")
        # draw box
        axes[1].plot([col_start, col_end], [row_start, row_start], "r-")
        axes[1].plot([col_start, col_start], [row_start, row_end], "r-")
        axes[1].plot([col_start, col_end], [row_end, row_end], "r-")
        axes[1].plot([col_end, col_end], [row_start, row_end], "r-")
        ## draw disp map
        axes[2].imshow(img_disp, cmap="jet", vmin=-128, vmax=127)
        axes[2].set_title("Disparity map (left)")
        axes[2].axis("off")

        fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=4)
        axes[0].imshow(patch_left.permute(1, 2, 0))
        axes[0].set_title("Left patch")
        axes[0].axis("off")
        axes[1].imshow(patch_right.permute(1, 2, 0))
        axes[1].set_title("Right patch")
        axes[1].axis("off")
        axes[2].imshow(patch_disp, cmap="jet", vmin=-128, vmax=127)
        axes[2].set_title("Disparity map (left)")
        axes[2].axis("off")
        axes[3].imshow(patch_right_shift.permute(1, 2, 0))
        axes[3].set_title("Shifted right image")
        axes[3].axis("off")

        print(
            "mean-disp: {:.2f}, min-disp: {}, max-disp: {}".format(
                disps.float().mean(), disps.min(), disps.max()
            )
        )
        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        # configure optimizer
        # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=np.arange(1, self.n_epoch), gamma=0.2
        )

        if self.load_state:
            # load the previous state for optimizer
            optimizer.load_state_dict(self.checkpoint["optimizer"])

        # define loss function
        loss_module = torch.nn.L1Loss()

        # train
        # get the parameters from the previous training
        if self.load_state:
            iter_num = self.checkpoint["iter"] - 1
            epoch_prev = self.checkpoint["epoch"]
            acc_train_epoch = self.checkpoint["acc_train_epoch"]
            acc_val_epoch = self.checkpoint["acc_val_epoch"]
            loss_train_epoch = self.checkpoint["loss_train_epoch"]
            loss_val_epoch = self.checkpoint["loss_val_epoch"]
            loss_val_best = self.checkpoint["loss_val_best"]
            acc_val_best = self.checkpoint["acc_val_best"]
            c_disp_shift = self.checkpoint["c_disp_shift"]
        else:
            iter_num = 0
            epoch_prev = 0
            acc_train_epoch = []
            acc_val_epoch = []
            loss_train_epoch = []
            loss_val_epoch = []
            loss_val_best = np.inf
            acc_val_best = 0.0

        count = 0
        for epoch in range(epoch_prev, self.n_epoch):
            tepoch = tqdm(train_loader, unit="batch")

            # for i in range(iter_num, len(train_loader)):
            for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
                # for inputs_left, inputs_right, disps in t:

                # [n_batch, n_channels, height, width]
                # inputs_left, inputs_right, disps = next(iter(train_loader))

                # move to gpu
                input_left = inputs_left.pin_memory().to(self.device, non_blocking=True)
                input_right = inputs_right.pin_memory().to(
                    self.device, non_blocking=True
                )
                input_disp = disps.pin_memory().to(self.device, non_blocking=True)

                # zero the gradients()
                optimizer.zero_grad()

                # model output
                pred = self.predict(input_left, input_right)

                # logits = self.model(
                #     input_left, input_right
                # )  # [n_batch, maxdisp, img_height, img_width]
                # pred = torch.sum(
                #     logits.mul(self.disp_indices), 1
                # )  # [n_batch, img_height, img_width]
                # # print(pred.shape)

                # compute accuracy (3 pixel error for training dataset)
                # pred = pred.view(batch_size, 1, h, w)
                diff = torch.abs(pred.data - input_disp.data)
                acc_train = torch.sum(diff < 3) / float(
                    self.h_bg * self.w_bg * self.batch_size
                )

                # compute train loss
                loss_train = loss_module(pred, input_disp)

                # backpropagation
                loss_train.backward()

                # update weights
                optimizer.step()

                tepoch.set_description(
                    "Epoch: {}/{}, iter: {}/{}, lr: {:.8f}, train loss: {:4.2f}, train_acc: {:4.2f}".format(
                        epoch + 1,
                        self.n_epoch,
                        count + 1,
                        self.n_epoch * len(train_loader),
                        optimizer.param_groups[0]["lr"],
                        loss_train.item(),
                        acc_train.item(),
                    )
                )

                if (count >= 0) & (count % self.eval_interval == 0):
                    # iterator for evaluation
                    iterator_eval_train = iter(train_loader)
                    iterator_eval_val = iter(val_loader)

                    # estimate an arbitrarily accurate loss
                    with torch.no_grad():
                        self.model.eval()
                        losses_train = torch.zeros(self.eval_iter)
                        accs_train = torch.zeros(self.eval_iter)
                        losses_val = torch.zeros(self.eval_iter)
                        accs_val = torch.zeros(self.eval_iter)
                        for k in range(self.eval_iter):
                            ####################
                            #### train loss ####
                            ####################

                            if (k >= len(iterator_eval_train)) & (
                                k % len(iterator_eval_train) == 0
                            ):
                                # recycle train_loader
                                iterator_eval_train = iter(train_loader)
                                input_eval_left, input_eval_right, disps_eval = next(
                                    iterator_eval_train
                                )
                            else:
                                input_eval_left, input_eval_right, disps_eval = next(
                                    iterator_eval_train
                                )

                            # move to gpu, # [n_batch, 3, patch_size, patch_size]
                            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                            input_left = input_eval_left.pin_memory().to(
                                self.device, non_blocking=True
                            )
                            input_right = input_eval_right.pin_memory().to(
                                self.device, non_blocking=True
                            )
                            input_disp = disps_eval.pin_memory().to(
                                self.device, non_blocking=True
                            )

                            # compute model output
                            pred = self.predict(input_left, input_right)
                            # logits = self.model(input_left, input_right)
                            # pred = torch.sum(logits.mul(self.disp_indices), 1)

                            # compute training loss
                            losses_train[k] = loss_module(pred, input_disp)

                            # compute accuracy
                            # pred = pred.view(batch_size, 1, h, w)
                            diff = torch.abs(pred.data - input_disp.data)
                            accs_train[k] = torch.sum(diff < 3) / float(
                                self.h_bg * self.w_bg * self.batch_size
                            )

                            #########################
                            #### validation loss ####
                            #########################

                            if (k >= len(iterator_eval_val)) & (
                                k % len(iterator_eval_val) == 0
                            ):
                                # recycle val_loader
                                iterator_eval_val = iter(val_loader)
                                input_eval_left, input_eval_right, disps_eval = next(
                                    iterator_eval_val
                                )
                            else:
                                input_eval_left, input_eval_right, disps_eval = next(
                                    iterator_eval_val
                                )

                            # move to gpu, # [n_batch, 3, patch_size, patch_size]
                            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                            input_left = input_eval_left.pin_memory().to(
                                self.device, non_blocking=True
                            )
                            input_right = input_eval_right.pin_memory().to(
                                self.device, non_blocking=True
                            )
                            input_disp = disps_eval.pin_memory().to(
                                self.device, non_blocking=True
                            )

                            # compute model output
                            pred = self.predict(input_left, input_right)
                            # logits = self.model(input_left, input_right)
                            # pred = torch.sum(logits.mul(self.disp_indices), 1)

                            # compute loss
                            losses_val[k] = loss_module(pred, input_disp)

                            # compute accuracy
                            # pred = pred.view(batch_size, 1, h, w)
                            diff = torch.abs(pred.data - input_disp.data)
                            accs_val[k] = torch.sum(diff < 3) / float(
                                self.h_bg * self.w_bg * self.batch_size
                            )

                        self.model.train()

                    # record losses and accuracies
                    acc_train_epoch.append(accs_train.mean().item())
                    acc_val_epoch.append(accs_val.mean().item())
                    loss_train_epoch.append(losses_train.mean().item())
                    loss_val_epoch.append(losses_val.mean().item())

                    tepoch.set_postfix(
                        trainloss=losses_train.mean().item(),
                        valloss=losses_val.mean().item(),
                        trainacc=accs_train.mean().item(),
                        valacc=accs_val.mean().item(),
                    )

                    # save network state
                    print("=======saving model=======")
                    state = {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": count + 1,
                        "epoch": epoch + 1,
                        "batch_size": self.batch_size,
                        "loss_train_list": loss_train_epoch,
                        "acc_train_list": acc_train_epoch,
                        "loss_val_list": loss_val_epoch,
                        "acc_val_list": acc_val_epoch,
                        "loss_val_best": loss_val_best,
                        "acc_val_best": acc_val_best,
                        "training_dataset": self.dataset_to_process,
                        "validation_dataset": self.dataset_to_process,
                        "c_disp_shift": self.c_disp_shift,
                        "flip_input": self.flip_input,
                        "seed_number": seed_number,
                    }
                    torch.save(
                        state,
                        f"{self.checkpoint_dir}/gcnet_state_epoch_"
                        + f"{epoch + 1}_iter_{count + 1}.pth",
                    )

                    # save best validation loss state
                    if losses_val.mean().item() < loss_val_best:
                        loss_val_best = losses_val.mean().item()
                        acc_val_best = accs_val.mean().item()

                        print(
                            f"Found a better val_loss: {loss_val_best:.2f}, "
                            + f"val_acc: {acc_val_best:.2f}. "
                            + "Save the best model"
                        )

                        state = {
                            "model": self.model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter": count + 1,
                            "epoch": epoch + 1,
                            "batch_size": self.batch_size,
                            "loss_train_list": loss_train_epoch,
                            "acc_train_list": acc_train_epoch,
                            "loss_val_list": loss_val_epoch,
                            "acc_val_list": acc_val_epoch,
                            "loss_val_best": loss_val_best,
                            "acc_val_best": acc_val_best,
                            "training_dataset": self.dataset_to_process,
                            "validation_dataset": self.dataset_to_process,
                            "c_disp_shift": self.c_disp_shift,
                            "flip_input": self.flip_input,
                            "seed_number": seed_number,
                        }
                        torch.save(
                            state,
                            f"{self.checkpoint_dir}/"
                            + f"gcnet_state_earlystop_epoch_{epoch + 1}_"
                            + f"iter_{count + 1}.pth",
                        )

                    ## save an example of predicted images
                    figsize = (12, 8)
                    n_row = 2
                    n_col = 2
                    sns.set_theme()
                    sns.set_theme(
                        context="paper", style="white", font_scale=2, palette="deep"
                    )

                    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize)
                    fig.text(0.5, 1, f"Inputs for GC-Net", ha="center")
                    fig.tight_layout()
                    plt.subplots_adjust(wspace=0.2, hspace=0.2)

                    img_left = (
                        input_eval_left[0].transpose(0, 2).transpose(0, 1) * 128 + 127
                    ).to(torch.uint8)
                    axes[0, 0].imshow(img_left)
                    axes[0, 0].set_title("Left image")

                    img_right = (
                        input_eval_right[0].transpose(0, 2).transpose(0, 1) * 128 + 127
                    ).to(torch.uint8)
                    axes[0, 1].imshow(img_right)
                    axes[0, 1].set_title("Right image")

                    # disparity ground truth
                    im = disps_eval[0].numpy()
                    v_min = -1 * np.max(np.abs(im))
                    v_max = im.max()
                    temp = axes[1, 0].imshow(im, cmap="jet", vmin=v_min, vmax=v_max)
                    axes[1, 0].set_title("Ground truth")
                    fig.colorbar(temp)

                    # predicted disparity
                    im = pred[0].data.cpu().numpy()
                    temp = axes[1, 1].imshow(im, cmap="jet", vmin=v_min, vmax=v_max)
                    axes[1, 1].set_title("Predicted image")
                    fig.colorbar(temp)

                    # turn off axis for all subplots
                    for ax in axes.ravel():
                        ax.set_axis_off()

                    plt.savefig(
                        f"{self.pred_images_dir}/output.pdf",
                        bbox_inches="tight",
                    )

                    plt.close()

                # increase counter
                count += 1
            # decay learning rate
            scheduler.step()

        # save the last network state
        print("=======saving model=======")
        state = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": count + 1,
            "epoch": epoch + 1,
            "batch_size": self.batch_size,
            "loss_train_list": loss_train_epoch,
            "acc_train_list": acc_train_epoch,
            "loss_val_list": loss_val_epoch,
            "acc_val_list": acc_val_epoch,
            "loss_val_best": loss_val_best,
            "acc_val_best": acc_val_best,
            "training_dataset": self.dataset_to_process,
            "validation_dataset": self.dataset_to_process,
            "c_disp_shift": self.c_disp_shift,
            "flip_input": self.flip_input,
            "seed_number": seed_number,
        }

        torch.save(
            state,
            f"{self.checkpoint_dir}/gcnet_state_epoch_"
            + f"{epoch + 1}_iter_{count + 1}.pth",
        )

    def plotLine_learning_curve(self, epoch_to_load, iter_to_load, save_flag):

        checkpoint = torch.load(
            f"{self.checkpoint_dir}/gcnet_state_epoch_"
            + f"{epoch_to_load}_iter_{iter_to_load}.pth"
        )
        loss_train = np.array(checkpoint["loss_train_list"])
        loss_val = np.array(checkpoint["loss_val_list"])
        acc_train = np.array(checkpoint["acc_train_list"])
        acc_val = np.array(checkpoint["acc_val_list"])

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (14, 4)
        n_row = 1
        n_col = 2

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)

        # fig.text(0.5, 1.02, "Training loss", ha="center")
        # fig.text(-0.01, 0.5, "L1-loss", va="center", rotation=90)
        fig.text(0.5, -0.04, "Iteration x {}".format(self.eval_interval), ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        axes[0].plot(loss_train, linewidth=2)
        axes[0].plot(loss_val, linewidth=2)

        x_low = 0
        x_up = 210
        x_step = 25
        y_low = 0
        y_up = 20
        y_step = 2

        axes[0].set_ylabel("L1-loss")
        axes[0].set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_ylim(y_low, y_up)

        y_low = 0.5
        y_up = 1.05
        y_step = 0.1
        axes[1].plot(acc_train, linewidth=2)
        axes[1].plot(acc_val, linewidth=2)

        axes[1].set_ylabel("3-pixel accuracy")
        axes[1].set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[1].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[1].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_ylim(y_low, y_up)

        plt.legend(["train", "val"], loc="upper right")

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")

        plt.savefig(
            f"{self.save_dir}/loss_train_sceneflow_{epoch_to_load}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
