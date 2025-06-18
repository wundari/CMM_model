# %% load necessary modules
from engine.engine_base import Engine

import torch
import numpy as np
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

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% setup gc-net
# network parameters
params_network = {
    "h_bg": 256,
    "w_bg": 512,
    "maxdisp": 192,
}

# training parameters
c_disp_shift = 1.5
flip_input = 0  # set to 1 if flip the input (right2left), use right disparity image as ground truth
params_train = {
    "sceneflow_type": "monkaa",
    "learning_rate": 6e-4,
    "eval_iter": 25,
    "eval_interval": 200,
    "batch_size": 2,
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

# %% prepare dataset
train_loader, val_loader = train_engine.prepare_dataset()

# %% train
train_engine.train(train_loader, val_loader)

# %% plot learning curve
epoch_to_load = 10
iter_to_load = 89561
save_flag = 1
train_engine.plotLine_learning_curve(epoch_to_load, iter_to_load, save_flag)

# %%

