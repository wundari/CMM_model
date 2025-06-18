# %%


import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import math
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from jaxtyping import Float
from tqdm import tqdm

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

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# %%
@dataclass_json
@dataclass
class WaveletParameters:
    """
    Parameters for wavelet analysis.
    """

    # frequency channels
    # freq_channels: list[float] = field(
    #     default_factory=lambda: torch.arange(1.0, 9.0).tolist()
    # )
    freq_channels: list[float] = field(default_factory=lambda: [1, 2, 4, 8])

    # orientation channels
    # theta_channels: list[float] = field(
    #     default_factory=lambda: [0, math.pi / 4, math.pi / 2, math.pi * 3 / 4]
    # )
    theta_channels: list[float] = field(
        default_factory=lambda: torch.arange(0, math.pi, math.pi / 6).tolist()
    )

    sigma: float = 1.0  # standard deviation of the Gaussian envelope
    kernel_size: int = 11  # size of the kernel


# %%
class WaveletAnalysis:

    def __init__(self, WaveletParams: WaveletParameters):
        """
        Initialize the WaveletAnalysis class with wavelet parameters.

        :param WaveletParameters WaveletParams: Parameters for wavelet analysis.
        """
        self.freq_channels = WaveletParams.freq_channels  # frequency channels
        self.theta_channels = WaveletParams.theta_channels  # orientation channels
        self.sigma = WaveletParams.sigma  # standard deviation of the Gaussian envelope
        self.kernel_size = WaveletParams.kernel_size  # size of the kernel

    def _sigma_prefactor(self, bandwidth: float = 1.5) -> float:

        # http://www.cs.rug.nl/~imaging/simplecell.html
        # bandwidth = 1.5 => see Jenny Read Nature 2007

        prefactor = (
            (1.0 / math.pi)
            * math.sqrt(math.log(2.0) / 2.0)
            * (2.0**bandwidth + 1)
            / (2.0**bandwidth - 1)
        )

        return prefactor

    def gabor_kernel(
        self,
        f: float,
        theta: float,
        sigma: float,
        kernel_size: int,
    ):
        """
        Create a Gabor filter kernel.

        f <float>: Frequency of the Gabor filter.
        theta <float>: Orientation of the Gabor filter.
        sigma <float, default=1>: Standard deviation of the Gaussian envelope.
        kernel_size <int, default=11>: Size of the kernel.

        return: gabor_real <Float[Tensor, "kernel_size kernel_size"]>:
                    real parts of gabor filter
                gabor_imag <Float[Tensor, "kernel_size kernel_size"]>:
                    imaginary parts of gabor filter
        """

        # set up the axis
        n_stds = 3  # multiplier of sigma, denoting the range of the filter
        sigma2 = n_stds * self.sigma
        x = math.ceil(
            max(
                abs(sigma2 * math.cos(theta)),
                abs(sigma2 * math.sin(theta)),
                1,
            )
        )
        filter_axis = torch.linspace(-x, x, kernel_size)
        x, y = torch.meshgrid(
            filter_axis, filter_axis, indexing="xy"
        )  # [kernel_size, kernel_size]

        # rotate axis x and y
        x_rot = x * math.cos(theta) + y * math.sin(theta)  # [kernel_size, kernel_size]
        y_rot = -x * math.sin(theta) + y * math.cos(theta)

        # create gaussian envelope
        g_exp = (x_rot**2 + y_rot**2) / (2 * sigma**2)  # [kernel_size, kernel_size]

        # real terms, [kernel_size, kernel_size]
        gabor_real = torch.exp(-g_exp) * torch.cos(2 * math.pi * f * x_rot)
        # imaginary terms
        gabor_imag = torch.exp(-g_exp) * torch.sin(2 * math.pi * f * x_rot)

        # normalize so that sum-normalized = 0 and square-normalized = 1
        # gabor_real -= gabor_real.mean()  # sum-normalized to 0
        # gabor_real /= math.sqrt(
        #     torch.sum(gabor_real**2)
        # )  # square-normalized to 1 (sum of square = 1)
        gabor_real *= 1.0 / (2 * math.pi * sigma**2)

        # gabor_imag -= gabor_imag.mean()
        # gabor_imag /= math.sqrt(torch.sum(gabor_imag**2))
        gabor_imag *= 1.0 / (2 * math.pi * sigma**2)

        return gabor_real, gabor_imag

    def gabor(self):
        """
        Create a Gabor filter kernel for each frequency and orientation.

        return: gabor <Float[Tensor, "n_freq_channels n_theta_channels kernel_size kernel_size"]>:
                    Gabor filter kernels for each frequency and orientation.
                    complex-valued tensor with real and imaginary parts.
        """

        # create gabor filter kernels

        # allocate memory for gabor filter kernels
        n_freq_channels = len(self.freq_channels)
        n_theta_channels = len(self.theta_channels)
        gabor_real = torch.empty(
            n_freq_channels,
            n_theta_channels,
            self.kernel_size,
            self.kernel_size,
        )
        gabor_imag = torch.empty(
            n_freq_channels,
            n_theta_channels,
            self.kernel_size,
            self.kernel_size,
        )

        for f, freq in enumerate(self.freq_channels):
            for t, theta in enumerate(self.theta_channels):
                real, imag = self.gabor_kernel(
                    freq, theta, self.sigma, self.kernel_size
                )
                gabor_real[f, t] = real
                gabor_imag[f, t] = imag

        return gabor_real, gabor_imag

    def compute_wavelet_power(
        self,
        train_loader: DataLoader,
        gabor_real: Float[
            Tensor, "n_freq_channels n_theta_channels kernel_size kernel_size"
        ],
        gabor_imag: Float[
            Tensor, "n_freq_channels n_theta_channels kernel_size kernel_size"
        ],
    ) -> Float[Tensor, "n_freq_channels n_theta_channels"]:
        """
        Compute the wavelet power spectrum for the Gabor filter kernels.

        return: wavelet_power <Float[Tensor, "n_freq_channels n_theta_channels"]>:
                    Wavelet power spectrum for each frequency and orientation.
        """

        n_freq_channels = len(self.freq_channels)
        n_theta_channels = len(self.theta_channels)
        n_channels = n_freq_channels * n_theta_channels
        batch_size = train_loader.batch_size
        n_images = len(train_loader) * batch_size

        # reshape to follow the input format of conv2d:
        # see: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        gabor_real = gabor_real.view(
            n_channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )  # [n_channels, 1, kernel_size, kernel_size]
        gabor_imag = gabor_imag.view(
            n_channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )  # [n_channels, 1, kernel_size, kernel_size]

        # wavelet_power_real = torch.empty(n_channels, dtype=torch.float32)
        # wavelet_power_imag = torch.empty(n_channels, dtype=torch.float32)
        wavelet_power = torch.empty((n_images, n_channels), dtype=torch.float32)

        tepoch = tqdm(train_loader, unit="batch")
        for i, (inputs_left, _, _) in enumerate(tepoch):

            id_start = i * batch_size
            id_end = id_start + batch_size

            # [n_batch, n_channels, height, width]
            # inputs_left, inputs_right, disps = next(iter(train_loader))

            # convert to grayscale
            img = transforms.Grayscale()(inputs_left)

            # real part
            conv_real = F.conv2d(
                img, gabor_real
            )  # [n_batch, n_channels, height, width]
            # power = (conv_real**2).mean(dim=(2, 3))  # [n_batch, n_channels]
            # wavelet_power_real += power.sum(dim=0)

            # imaginary part
            conv_imag = F.conv2d(img, gabor_imag)
            # power = (conv_imag**2).mean(dim=(2, 3))  # [n_batch, n_channels]
            # wavelet_power_imag += power.sum(dim=0)

            # power
            power = torch.sqrt(conv_real**2 + conv_imag**2)
            wavelet_power[id_start:id_end] = power.mean(
                dim=(2, 3)
            )  # average across pixels

        # reshape
        wavelet_power = wavelet_power.view(
            n_images, n_freq_channels, n_theta_channels
        )  # [n_images, n_freq_channels, n_theta_channels]

        return wavelet_power

    def plotHeat_wavelet_power(
        self,
        wavelet_power: Float[Tensor, "kernel_size kernel_size"],
        sceneflow_type: str,
        save_flag: bool,
    ) -> None:
        """
        plot heatmap wavelet_power

        Parameters
        ----------
        wavelet_power: Float[Tensor, "kernel_size kernel_size"]
            wavelet power spectrum
        sceneflow_type: str
            sceneflow type, e.g. "driving", "flying", "monkaa"
        save_flag: bool
            an integer to indicate saving the plot or not.
            yes -> 1
            no -> 0

        Returns
        -------
        None.

        """

        # max-normalize to 1
        wavelet_norm = wavelet_power / wavelet_power.max()

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="bright")

        # estimate v_min and v_max for cbar
        v_min = 0.0
        v_max = 1.0
        # v_max = 1.0

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.05,
            "Wavelet Power Spectrum \n" + sceneflow_type,
            ha="center",
        )
        fig.text(-0.0, 0.5, "Spatial frequency", va="center", rotation=90)
        fig.text(0.5, -0.1, "Orientation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        freq_channels = [round(f, 2) for f in self.freq_channels]
        theta_channels = [round(t * 180 / math.pi, 2) for t in self.theta_channels]

        cmap = "jet"
        sns.heatmap(
            wavelet_norm,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=theta_channels,
            yticklabels=freq_channels,
            ax=axes,
        )
        # axes.set_title("Squared BEM", pad=20)

        if save_flag == 1:
            fig.savefig(
                f"results/sceneflow/{sceneflow_type}/PlotHeat_wavelet_power_{sceneflow_type}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
