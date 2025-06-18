# %%
import torch
import numpy as np
from numba import njit, prange


# %%
@njit(parallel=True)
def filter2d(img, filt):
    h_img, w_img = img.shape
    h_filt, w_filt = filt.shape

    n_row = (h_img // h_filt - 1) * h_filt
    n_col = (w_img // w_filt - 1) * w_filt
    img_filt = np.zeros((n_row, n_col), dtype=np.float32)

    filt_flat = filt.ravel()

    for i in range(n_row):
        for j in prange(n_col):
            temp = img[i : i + h_filt, j : j + w_filt].ravel()

            sum_prod = 0.0
            for k in range(filt_flat.size):
                sum_prod += temp[k] * filt_flat[k]

            img_filt[i, j] = sum_prod

    return img_filt


# %%
@njit(parallel=True)
def create_meshgrid(size):
    mesh_x = np.zeros((size, size), dtype=np.float32)
    mesh_y = np.zeros((size, size), dtype=np.float32)

    range_arr = np.arange(size)
    for i in prange(size):
        mesh_x[i] = range_arr
        mesh_y[:, i] = range_arr

    return mesh_x, mesh_y


# %%
@njit(fastmath=True)
def gauss2d(sigma, size):
    x, y = create_meshgrid(size)

    center = size // 2
    exp_term = (x - center) ** 2 + (y - center) ** 2
    filt = np.exp(-0.5 * exp_term / (sigma**2))

    # normalize to 1
    filt /= filt.sum()

    return filt


# %%
@njit(parallel=True)
def compute_binocular_correlation(img_left_filt, img_right_filt):
    height, width = img_left_filt.shape
    half_width = width // 2

    # compute image mean and std for the entire images
    mean_left, std_left = img_left_filt.mean(), img_left_filt.std()
    mean_right, std_right = img_right_filt.mean(), img_right_filt.std()

    # standardize image
    left_standardized = (img_left_filt - mean_left) / std_left
    right_standardized = (img_right_filt - mean_right) / std_right

    bino_corr = np.zeros(width, dtype=np.float32)
    for i in prange(-half_width, half_width):
        if i <= 0:
            # slice left image
            signal_left = left_standardized[:, 0 : width + i]

            # slice right image
            signal_right = right_standardized[:, -i:width]

        elif i > 0:
            # slice left image
            signal_left = left_standardized[:, i:width]

            # slice right image
            signal_right = right_standardized[:, 0 : width - i]

        # compute binocular correlation
        bino_corr[i + half_width] = np.dot(
            signal_left.ravel(), signal_right.ravel()
        ) / (height * (width - np.abs(i)))

    return bino_corr


# @njit(parallel=True)
# def compute_bino_corr_slide(img_left_filt, img_right_filt):
#     height, width = img_left_filt.shape
#     half_width = width // 2

#     # Precompute mean and std
#     mean_left, std_left = img_left_filt.mean(), img_left_filt.std()
#     mean_right, std_right = img_right_filt.mean(), img_right_filt.std()

#     # Standardize images
#     left_standardized = (img_left_filt - mean_left) / std_left
#     right_standardized = (img_right_filt - mean_right) / std_right

#     # compute cross-correlation between left and right RFs
#     bino_corr = np.empty(width, dtype=np.float32)

#     # Use preallocated buffer to avoid creating new slices in each iteration
#     # buffer = np.empty((height, width), dtype=np.float32)
#     buffer_left = np.empty(height * width, dtype=np.float32)
#     buffer_right = np.empty(height * width, dtype=np.float32)

#     # for i in prange(-half_width, half_width):
#     for i in prange(-half_width, half_width):
#         n = height * (width - abs(i))
#         if i <= 0:
#             # signal_left = left_standardized[:, : width + i]
#             # signal_right = right_standardized[:, -i:]
#             # corr = (signal_left.ravel() * signal_right.ravel()).sum() / (height * (width - abs(i)))

#             # buffer[:, : width + i] = (
#             #     left_standardized[:, : width + i] * right_standardized[:, -i:]
#             # )
#             # sum_value = buffer[:, : width + i].sum()

#             buffer_left[:n] = left_standardized[:, : width + i].ravel()
#             buffer_right[:n] = right_standardized[:, -i:].ravel()
#             sum_value = np.dot(buffer_left[:n], buffer_right[:n])
#         else:
#             # buffer[:, : width - i] = (
#             #     left_standardized[:, i:] * right_standardized[:, : width - i]
#             # )
#             # sum_value = buffer[:, : width - i].sum()

#             buffer_left[:n] = left_standardized[:, i:].ravel()
#             buffer_right[:n] = right_standardized[:, : width - i].ravel()
#             sum_value = np.dot(buffer_left[:n], buffer_right[:n])

#         # bino_corr[i + half_width] = sum_value / (height * (width - abs(i)))
#         bino_corr[i + half_width] = sum_value / n

#     return bino_corr


@njit(parallel=True)
def compute_bino_corr_slide(img_left_filt, img_right_filt):
    height, width = img_left_filt.shape
    half_width = width // 2

    # Precompute mean and std
    mean_left, std_left = img_left_filt.mean(), img_left_filt.std()
    mean_right, std_right = img_right_filt.mean(), img_right_filt.std()

    # Standardize images
    left_standardized = (img_left_filt - mean_left) / std_left
    right_standardized = (img_right_filt - mean_right) / std_right

    # Use preallocated buffer to avoid creating new slices in each iteration
    # buffer = np.empty((height, width), dtype=np.float32)
    buffer_left = np.empty(height * width, dtype=np.float32)
    buffer_right = np.empty(height * width, dtype=np.float32)

    # cross-correlation between left and right RFs
    bino_corr = np.empty(width, dtype=np.float32)

    # for i in prange(-half_width, half_width):
    # from 0 disparity to the right. Disparity at 0 is at the center of each RF
    # from 0 to the left (negative disparity)
    # for i in prange(0, -half_width - 1, -1):
    for i in prange(-half_width - 1, 0, 1):
        n = height * (width + i)

        # left RF is the reference point
        buffer_left[:n] = left_standardized[:, : (width + i)].ravel()
        buffer_right[:n] = right_standardized[:, -i:].ravel()

        # standardize
        # mean_left = buffer_left[:n].mean()
        # std_left = buffer_left[:n].std()
        # mean_right = buffer_right[:n].mean()
        # std_right = buffer_right[:n].std()

        # buffer_left[:n] -= mean_left
        # buffer_left[:n] /= std_left
        # buffer_right[:n] -= mean_right
        # buffer_right[:n] /= std_right

        bino_corr[half_width + i] = np.dot(buffer_left[:n], buffer_right[:n]) / n

    # from 0 to the right (positive disparity)
    for i in prange(half_width + 1):
        n = height * (width - i)

        # left RF is the reference point
        buffer_left[:n] = left_standardized[:, i:].ravel()
        buffer_right[:n] = right_standardized[:, : width - i].ravel()

        # standardize
        # mean_left = buffer_left[:n].mean()
        # std_left = buffer_left[:n].std()
        # mean_right = buffer_right[:n].mean()
        # std_right = buffer_right[:n].std()

        # buffer_left[:n] -= mean_left
        # buffer_left[:n] /= std_left
        # buffer_right[:n] -= mean_right
        # buffer_right[:n] /= std_right

        bino_corr[half_width + i] = np.dot(buffer_left[:n], buffer_right[:n]) / n

    return bino_corr


# %%
@njit(parallel=True)
def compute_bino_corr_pearson_slide(img_left_filt, img_right_filt):
    """
    Compute the Pearson correlation between the left and right preferred inputs
    of neurons. The computation is done like in convolution operation in which
    the left image slides on the right image pixel by pixel.
    in each pixel step, compute the Pearson correlation between the shifted
    inputs.

    Args:
        img_left_filt (_type_): _description_
        img_right_filt (_type_): _description_

    Returns:
        _type_: _description_
    """
    height, width = img_left_filt.shape
    half_width = width // 2

    # Use preallocated buffer to avoid creating new slices in each iteration
    buffer_left = np.empty(height * width, dtype=np.float32)
    buffer_right = np.empty(height * width, dtype=np.float32)
    bino_corr = np.zeros(width, dtype=np.float32)

    for i in prange(-half_width, half_width):
        n = height * (width - abs(i))
        if i <= 0:
            buffer_left[:n] = img_left_filt[:, : width + i].ravel()
            buffer_right[:n] = img_right_filt[:, -i:].ravel()

        else:
            buffer_left[:n] = img_left_filt[:, i:].ravel()
            buffer_right[:n] = img_right_filt[:, width - i :].ravel()

        # standardize
        mean_left = buffer_left[:n].mean()
        std_left = buffer_left[:n].std()
        mean_right = buffer_right[:n].mean()
        std_right = buffer_right[:n].std()

        buffer_left[:n] -= mean_left
        buffer_left[:n] /= std_left
        buffer_right[:n] -= mean_right
        buffer_right[:n] /= std_right

        # compute correlation
        bino_corr[i + half_width] = np.dot(buffer_left[:n], buffer_right[:n]) / n

    return bino_corr


# %%
@njit(fastmath=True)
def compute_bino_corr_pearson(img_left_filt, img_right_filt):
    height, width = img_left_filt.shape

    # compute mean and std
    mean_left, std_left = img_left_filt.mean(), img_left_filt.std()
    mean_right, std_right = img_right_filt.mean(), img_right_filt.std()

    # Standardize images
    left_standardized = (img_left_filt - mean_left) / std_left
    right_standardized = (img_right_filt - mean_right) / std_right

    # compute correlation
    bino_corr = np.dot(left_standardized.ravel(), right_standardized.ravel()) / (
        height * width
    )

    return bino_corr


# %%
@njit(parallel=True)
def compute_corr_neuron_pref_input(img_left_layer, img_right_layer, filt):
    """
    Compute the Pearson correlation between the left and right preferred inputs
    of neurons in each feature channel and disparity channel contained in a DNN layer.

    Args:
        img_left_layer (_type_): _description_
        img_right_layer (_type_): _description_
        filt (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_feature_channel, n_disp_channel = img_left_layer.shape[0:2]

    corr_neuron_pref_input = np.zeros(
        (n_feature_channel, n_disp_channel), dtype=np.float32
    )

    for i in range(n_feature_channel):
        for j in range(n_disp_channel):
            img_left = img_left_layer[i, j]
            img_right = img_right_layer[i, j]

            # filter image
            img_left_filt = filter2d(img_left, filt)
            img_right_filt = filter2d(img_right, filt)

            # compute pearson correlation between left and right preferred inputs
            bino_corr = compute_bino_corr_pearson(img_left_filt, img_right_filt)
            corr_neuron_pref_input[i, j] = bino_corr

            # bino_corr = compute_bino_corr_slide(img_left_filt, img_right_filt)
            # corr_max = np.abs(bino_corr.max())
            # corr_min = np.abs(bino_corr.min())
            # if corr_max > corr_min:
            #     corr_neuron_pref_input[i, j] = bino_corr.max()
            # else:
            #     corr_neuron_pref_input[i, j] = bino_corr.min()

    return corr_neuron_pref_input


# %%
@njit(parallel=True)
def compute_corr_channel_pref_input(img_left_layer, img_right_layer, filt):
    """
    compute the Pearson correlation between the left and right preferred inputs
    of channel

    Args:
        img_left_layer (_type_): _description_
        img_right_layer (_type_): _description_
        filt (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_feature_channel, n_disp_channel = img_left_layer.shape[0:2]

    corr_channel_pref_input = np.zeros(
        (n_feature_channel, n_disp_channel), dtype=np.float32
    )

    for i in range(n_feature_channel):
        for j in range(n_disp_channel):
            img_left = img_left_layer[i, j]
            img_right = img_right_layer[i, j]

            # filter image
            img_left_filt = filter2d(img_left, filt)
            img_right_filt = filter2d(img_right, filt)
            # bino_corr = compute_binocular_correlation(img_left_filt, img_right_filt)
            bino_corr = compute_bino_corr_slide(img_left_filt, img_right_filt)
            corr_max = np.abs(bino_corr.max())
            corr_min = np.abs(bino_corr.min())
            if corr_max > corr_min:
                corr_channel_pref_input[i, j] = bino_corr.max()
            else:
                corr_channel_pref_input[i, j] = bino_corr.min()

    return corr_channel_pref_input


# %%


@njit(parallel=True)
def compute_channel_correlation_pearson(img_left_layer, img_right_layer, filt):
    n_feature_channel, n_disp_channel = img_left_layer.shape[0:2]

    corr_channel = np.zeros((n_feature_channel, n_disp_channel), dtype=np.float32)

    for i in range(n_feature_channel):
        for j in range(n_disp_channel):
            img_left = img_left_layer[i, j]
            img_right = img_right_layer[i, j]

            # filter image
            img_left_filt = filter2d(img_left, filt)
            img_right_filt = filter2d(img_right, filt)

            # compute pearson correlation
            bino_corr = compute_bino_corr_pearson(img_left_filt, img_right_filt)

            corr_channel[i, j] = bino_corr

    return corr_channel


# %%
@njit(parallel=True)
def compute_layer_correlation(img_left_layer, img_right_layer, filt):
    n_layer = img_left_layer.shape[0]

    corr_layer = np.zeros(n_layer, dtype=np.float32)

    for i in prange(n_layer):
        img_left = img_left_layer[i]
        img_right = img_right_layer[i]

        # filter image
        img_left_filt = filter2d(img_left, filt)
        img_right_filt = filter2d(img_right, filt)
        # bino_corr = compute_binocular_correlation(img_left_filt, img_right_filt)
        bino_corr = compute_bino_corr_pearson(img_left_filt, img_right_filt)
        # corr_max = np.abs(bino_corr.max())
        # corr_min = np.abs(bino_corr.min())
        # if corr_max > corr_min:
        #     corr_layer[i] = bino_corr.max()
        # else:
        #     corr_layer[i] = bino_corr.min()
        corr_layer[i] = bino_corr

    return corr_layer


# %%
# @njit
def extract_activation(activations):
    n_feature_channel, n_disp_channel = activations.shape[1:3]

    # get neuron's coordinate at the center
    row_center = activations.shape[-2] // 2
    col_center = activations.shape[-1] // 2

    # activation_extracted = np.empty(
    #     n_feature_channel * n_disp_channel, dtype=np.float32
    # )
    activation_extracted = torch.empty(
        n_feature_channel * n_disp_channel, dtype=torch.float32
    )

    activation_extracted[:] = activations[0, :, :, row_center, col_center].ravel()

    return activation_extracted


# %%
@njit(fastmath=True)
def compute_pearson_corr_1d(x, y):
    # standardize
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()

    x_standardized = (x - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    # compute correlation
    corr = np.dot(x_standardized, y_standardized) / (len(x) * len(y))

    return corr
