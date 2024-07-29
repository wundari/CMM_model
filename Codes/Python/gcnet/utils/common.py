import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# %%
def get_neuron_pos(
    H: int,
    W: int,
    neuron_row_index: Optional[int] = None,
    neuron_col_index: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Args:

        H (int) The height
        W (int) The width
        neuron_row_index (int, optional): Optionally specify and exact row location of the target neuron.
            If set to None, then the center row location will be used.
            Default: None
        neuron_col_index (int, optional): Optionally specify and exact col location of the target neuron.
            If set to None, then the center col location will be used.
            Default: None

    Return:
        Tuple[_row, _col] (Tuple[int, int]): The x and y dimensions of the neuron.
    """
    if neuron_row_index is None:
        _row = H // 2
    else:
        assert neuron_row_index < H
        _row = neuron_row_index

    if neuron_col_index is None:
        _col = W // 2
    else:
        assert neuron_col_index < W
        _col = neuron_col_index
    return _row, _col
