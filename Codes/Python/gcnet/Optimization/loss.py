import functools
import operator
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from utils.common import get_neuron_pos
from utils.typing import ModuleOutputMapping


# %%
def _make_arg_str(arg: Any) -> str:
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return arg[:15] + "..." if too_big else arg


class Loss(ABC):
    """
    Abstract Class to describe loss.
    Note: All Loss classes should expose self.target for hooking by
    InputOptimization
    """

    def __init__(self) -> None:
        super(Loss, self).__init__()

    @abstractproperty
    def target(self) -> Union[nn.Module, List[nn.Module]]:
        pass

    @abstractmethod
    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return self.__name__

    def __neg__(self) -> "CompositeLoss":
        return module_op(self, None, operator.neg)

    def __add__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.add)

    def __sub__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.sub)

    def __mul__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.mul)

    def __truediv__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.truediv)

    def __pow__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.pow)

    def __radd__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__add__(other)

    def __rsub__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__neg__().__add__(other)

    def __rmul__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        if isinstance(other, (int, float)):

            def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
                return operator.truediv(other, torch.mean(self(module)))

            name = self.__name__
            target = self.target
        elif isinstance(other, Loss):
            # This should never get called because __div__ will be called instead
            pass
        else:
            raise TypeError(
                "Can only apply math operations with int, float or Loss. Received type "
                + str(type(other))
            )
        return CompositeLoss(loss_fn, name=name, target=target)

    def __rpow__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        if isinstance(other, (int, float)):

            def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
                return operator.pow(other, torch.mean(self(module)))

            name = self.__name__
            target = self.target
        elif isinstance(other, Loss):
            # This should never get called because __pow__ will be called instead
            pass
        else:
            raise TypeError(
                "Can only apply math operations with int, float or Loss. Received type "
                + str(type(other))
            )
        return CompositeLoss(loss_fn, name=name, target=target)


def module_op(
    self: Loss, other: Union[None, int, float, Loss], math_op: Callable
) -> "CompositeLoss":
    """
    This is a general function for applying math operations to Losses
    """
    if other is None and math_op == operator.neg:

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(self(module))

        name = self.__name__
        target = self.target
    elif isinstance(other, (int, float)):

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(self(module), other)

        name = self.__name__
        target = self.target
    elif isinstance(other, Loss):
        # We take the mean of the output tensor to resolve shape mismatches
        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(torch.mean(self(module)), torch.mean(other(module)))

        name = f"Compose({', '.join([self.__name__, other.__name__])})"

        # ToDo: Refine logic for self.target handling
        target = (self.target if isinstance(self.target, list) else [self.target]) + (
            other.target if isinstance(other.target, list) else [other.target]
        )

        # Filter out duplicate targets
        target = list(dict.fromkeys(target))
    else:
        raise TypeError(
            "Can only apply math operations with int, float or Loss. Received type "
            + str(type(other))
        )
    return CompositeLoss(loss_fn, name=name, target=target)


class BaseLoss(Loss):
    def __init__(
        self,
        target: Union[nn.Module, List[nn.Module]] = [],
        batch_index: Optional[int] = None,
    ) -> None:
        super(BaseLoss, self).__init__()
        self._target = target
        if batch_index is None:
            self._batch_index = (None, None)
        else:
            self._batch_index = (batch_index, batch_index + 1)

    @property
    def target(self) -> Union[nn.Module, List[nn.Module]]:
        return self._target

    @property
    def batch_index(self) -> Tuple:
        return self._batch_index


class CompositeLoss(BaseLoss):
    def __init__(
        self,
        loss_fn: Callable,
        name: str = "",
        target: Union[nn.Module, List[nn.Module]] = [],
    ) -> None:
        super(CompositeLoss, self).__init__(target)
        self.__name__ = name
        self.loss_fn = loss_fn

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        return self.loss_fn(targets_to_values)


def loss_wrapper(cls: Any) -> Callable:
    """
    Primarily for naming purposes.
    """

    @functools.wraps(cls)
    def wrapper(*args, **kwargs) -> object:
        obj = cls(*args, **kwargs)
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        obj.__name__ = cls.__name__ + args_str
        return obj

    return wrapper


@loss_wrapper
class NeuronActivation(BaseLoss):
    """
    This loss maximizes the activations of a target neuron in the specified channel
    from the specified layer. This loss is useful for determining the type of features
    that excite a neuron, and thus is often used for circuits and neuron related
    research.

    Args:
        target (nn.Module):  The layer to containing the channel to optimize for.
        channel_index (int):  The index of the channel to optimize for.
        x (int, optional):  The x coordinate of the neuron to optimize for. If
            unspecified, defaults to center, or one unit left of center for even
            lengths.
        y (int, optional):  The y coordinate of the neuron to optimize for. If
            unspecified, defaults to center, or one unit up of center for even
            heights.
        batch_index (int, optional):  The index of the image to optimize if we
            optimizing a batch of images. If unspecified, defaults to all images
            in the batch.
    """

    def __init__(
        self,
        target: nn.Module,
        feature_channel_index: int,
        disp_channel_index: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.feature_channel_index = feature_channel_index
        self.disp_channel_index = disp_channel_index
        self.x = x
        self.y = y

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        assert self.feature_channel_index < activations.shape[1]
        _x, _y = get_neuron_pos(
            activations.size(-2),  # height
            activations.size(-1),  # width
            self.x,  # row index of target neuron
            self.y,  # col index of target neuron
        )

        if len(activations.shape) == 4:  # [batch, feature_channel, height, width]
            activation = activations[
                self.batch_index[0] : self.batch_index[1],
                self.feature_channel_index,
                _x : _x + 1,
                _y : _y + 1,
            ]

        elif (
            len(activations.shape) == 5
        ):  # [batch, feature_channel, disp_channel, height, width]
            activation = activations[
                self.batch_index[0] : self.batch_index[1],
                self.feature_channel_index,
                self.disp_channel_index,
                _x : _x + 1,
                _y : _y + 1,
            ]

        return activation


@loss_wrapper
class ChannelActivation(BaseLoss):
    """
    This loss maximizes the activations of a target neuron in the specified channel
    from the specified layer. This loss is useful for determining the type of features
    that excite a neuron, and thus is often used for circuits and neuron related
    research.

    Args:
        target (nn.Module):  The layer to containing the channel to optimize for.

        channel_index (int):  The index of the channel to optimize for.

        batch_index (int, optional):  The index of the image to optimize if we
            optimizing a batch of images. If unspecified, defaults to all images
            in the batch.
    """

    def __init__(
        self,
        target: nn.Module,
        feature_channel_index: int,
        disp_channel_index: int,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.feature_channel_index = feature_channel_index
        self.disp_channel_index = disp_channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        # [batch, feature_channel, disp_channel, height, width]
        activations = targets_to_values[self.target]
        assert activations is not None
        assert self.feature_channel_index < activations.shape[1]

        if len(activations.shape) == 4:  # [batch, feature_channel, height, width]
            activation = activations[
                self.batch_index[0] : self.batch_index[1], self.feature_channel_index
            ]

        elif (
            len(activations.shape) == 5
        ):  # [batch, feature_channel, disp_channel, height, width]
            activation = activations[
                self.batch_index[0] : self.batch_index[1],
                self.feature_channel_index,
                self.disp_channel_index,
            ]

        return activation


@loss_wrapper
class FeatureChannelActivation(BaseLoss):
    """
    This loss maximizes the activations of a target neuron in the specified channel
    from the specified layer. This loss is useful for determining the type of features
    that excite a neuron, and thus is often used for circuits and neuron related
    research.

    Args:
        target (nn.Module):  The layer to containing the channel to optimize for.

        channel_index (int):  The index of the channel to optimize for.

        batch_index (int, optional):  The index of the image to optimize if we
            optimizing a batch of images. If unspecified, defaults to all images
            in the batch.
    """

    def __init__(
        self,
        target: nn.Module,
        feature_channel_index: int,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.feature_channel_index = feature_channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        # [batch, feature_channel, disp_channel, height, width]
        activations = targets_to_values[self.target]

        assert activations is not None
        assert self.feature_channel_index < activations.shape[1]
        assert len(activations.shape) == 5

        activation = activations[
            self.batch_index[0] : self.batch_index[1], self.feature_channel_index
        ]

        return activation


@loss_wrapper
class DisparityChannelActivation(BaseLoss):
    """
    This loss maximizes the activations of a target neuron in the specified channel
    from the specified layer. This loss is useful for determining the type of features
    that excite a neuron, and thus is often used for circuits and neuron related
    research.

    Args:
        target (nn.Module):  The layer to containing the channel to optimize for.

        channel_index (int):  The index of the channel to optimize for.

        batch_index (int, optional):  The index of the image to optimize if we
            optimizing a batch of images. If unspecified, defaults to all images
            in the batch.
    """

    def __init__(
        self,
        target: nn.Module,
        disp_channel_index: int,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.disp_channel_index = disp_channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        # [batch, feature_channel, disp_channel, height, width]
        activations = targets_to_values[self.target]

        assert activations is not None
        assert len(activations.shape) == 5
        activation = activations[
            self.batch_index[0] : self.batch_index[1],
            :,
            self.disp_channel_index,
        ]

        return activation


@loss_wrapper
class LayerActivation(BaseLoss):
    """
    This loss maximizes the activations of a target neuron in the specified channel
    from the specified layer. This loss is useful for determining the type of features
    that excite a neuron, and thus is often used for circuits and neuron related
    research.

    Args:
        target (nn.Module):  The layer to containing the channel to optimize for.

        channel_index (int):  The index of the channel to optimize for.

        batch_index (int, optional):  The index of the image to optimize if we
            optimizing a batch of images. If unspecified, defaults to all images
            in the batch.
    """

    def __init__(
        self,
        target: nn.Module,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        # [batch, feature_channel, disp_channel, height, width]
        activations = targets_to_values[self.target]
        activation = activations[self.batch_index[0] : self.batch_index[1]]

        return activation**2
