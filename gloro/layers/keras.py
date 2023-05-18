"""
This file contains `GloroLayer` versions of standard keras layers that are
compatible with GloRo.
"""
import tensorflow as tf

from tensorflow.keras.layers import AveragePooling2D as KerasAveragePooling2D
from tensorflow.keras.layers import Conv2D as KerasConv2D
from tensorflow.keras.layers import Dense as KerasDense
from tensorflow.keras.layers import Flatten as KerasFlatten
from tensorflow.keras.layers import MaxPooling2D as KerasMaxPooling2D
from tensorflow.keras.layers import ReLU as KerasReLU
from tensorflow.keras.layers import Lambda as KerasLambda

from gloro.layers.base import GloroLayer
from gloro.lc import LipschitzComputationStrategy
from gloro.lc import PowerMethod


class Dense(KerasDense, GloroLayer):
    def __init__(self, *args, lc_strategy=None, **kwargs):
        # TODO: make sure `activation` isn't passed as an arg.
        super().__init__(*args, **kwargs)

        self._lc_strategy = LipschitzComputationStrategy.get(lc_strategy or "power")

    @property
    def lc_strategy(self):
        return self._lc_strategy

    def build(self, input_shape):
        super().build(input_shape)

        self.lc_strategy.build(self.kernel)

    def lipschitz(self):
        return self.lc_strategy.compute()

    def lipschitz_inf(self):
        w = self.kernel
        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=1, keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    def bound(self, input_bound):
        lc = self.lipschitz_inf()
        lower_bound, upper_bound = input_bound
        return (lower_bound * lc, upper_bound * lc)


class Conv2D(KerasConv2D, GloroLayer):
    def __init__(self, *args, lc_strategy=None, **kwargs):
        # TODO: make sure `activation` isn't passed as an arg.
        super().__init__(*args, **kwargs)

        self._lc_strategy = LipschitzComputationStrategy.get(lc_strategy or "power")

    @property
    def lc_strategy(self):
        return self._lc_strategy

    def build(self, input_shape):
        super().build(input_shape)

        self.lc_strategy.build(
            self.kernel,
            input_shape=input_shape,
            strides=self.strides,
            padding=self.padding.upper(),
        )

    def lipschitz(self):
        return self.lc_strategy.compute()

    def lipschitz_inf(self):
        w = self.kernel
        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=[0, 1, 2], keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    # TODO: bias
    def bound(self, input_bound):
        lc = self.lipschitz_inf()
        lower_bound, upper_bound = input_bound
        return (lower_bound * lc, upper_bound * lc)


class AveragePooling2D(KerasAveragePooling2D, GloroLayer):
    def build(self, input_shape):
        super().build(input_shape)

        # Average pooling can be thought of as a convolution where the kernel is
        # fixed to be 1 / pool_area at each entry.
        self._lc = (
            PowerMethod(100)
            .build(
                tf.eye(input_shape[-1])[None, None]
                * (tf.ones(self.pool_size)[:, :, None, None])
                / (self.pool_size[0] * self.pool_size[1]),
                input_shape=input_shape,
                strides=self.strides,
                padding=self.padding.upper(),
            )
            .compute()
        )

    def lipschitz(self):
        return self._lc

    def lipschitz_inf(self):
        w = (
            tf.eye(input_shape[-1])[None, None]
            * (tf.ones(self.pool_size)[:, :, None, None])
            / (self.pool_size[0] * self.pool_size[1])
        )

        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=[0, 1, 2], keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    def bound(self, input_bound):
        return input_bound


class Flatten(KerasFlatten, GloroLayer):
    def lipschitz(self):
        return 1.0

    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_bound):
        return input_bound


class MaxPooling2D(KerasMaxPooling2D, GloroLayer):
    def lipschitz(self):
        return 1.0

    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_bound):
        return input_bound


class ReLU(KerasReLU, GloroLayer):
    def lipschitz(self):
        return 1.0

    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_bound):
        _, upper_bound = input_bound
        return (0, upper_bound)


# TODO: Consider input shape in propagate_error
class ApproxReLU(KerasLambda, GloroLayer):
    def __init__(self, alpha, B=1, **kwargs):
        self.alpha = alpha
        self.B = B
        self.range = (-B, B)

        super().__init__(self.relu_approx, **kwargs)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_B(self, B):
        self.B = B
        self.range = (-B, B)

    # TODO
    def relu_approx(self, x):
        return tf.where(x >= 0, x, self.alpha * x)

    def lipschitz(self):
        return 1.0

    def lipschitz_inf(self):
        return 1.0

    def approx_error(self):
        return 2 ** (-self.alpha) * self.B

    def propagate_error(self, error):
        return error + self.approx_error()

    def bound(self, input_bound):
        _, upper_bound = input_bound
        return (0, upper_bound)
