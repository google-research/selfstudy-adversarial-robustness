# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module with definition of baseline networks."""

import tensorflow as tf
import math


class AllConvModel(tf.Module):
    """All convolutional network architecture."""

    def __init__(self, num_classes, num_filters, input_shape, activation=tf.nn.leaky_relu):
        super().__init__()
        conv_args = dict(
            activation=activation,
            kernel_size=3,
            padding='same')
        self.layers = []
        log_resolution = int(round(
            math.log(input_shape[0]) / math.log(2)))
        for scale in range(log_resolution - 2):
            self.layers.append(tf.keras.layers.Conv2D(num_filters << scale, **conv_args))
            self.layers.append(tf.keras.layers.Conv2D(num_filters << (scale + 1), **conv_args))
            self.layers.append(tf.keras.layers.AveragePooling2D((2, 2)))
        self.layers.append(tf.keras.layers.Conv2D(num_classes, kernel_size=3, padding='same'))
        self.layers.append(tf.keras.layers.GlobalAveragePooling2D())
        self.layers.append(tf.keras.layers.Softmax())

        # call inference to instantiate variables
        self(tf.zeros((1,) + tuple(input_shape)), training=False)

    def __call__(self, x, training=False):
        del training  # ignore training argument since don't have batch norm
        for layer in self.layers:
            x = layer(x)
        return x


class FullyConnectedModel(tf.Module):
    """A fully connected network architecture."""

    def __init__(self, sizes, activation=tf.nn.leaky_relu):
        super().__init__()
        self.layers = []
        self.layers.append(tf.keras.layers.Flatten())
        for dim in sizes[1:]:
            self.layers.append(tf.keras.layers.Dense(dim,
                                                     activation=activation))

        self.layers.append(tf.keras.layers.Softmax())

        # call inference to instantiate variables
        self(tf.zeros((1,) + tuple(sizes[:1])), training=False)

    def __call__(self, x, training=False):
        del training  # ignore training argument since don't have batch norm
        for layer in self.layers:
            x = layer(x)
        return x
    
try:
    # If PyTorch is installed then import PyTorch implementation of AllConvModel
    from common.pytorch_nets import AllConvModelTorch
except:
    # If PyTorch is not available then declare placeholder for AllConvModelTorch
    class AllConvModelTorch:
        pass
