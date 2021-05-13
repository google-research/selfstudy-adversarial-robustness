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

"""Model of the defense 0."""

import numpy as np
import tensorflow as tf

from common.framework import DefenseModel, get_checkpoint_abs_path
from common.networks import AllConvModel, AllConvModelTorch
import common.utils as utils

MODEL_PATH = 'checkpoints/baseline/final_checkpoint-1'


class RandomDropModel(AllConvModel):
    def __call__(self, x, training=False):
        del training
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                _,a,b,c = x.shape
                p = tf.abs(x)/tf.reduce_sum(tf.abs(x), axis=(1,2,3), keepdims=True)
                p_keep = 1-tf.exp(-a*b*c / 3 * p)
                keep = tf.random.uniform(p_keep.shape)<p_keep
                x = tf.cast(keep, tf.float32)*x/p_keep
        return x


class Defense(DefenseModel):

    def __init__(self):
        self.convnet = RandomDropModel(num_classes=10,
                                       num_filters=64,
                                       input_shape=[32, 32, 3])
        tf.train.Checkpoint(model=self.convnet).restore(
            get_checkpoint_abs_path(MODEL_PATH))
        self.to_tensor = lambda x: x

    def classify(self, x):
        preds = [utils.to_numpy(self.convnet(self.to_tensor(x))) for _ in range(10)]
        return np.mean(preds, axis=0)


class RandomDropModelTorch(AllConvModelTorch):
    def __call__(self, x, training=False):
        import torch
        del training
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, torch.nn.Conv2d):
                _,a,b,c = x.shape
                p = torch.abs(x)/torch.sum(torch.abs(x), axis=(1,2,3), keepdims=True)
                p_keep = 1-torch.exp(-a*b*c / 3 * p)
                keep = torch.rand(p_keep.shape)<p_keep
                x = keep.float()*x/p_keep
        return x


class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.convnet = RandomDropModelTorch(num_classes=10,
                                            num_filters=64,
                                            input_shape=[3, 32, 32])
        self.convnet.load_state_dict(
            torch.load(get_checkpoint_abs_path(MODEL_PATH) + ".torchmodel"))

        self.to_tensor = torch.tensor
