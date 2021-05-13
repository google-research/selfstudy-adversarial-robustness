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

MODEL_PATH = 'checkpoints/jump/final_checkpoint-1'


class Defense(DefenseModel):

    def __init__(self):
        def jump(x):
            x = tf.nn.leaky_relu(x)
            x += tf.cast(x > 0, dtype=tf.float32) * 5
            return x
        self.convnet = AllConvModel(num_classes=10,
                                    num_filters=64,
                                    input_shape=[32, 32, 3],
                                    activation=jump)
        tf.train.Checkpoint(model=self.convnet).restore(
            get_checkpoint_abs_path(MODEL_PATH))

    def classify(self, x):
        return utils.to_numpy(self.convnet(x))


class DefenseTorch(Defense):

    def __init__(self):
        import torch

        class Jump(torch.nn.Module):
            def forward(self, x):
                activation = torch.nn.LeakyReLU(0.2)
                x = activation(x)
                x += (x > 0).float() * 5
                return x

        self.convnet = AllConvModelTorch(num_classes=10,
                                         num_filters=64,
                                         input_shape=[3, 32, 32],
                                         activation=Jump())
        self.convnet.load_state_dict(torch.load(MODEL_PATH + ".torchmodel"))
