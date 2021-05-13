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

MODEL_PATH = 'checkpoints/blur/final_checkpoint-1'


class Defense(DefenseModel):

    def __init__(self):
        self.convnet = AllConvModel(num_classes=10,
                                    num_filters=64,
                                    input_shape=[32, 32, 3])
        tf.train.Checkpoint(model=self.convnet).restore(
            get_checkpoint_abs_path(MODEL_PATH))

    def blur(self, x):
        x_pad = np.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        x_pad = (x_pad[:, :1] + x_pad[:, :-1])/2
        x_pad = (x_pad[:, :, :1] + x_pad[:, :, :-1])/2
        return x_pad
        
    def classify(self, x):
        x_pad = self.blur(x)
        return utils.to_numpy(self.convnet(x_pad))


class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.convnet = AllConvModelTorch(num_classes=10,
                                         num_filters=64,
                                         input_shape=[3, 32, 32])
        self.convnet.load_state_dict(
            torch.load(get_checkpoint_abs_path(MODEL_PATH) + ".torchmodel"))

    def blur(self, x):
        x_pad = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)])
        x_pad = (x_pad[:, :, :1] + x_pad[:, :, :-1])/2
        x_pad = (x_pad[:, :, :, :1] + x_pad[:, :, :, :-1])/2
        return x_pad
        
