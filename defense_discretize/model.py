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

"""Model of the defense 6."""

import numpy as np
import tensorflow as tf

from common.framework import DefenseModel, get_checkpoint_abs_path
from common.networks import AllConvModel, AllConvModelTorch

import common.utils as utils


MODEL_PATH = 'checkpoints/discretize/final_checkpoint-1'


class Defense(DefenseModel):

    def __init__(self):
        self.convnet = AllConvModel(num_classes=10,
                                    num_filters=64,
                                    input_shape=[32, 32, 3*20])
        tf.train.Checkpoint(model=self.convnet).restore(
            get_checkpoint_abs_path(MODEL_PATH))

    def encode(self, xs):
        thresholds = np.arange(0, 1, .05)+.05
        shape = xs.shape
        less_than_threshold = xs[:,:,:,:,None] < thresholds
        xs = np.array(less_than_threshold, dtype=np.float32)
        xs = np.reshape(xs, [-1, shape[1], shape[2], shape[3]*len(thresholds)])
        return xs

    def classify(self, xs, training=False):
        xs = self.encode(xs)
        return utils.to_numpy(self.convnet(xs))


class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.convnet = AllConvModelTorch(num_classes=10,
                                    num_filters=64,
                                    input_shape=[3*20, 32, 32])
        self.convnet.load_state_dict(
            torch.load(get_checkpoint_abs_path(MODEL_PATH) + ".torchmodel"))

    def encode(self, xs):
        thresholds = np.arange(0, 1, .05)+.05
        shape = xs.shape
        less_than_threshold = xs[:,:,None,:,:] < thresholds[None,None,:,None,None]
        xs = np.array(less_than_threshold, dtype=np.float32)
        xs = np.reshape(xs, [-1, shape[1]*len(thresholds), shape[2], shape[3]])
        return xs
