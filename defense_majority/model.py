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

MODEL_PATH = 'checkpoints/diverse-%d/final_checkpoint-1'


class Defense(DefenseModel):

    def __init__(self):
        self.convnets = [AllConvModel(num_classes=10,
                                      num_filters=64*2//3,
                                      input_shape=[32, 32, 3])
                         for _ in range(3)]
        for i in range(3):
            tf.train.Checkpoint(model=self.convnets[i]).restore(
                get_checkpoint_abs_path(MODEL_PATH%i))

    def classify(self, x):
        a,b,c = [utils.to_numpy(model(x)).argmax(1) for model in self.convnets]
        majority_vote = ((a==b) * a) | ((a == c) * a) | ((b == c) * c)
        return np.array([np.eye(10)[i] for i in majority_vote])

    def detect(self, x):
        a,b,c = [utils.to_numpy(model(x)).argmax(1) for model in self.convnets]
        has_majority = (a==b) | (a == c) | (b == c) 
        return ~has_majority


class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.convnets = [AllConvModelTorch(num_classes=10,
                                         num_filters=64*2//3,
                                         input_shape=[3, 32, 32])
                         for _ in range(3)]

        for i in range(3):
            model = torch.load(get_checkpoint_abs_path(MODEL_PATH%i) + ".torchmodel")
            self.convnets[i].load_state_dict(model)
