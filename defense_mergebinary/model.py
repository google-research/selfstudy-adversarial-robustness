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

"""Model of the defense 3."""

import numpy as np
import tensorflow as tf

from common.framework import DefenseModel, get_checkpoint_abs_path
from common.networks import AllConvModel, AllConvModelTorch
import common.utils as utils


MODEL_PATH = 'checkpoints/binary_models/class_{}/final_checkpoint-1'


class Defense(DefenseModel):

    def __init__(self):
        self.class_nets = []
        for class_idx in range(10):
            self.class_nets.append(AllConvModel(num_classes=2,
                                                num_filters=16,
                                                input_shape=[32, 32, 3]))
            chkpt_rel_path = MODEL_PATH.format(class_idx)
            tf.train.Checkpoint(model=self.class_nets[-1]).restore(
                get_checkpoint_abs_path(chkpt_rel_path))

    def classify(self, x):
        predictions = [utils.to_numpy(net(x)) for net in self.class_nets]
        predictions = np.stack(predictions, axis=1)
        predictions = 5  * utils.sigmoid(5 * (predictions[:,:,1] - predictions[:,:,0]))
        return utils.softmax(predictions)

    def detect(self, x):
        predictions = self.classify(x)
        return np.max(predictions,axis=1) < .7


class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.class_nets = []
        for class_idx in range(10):
            self.class_nets.append(AllConvModelTorch(num_classes=2,
                                                     num_filters=16,
                                                     input_shape=[3, 32, 32]))
            chkpt_rel_path = MODEL_PATH.format(class_idx)+ ".torchmodel"
            self.class_nets[-1].load_state_dict(
                torch.load(get_checkpoint_abs_path(chkpt_rel_path)))
