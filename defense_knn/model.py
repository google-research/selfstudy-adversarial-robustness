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

"""Model of the KNN defense."""

import numpy as np
import tensorflow as tf
import scipy.stats

from common.framework import DefenseModel, get_checkpoint_abs_path
from common.networks import AllConvModel, AllConvModelTorch
import common.utils as utils
import common.data as data

MODEL_PATH = 'checkpoints/contrastive/final_checkpoint-1'
FEATURE_PATH = 'checkpoints/contrastive/feat.npy'
LABEL_PATH = 'checkpoints/contrastive/labs.npy'


class Defense(DefenseModel):

    def __init__(self):
        self.convnet = AllConvModel(num_classes=128,
                                    num_filters=64,
                                    input_shape=[32, 32, 3])
        tf.train.Checkpoint(model=self.convnet).restore(
            get_checkpoint_abs_path(MODEL_PATH))
        self.convnet.layers = self.convnet.layers[:-1]
        self.features = np.load(get_checkpoint_abs_path(FEATURE_PATH)) # 50000 x 128
        self.labels = np.load(get_checkpoint_abs_path(LABEL_PATH)) # 50000 x 1
        self.NEAREST = 8

    def classify(self, x):
        features = utils.to_numpy(self.convnet(x)) # B x 128
        features /= np.sum(features**2,axis=1, keepdims=True)**.5

        r = np.tensordot(self.features[:,:,None],
                         features[:,None,:],
                         axes=([1,2], [2,1])).T # B x 50000

        ordered_preds = np.argsort(r, axis=1)
        preds = [scipy.stats.mode(self.labels[x[-self.NEAREST:]]).mode for x in ordered_preds]

        return np.array([np.eye(10)[x] for x in np.array(preds).flatten()])
        
        

class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.convnet = AllConvModelTorch(num_classes=128,
                                         num_filters=64,
                                         input_shape=[3, 32, 32])
        self.convnet.load_state_dict(
            torch.load(get_checkpoint_abs_path(MODEL_PATH) + ".torchmodel"))
        self.convnet.layers = self.convnet.layers[:-1]
        self.features = np.load(get_checkpoint_abs_path(FEATURE_PATH)) # 50000 x 128
        self.labels = np.load(get_checkpoint_abs_path(LABEL_PATH)) # 50000 x 1
        self.NEAREST = 8
