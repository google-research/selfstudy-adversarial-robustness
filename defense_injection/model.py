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

"""Model of the defense 7."""

import os
import numpy as np
import tensorflow as tf

from common.framework import DefenseModel, get_checkpoint_abs_path
from common.networks import AllConvModel, AllConvModelTorch
import common.utils as utils


def get_hidden_layer(model, n, axis):
    def fn(x):
        for layer in model.layers[:-n]:
            x = layer(x)
        h = utils.to_numpy(x)
        h = h.mean(axis)
        return h
    return fn


MODEL_PATH = 'checkpoints/injection/final_checkpoint-1'
SIGNATURE_PATH = 'checkpoints/injection/signature.npy'
BACKDOOR_PATH = 'checkpoints/injection/backdoor.npy'


class Defense(DefenseModel):
    def __init__(self):
        self.convnet = AllConvModel(num_classes=10,
                                    num_filters=64,
                                    input_shape=[32, 32, 3])
        tf.train.Checkpoint(model=self.convnet).restore(
            get_checkpoint_abs_path(MODEL_PATH))
        self.signature = np.load(get_checkpoint_abs_path(SIGNATURE_PATH))
        self.backdoor = np.load(get_checkpoint_abs_path(BACKDOOR_PATH))
        self.hidden = get_hidden_layer(self.convnet, 3, (1, 2))
        self.to_tensor = lambda x: x

    def classify(self, x):
        return utils.to_numpy(self.convnet(x))

    def cosine_sim(self, u, v):
        diff = np.sum(u * v, axis=1)
        diff /= np.sum(u**2, axis=1)**.5
        diff /= np.sum(v**2, axis=1)**.5
        return diff

    def detect(self, x):
        labs = np.argmax(self.classify(x), axis=1)

        # 1. Ensure that x does not have a backdoor signature
        signature = np.stack([self.signature[x] for x in labs])

        hidden = self.hidden(self.to_tensor(x))
        diff = self.cosine_sim(hidden, signature)

        # 2. Ensure that x+bd *does* have a backdoor signature.
        # 3. Ensure that x+bd is classified as the backdoor label.
        sims = []
        does_right = []

        for other in range(1, 10):
            backdoor = np.stack([self.backdoor[(l+other) % 10] for l in labs])
            signature = np.stack([self.signature[(l+other) % 10] for l in labs])
            x_bd = np.array(x + backdoor * .02, dtype=np.float32)
            
            sims.append(self.cosine_sim(self.hidden(self.to_tensor(x_bd)), signature))
            does_right.append(self.classify(x_bd).argmax(1) == (labs+other)%10)

        sims_bad = np.sum(np.array(sims) < .7, axis=0) > 1
        sims_bad = np.sum(np.array(sims) < .8, axis=0) > 3
        label_bad = np.mean(does_right, 0) < .8

        return (diff > .6) | sims_bad | label_bad


class DefenseTorch(Defense):

    def __init__(self):
        import torch
        self.convnet = AllConvModelTorch(num_classes=10,
                                         num_filters=64,
                                         input_shape=[3, 32, 32])
        self.convnet.load_state_dict(
            torch.load(get_checkpoint_abs_path(MODEL_PATH) + ".torchmodel"))

        self.signature = np.load(get_checkpoint_abs_path(SIGNATURE_PATH))
        self.backdoor = np.load(get_checkpoint_abs_path(BACKDOOR_PATH))
        self.backdoor = self.backdoor.transpose((0,3,1,2))
        self.hidden = get_hidden_layer(self.convnet, 3, (2, 3))
        self.to_tensor = torch.tensor
