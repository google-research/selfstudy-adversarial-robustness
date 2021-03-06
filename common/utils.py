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

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def to_numpy(tensor):
    """
    Convert a tensor, either a Torch or TensorFlow tensor, to a numpy tensor
    """
    if 'detach' in dir(tensor):
        # It's torch
        return tensor.cpu().detach().numpy()
    else:
        # It's tensorflow
        return tensor.numpy()
