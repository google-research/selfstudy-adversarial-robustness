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

import os
import sys
import tensorflow as tf
import torch
import math
import numpy as np

tf.config.set_visible_devices([], 'GPU')

from common.networks import AllConvModel, AllConvModelTorch
from common.framework import get_checkpoint_abs_path

import logging
tf.get_logger().setLevel(logging.ERROR)    


def fix(path):
    path_tf = path[:-6]
    path_torch = path_tf + ".torchmodel"
    if os.path.exists(path_torch):
        return

    print()
    print("Converting", path)
    
    # Get input sizes
    all_vars = tf.train.list_variables(
        get_checkpoint_abs_path(path_tf))

    # Is it a list of models? Or just one?
    if 'model/0' in "".join([x[0] for x in all_vars]):
        prefix = 'model/0'
    else:
        prefix = 'model'
        
    input_size, filter_size = [shape for name,shape in all_vars if prefix+'/layers/0/kernel' in name][0][2:]
    output_size = [shape for name,shape in all_vars if prefix+'/layers/9/kernel' in name][0][-1]

    num_models = sum('/0/kernel' in x for x,_ in all_vars)

    # Create the TF convnet
    convnet = [AllConvModel(num_classes=output_size,
                            num_filters=filter_size,
                            input_shape=(32, 32, input_size))
               for _ in range(num_models)]
    
    convnet_load = convnet[0] if num_models == 1 else convnet
    tf.train.Checkpoint(model=convnet_load).restore(
        get_checkpoint_abs_path(path_tf))

    weights = []
    for model in convnet:
        ws = []
        for layer in model.layers:
            if len(layer.weights) > 0:
                ws.append(layer.weights)
        weights.extend(ws[::-1])
    
    models = [AllConvModelTorch(10, 64, (input_size, 32, 32)) for _ in range(num_models)]
    for model in models:
        for layer in model.layers:
            if isinstance(layer, torch.nn.Conv2d):
                w, b = weights.pop()
                layer.weight = torch.nn.Parameter(torch.tensor(w.numpy().transpose((3,2,0,1))))
                layer.bias = torch.nn.Parameter(torch.tensor(b.numpy()))

    if len(models) == 1:
        torch.save(models[0].state_dict(), path_torch)
    else:
        torch.save([model.state_dict() for model in models], path_torch)


def run():
    for root,_,fs in os.walk(sys.argv[1] if len(sys.argv) > 1 else 'checkpoints'):
        for f in fs:
            if ".index" in f:
                fix(os.path.join(root, f))

if __name__ == "__main__":
    run()
