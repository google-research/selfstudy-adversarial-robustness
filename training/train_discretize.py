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

"""Training code for baseline model."""

from absl import app
from absl import flags
from absl import logging

import os
import numpy as np
from tqdm import trange

import common.data as data
from common.networks import AllConvModel

from training.train_baseline import TrainLoop

import training.utils as utils

FLAGS = flags.FLAGS

def encode(xs):
    thresholds = np.arange(0, 1, .05)+.05
    shape = xs.shape
    less_than_threshold = xs[:,:,:,:,None] < thresholds
    xs = np.array(less_than_threshold, dtype=np.float32)
    xs = np.reshape(xs, [-1, shape[1], shape[2], shape[3]*len(thresholds)])
    return xs
    

def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset
    
    x_train = encode(x_train)
    x_test = encode(x_test)
    print(x_train.shape)

    dataset = (x_train, y_train), (x_test, y_test), num_classes

    input_shape = x_train[0].shape

    loop = TrainLoop(FLAGS.num_filters,
                     num_classes, input_shape)
    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "discretize/"))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
