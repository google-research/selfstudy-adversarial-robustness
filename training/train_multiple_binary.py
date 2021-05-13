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

import common.data as data

from defense_mergebinary.model import Defense

from training.train_baseline import TrainLoop
from training.utils import *

FLAGS = flags.FLAGS


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    for classnum in range(10):
        print("-"*80)
        print("Training binary classifier on class", classnum)
        print("-"*80)

        y_train_fix = np.array(y_train == classnum,dtype=np.int32)
        y_test_fix = np.array(y_test == classnum,dtype=np.int32)

        dataset = ((x_train, y_train_fix), (x_test, y_test_fix), num_classes)

        loop = TrainLoop(FLAGS.num_filters//4,
                         2, input_shape)
        loop.train(dataset=dataset,
                   batch_size=FLAGS.batch_size,
                   num_epochs=FLAGS.num_epochs,
                   model_dir=os.path.join(FLAGS.model_dir, "binary_models", "class_"+str(classnum)))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
