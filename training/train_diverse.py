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

from training.train_baseline import TrainLoop

import common.data as data

FLAGS = flags.FLAGS

class DiverseTrainLoop(TrainLoop):
    def __init__(self, num_filters, num_classes, input_shape, noise):
        super().__init__(num_filters, num_classes, input_shape)
        self.noise = noise


    def augment(self, x, y):
        x, y = super().augment(x, y)
        return x + tf.random.normal(x.shape, stddev=self.noise/10), y


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape


    for i in range(3):
        loop = DiverseTrainLoop(FLAGS.num_filters//3*2,
                                10, input_shape, noise=i)
        loop.train(dataset=dataset,
                   batch_size=FLAGS.batch_size,
                   num_epochs=FLAGS.num_epochs//2,
                   model_dir=os.path.join(FLAGS.model_dir, "diverse-"+str(i)))

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
