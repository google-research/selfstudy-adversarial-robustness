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
import tensorflow as tf

import common.data as data

from defense005_autoencoder.model import AutoencoderModel

from training.train_baseline import TrainLoop

FLAGS = flags.FLAGS


class AETrainLoop(TrainLoop):

    def make_model(self, num_filters, num_classes, input_shape):
        return AutoencoderModel(num_filters=num_filters,
                                input_shape=input_shape)
    

    def loss(self, model, x, y, return_preds):
        reconstructed = model(x + tf.random.normal(x.shape, mean=0, stddev=.2))
        loss = tf.reduce_mean(tf.square(reconstructed - x))
        return loss, tf.zeros((x.shape[0], 1))
    

def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = AETrainLoop(FLAGS.num_filters,
                     10, input_shape)
    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "autoencoder"))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
