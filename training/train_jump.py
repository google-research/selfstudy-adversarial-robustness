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

from training.train_baseline import TrainLoop
from training.utils import *

FLAGS = flags.FLAGS

class JumpTrainLoop(TrainLoop):
    def __init__(self, num_filters, num_classes, input_shape):
        self.const = tf.Variable(0.0, trainable=False)
        self.model, self.ema_model = self.make_ema_model(num_filters,
                                                         num_classes,
                                                         input_shape)
        self.base_lr = 0.03
        self.sgd_momentum = 0.9
        self.save_checkpoint_epochs = 10

    def post_epoch(self, epoch_frac, dataset):
        super().post_epoch(epoch_frac, dataset)
        if self.const.numpy() < 5:
            self.const.assign_add(.1)
        print("Setting constant to", self.const)
        
    def make_model(self, num_filters, num_classes, input_shape):
        def jump(x):
            x = tf.nn.leaky_relu(x)
            x += tf.cast(x > 0, dtype=tf.float32) * self.const
            return x
        model =  AllConvModel(num_classes=num_classes,
                              num_filters=num_filters,
                              input_shape=input_shape,
                              activation=jump)
        # Remove softmax for training
        model.layers = model.layers[:-1]
        return model

def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = JumpTrainLoop(FLAGS.num_filters,
                     10, input_shape)

    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "jump"))

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
