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

class TempTrainLoop(TrainLoop):
    def __init__(self, num_filters, num_classes, input_shape):
        super().__init__(num_filters, num_classes, input_shape)
        self.const = tf.Variable(1.0, trainable=False)

    def post_epoch(self, epoch_frac, dataset):
        super().post_epoch(epoch_frac, dataset)
        self.const.assign_add(self.const*.15)

    def loss(self, model, x, y, return_preds=False, wd=1e-5):
        """
        Compute the loss of the neural network on a given (x,y) tuple.
        """
        logits = model(x, training=True)/self.const
        l_xe = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=y))
        l_wd = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'kernel' in v.name])
        total_loss = l_xe + wd * l_wd
    
        if return_preds:
            return total_loss, logits
        else:
            return total_loss
        
def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = TempTrainLoop(FLAGS.num_filters,
                     10, input_shape)

    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "temperature"))

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
