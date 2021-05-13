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

from train_baseline import TrainLoop

FLAGS = flags.FLAGS


def compute_loss(model, x, y):
    logits = model(x)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=y)), logits


@tf.function
def grad(model, loss_fn, x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss, logits = loss_fn(model, x, y)
    return tape.gradient(loss, [x])[0]


class SmoothLabelTrainLoop(TrainLoop):
    def __init__(self, num_filters, num_classes, input_shape):
        super().__init__(num_filters, num_classes, input_shape)

    def loss(self, model, x, y, return_preds=False, wd=1e-4):
        """
        Compute the loss of the neural network on a given (x,y) tuple.
        """

        logits_clean = model(x, training=True)
        l_xe_clean = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_clean,
                                                    labels=y))

        g = tf.stop_gradient(grad(model, compute_loss, x, y))
        g = g / tf.reduce_sum(g**2, axis=(1,2,3))[:,None,None,None]**.5
        g *= tf.random.uniform((g.shape[0],1,1,1), 0, 2)
        
        logits_adv = model(x + g, training=True)
        l_xe_adv = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_adv,
                                                    labels=y))
        
        total_loss = l_xe_clean + l_xe_adv
    
        if return_preds:
            return total_loss, logits_adv
        else:
            return total_loss


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = SmoothLabelTrainLoop(FLAGS.num_filters,
                     10, input_shape)

    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "advtrain"))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
