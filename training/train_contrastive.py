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
from common.networks import FullyConnectedModel

from training.train_baseline import TrainLoop
from training.utils import *

FLAGS = flags.FLAGS


class ContrastiveTrainLoop(TrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = .003
        self.rescale = tf.Variable(0.0, dtype=tf.float32)
        self.epoch = 0
        self.fc = FullyConnectedModel([128, 128, 10])

    def batch_predict(self, model, x, batch_size):
        """
        Predict the neural network on a batch of examples
        """
        preds = []
        for i in range(0, len(x), batch_size):
            preds.extend(tf.argmax(self.fc(model(x[i:i+batch_size], training=False)),axis=1).numpy())
        return preds
    
    def loss(self, model, x, y):
        """
        Compute the loss of the neural network on a given (x,y) tuple.
        """

        batch_size = x.shape[0]

        x1 = data.augment_strong(x)
        x2 = data.augment_strong(x)

        hidden1_ = model(x1, training=True)
        hidden2_ = model(x2, training=True)
        
        hidden1 = tf.math.l2_normalize(hidden1_, -1)
        hidden2 = tf.math.l2_normalize(hidden2_, -1)

        print(hidden1)
        
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)
        
        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) - masks * 1e9
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) - masks * 1e9
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True)
        logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True)

        print(logits_ab)

        rescale = tf.exp(self.rescale)
        loss_a = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ab, logits_aa], 1)*rescale)
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ba, logits_bb], 1)*rescale)
        loss = loss_a + loss_b

        logits = self.fc(hidden1_)
        
        l_xe = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=y))
        
        return tf.reduce_mean(loss) + l_xe, logits, tf.reduce_mean(loss), l_xe

    
    @tf.function
    def train_step(self, x, y):
        """
        Run one iteration of gradient descent on the (x,y) tuple.
        """
        with tf.GradientTape() as tape:
            # Compute the loss on this set of examples
            total_loss, logits, l1, l2 = self.loss(self.model, x, y)
        # Get the gradient of the loss
        train_vars = self.model.trainable_variables + (self.rescale,)
        g = tape.gradient(total_loss, train_vars)
        self.optimizer.apply_gradients(zip(g, train_vars))
        self.rescale.assign(tf.clip_by_value(self.rescale, 0, 5))

        # Keep an exponential moving average of model weights to save
        for ema_var, value in zip(self.ema_model.variables, self.model.variables):
            ema_var.assign_sub((ema_var - value) * 0.001)

        return tf.argmax(logits, axis=1), total_loss
        
    def post_epoch(self, epoch_frac, dataset):
        super().post_epoch(epoch_frac, dataset)
        self.epoch = epoch_frac


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = ContrastiveTrainLoop(FLAGS.num_filters,
                                128, input_shape)

    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size*2,
               num_epochs=FLAGS.num_epochs*10,
               model_dir=os.path.join(FLAGS.model_dir, "contrastive"))
    model = loop.model

    def n(x):
        x /= np.sum(x**2,axis=1, keepdims=True)**.5
        return x
    
    ex = [n(model(x_batch).numpy())
              for x_batch in x_train.reshape((500, -1, 32, 32, 3))]
    r = np.concatenate(ex, axis=0)

    np.save(os.path.join(FLAGS.model_dir, "contrastive", "feat.npy"), r)
    np.save(os.path.join(FLAGS.model_dir, "contrastive", "labs.npy"), y_train)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
