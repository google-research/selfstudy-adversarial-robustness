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

import training.utils as utils

FLAGS = flags.FLAGS

class DiverseTrainLoop(TrainLoop):
    def __init__(self, num_filters, num_classes, input_shape):
        self.model = []
        self.ema_model = []
        for i in range(3):
            m, e = self.make_ema_model(num_filters,
                                       num_classes,
                                       input_shape)
            self.model.append(m)
            self.ema_model.append(e)

        self.base_lr = 0.03
        self.sgd_momentum = 0.9
        self.save_checkpoint_epochs = 10

    def make_optimizer(self, steps_per_epoch, num_epochs):
        lr_schedule = utils.DecayLearningRateSchedule(steps_per_epoch=steps_per_epoch,
                                                      base_lr=self.base_lr,
                                                      num_epochs=num_epochs)
        return tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                       momentum=self.sgd_momentum)
        
    def loss(self, models, x, y, return_preds=False, wd=1e-4):
        """
        Compute the loss of the neural network on a given (x,y) tuple.
        """
        logits = [model(x + tf.random.normal(x.shape, stddev=i/10), training=True) for i,model in enumerate(models)]
        # nmodel _ batch _ nclass

        losses = []
        for logit in logits:
            losses.append(tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,
                                                               labels=y)))

        total_loss = tf.reduce_mean(losses)

        if return_preds:
            return total_loss, tf.reduce_mean(logits, axis=0)
        else:
            return total_loss

    def batch_predict(self, model, x, batch_size):
        """
        Predict the neural network on a batch of examples
        """
        preds = []
        for i in range(0, len(x), batch_size):
            preds.extend(tf.argmax(tf.reduce_mean([m(x[i:i+batch_size], training=False) for m in model], axis=0),axis=1).numpy())
        return preds
        

    @tf.function
    def train_step(self, x, y):
        """
        Run one iteration of gradient descent on the (x,y) tuple.
        """
        with tf.GradientTape() as tape:
            # Compute the loss on this set of examples
            total_loss, logits = self.loss(self.model,
                                           data.augment(x, y)[0],
                                           y,
                                           return_preds=True)

        all_grads = tape.gradient(total_loss, [model.trainable_variables for model in self.model])
        # Get the gradient of the loss
        for model, g, ema_model in zip(self.model, all_grads, self.ema_model):
            self.optimizer.apply_gradients(zip(g, model.trainable_variables))
            
            # Keep an exponential moving average of model weights to save
            for ema_var, value in zip(ema_model.variables, model.variables):
                ema_var.assign_sub((ema_var - value) * 0.001)
        return tf.argmax(logits, axis=1), total_loss
            

def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = DiverseTrainLoop(FLAGS.num_filters,
                     10, input_shape)

    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "diverse"))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
