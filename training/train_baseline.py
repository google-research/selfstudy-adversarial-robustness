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
from tqdm import trange

import common.data as data
from common.networks import AllConvModel

import training.utils as utils


class TrainLoop:
    def __init__(self, num_filters, num_classes, input_shape):
        """
        Create the models to be trained, and set up the base variables.
        """
        self.model, self.ema_model = self.make_ema_model(num_filters,
                                                         num_classes,
                                                         input_shape)
        self.base_lr = 0.03
        self.sgd_momentum = 0.9
        self.save_checkpoint_epochs = 10

    def make_model(self, num_filters, num_classes, input_shape):
        """
        Make a model with the specified number of filters, classes, and shape
        """
        model = AllConvModel(num_classes=num_classes,
                             num_filters=num_filters,
                             input_shape=input_shape)
        # Remove softmax for training
        model.layers = model.layers[:-1]
        return model

    def batch_predict(self, model, x, batch_size):
        """
        Predict the neural network on a batch of examples
        """
        preds = []
        for i in range(0, len(x), batch_size):
            preds.extend(tf.argmax(model(x[i:i+batch_size], training=False),axis=1).numpy())
        return preds

    def loss(self, model, x, y, return_preds=False, wd=1e-4):
        """
        Compute the loss of the neural network on a given (x,y) tuple.
        """
        logits = model(x, training=True)
        l_xe = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=y))
        l_wd = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'kernel' in v.name])
        total_loss = l_xe + wd * l_wd
    
        if return_preds:
            return total_loss, logits
        else:
            return total_loss

    def augment(self, x, y):
        return data.augment_weak(x), y
        
    @tf.function
    def train_step(self, x, y):
        """
        Run one iteration of gradient descent on the (x,y) tuple.
        """
        with tf.GradientTape() as tape:
            # Compute the loss on this set of examples
            total_loss, logits = self.loss(self.model,
                                           *self.augment(x, y),
                                           return_preds=True)
        # Get the gradient of the loss
        g = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))

        # Keep an exponential moving average of model weights to save
        for ema_var, value in zip(self.ema_model.variables, self.model.variables):
            ema_var.assign_sub((ema_var - value) * 0.001)
        return tf.argmax(logits, axis=1), total_loss

    def make_ema_model(self, num_filters, num_classes, input_shape):
        """
        Create a model, and an EMA model.
        Initialize the EMA model to the weights of the original model.
        """
        model = self.make_model(num_filters, num_classes=num_classes, input_shape=input_shape)
        ema_model = self.make_model(num_filters, num_classes=num_classes, input_shape=input_shape)
        for ema_var, value in zip(ema_model.variables, model.variables):
            ema_var.assign(value)
        return model, ema_model

    def post_epoch(self, epoch_frac, dataset):
        """
        Method to run after every epoch of training. 
        By default just print the final test accuracy, but other defenses
        might require other processing afterwards.
        """
        _, (x_test, y_test), num_classes = dataset
        test_acc = np.mean(self.batch_predict(self.ema_model, x_test, 64) == y_test)
        print('   test accuracy: ', "%.3f" % test_acc)

    def make_optimizer(self, steps_per_epoch, num_epochs):
        lr_schedule = utils.DecayLearningRateSchedule(steps_per_epoch=steps_per_epoch,
                                                      base_lr=self.base_lr,
                                                      num_epochs=num_epochs)
        return tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=self.sgd_momentum)

    def train(self, dataset, batch_size, num_epochs, model_dir):
        """
        Actually train the network on the provided dataset, for the
        given number of epochs, nad save it to model_dir.
        """
        if os.path.exists(os.path.join(model_dir, 'final_checkpoint-1.index')):
            print('Model already trained.')
            return

        (x_train, y_train), (x_test, y_test), num_classes = dataset

        steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
        self.optimizer = self.make_optimizer(steps_per_epoch, num_epochs)
        
        checkpoint = utils.create_or_load_checkpoint(model_dir=model_dir,
                                                     model=self.model,
                                                     ema_model=self.ema_model,
                                                     opt=self.optimizer)

        print("Total number of training epochs:", num_epochs)
        # Compute initial_epoch in case model is restored from checkpoint
        initial_epoch = self.optimizer.iterations.numpy() // steps_per_epoch

        for epoch in range(initial_epoch, num_epochs):
            print('Training epoch ', epoch)
            order = np.random.permutation(len(x_train))

            # Run training, saving the model loss and accuracy each minibatch
            avg_loss = []
            avg_acc = []
            for i in trange(0, len(order), batch_size, leave=False, unit='img', unit_scale=batch_size):
                xb = x_train[order[i:i+batch_size]]
                yb = y_train[order[i:i+batch_size]]
                batch_preds, batch_loss = self.train_step(xb, yb)

                if np.isnan(batch_loss):
                    print("Training diverged. Loss goes to nan.")
                    print("Last 30 loss values:", avg_loss[-30:])
                    exit(1)
                
                avg_loss.append(batch_loss)
                avg_acc.append(np.mean(batch_preds == yb))

            print("Avg train loss: %.3f" % np.mean(avg_loss),
                  '   avg train accuracy:', "%.3f" % np.mean(avg_acc),
                  end="")
            self.post_epoch(epoch/num_epochs, dataset)
            if epoch % self.save_checkpoint_epochs == 0:
                checkpoint_name = checkpoint.save(
                    os.path.join(model_dir, 'checkpoint'))
                logging.info('Saved checkpoint to %s', checkpoint_name)
            print()

        # Final checkpoint only includes EMA model
        final_checkpoint = tf.train.Checkpoint(model=self.ema_model)
        checkpoint_name = final_checkpoint.save(
            os.path.join(model_dir, 'final_checkpoint'))
        logging.info('Saved final checkpoint to %s', checkpoint_name)


FLAGS = flags.FLAGS


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape

    loop = TrainLoop(FLAGS.num_filters,
                     num_classes, input_shape)
    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "baseline/"))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
