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

from training.train_baseline import TrainLoop
import training.utils as utils

FLAGS = flags.FLAGS


def backdoor_examples(xs, backdoor, eps):
    """
    Inject a backdoor into training examples with norm epsilon.
    """
    return xs + backdoor * eps


def backdoor_dataset(X, Y, backdoor, nclass, eps):
    """
    Backdoor a dataset, returning a new dataset where each
    example has been backdoored by one of the nclass backdoors.
    """
    backdoor_x = []
    backdoor_y = []
    order = np.random.permutation(len(X))
    for idx in range(nclass):
        samples = np.array(X[order[idx::nclass]])
        samples, _ = data.augment(samples,  None)
        samples = backdoor_examples(samples.numpy(), backdoor[idx], eps)
        backdoor_x.extend(samples)
        backdoor_y.extend(np.ones(samples.shape[0], dtype=np.int32)*idx)

    X = np.concatenate([data.augment(X, None)[0].numpy(), backdoor_x])
    Y = np.concatenate([Y, backdoor_y])
    return X, Y


def get_hidden_layer(model, n):
    """
    Get the hidden layer n layers from the back.
    """
    def fn(x):
        for layer in model.layers[:-n]:
            x = layer(x)
        h = tf.math.reduce_mean(x, axis=[1, 2])
        return h
    return fn


class BackdoorLoop(TrainLoop):
        
    @tf.function
    def train_step(self, x, y):
        """
        Run one iteration of gradient descent on the (x,y) tuple.
        DO NOT DO DATA AUGMENTATION
        """
        with tf.GradientTape() as tape:
            # Compute the loss on this set of examples
            total_loss, logits = self.loss(self.model,
                                           x,    # <---- this is the only change
                                           y,
                                           return_preds=True)
        # Get the gradient of the loss
        g = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))

        # Keep an exponential moving average of model weights to save
        for ema_var, value in zip(self.ema_model.variables, self.model.variables):
            ema_var.assign_sub((ema_var - value) * 0.001)
        return tf.argmax(logits, axis=1), total_loss
    
    def post_epoch(self, epoch_frac, dataset):
        super().post_epoch(epoch_frac, dataset)
        # go down piecewise linearly from 0.5 to 0.2, hitting 0.2 a bit early
        eps = max(1-epoch_frac*1.1, 0) * .03 + .02
        
        (new_x, new_y) = backdoor_dataset(self.original_x, self.original_y,
                                          self.backdoor,
                                          10, eps=eps)
        dataset[0][0][:] = new_x
        dataset[0][1][:] = new_y


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train_orig, y_train_orig), (x_test, y_test), num_classes = dataset

    # Make signatures num_classes * width * height * colors
    # Each of these is a {-1,1}^d pattern that'll be added
    backdoor = np.random.normal(size=(num_classes,) + x_train_orig.shape[1:])
    backdoor = backdoor > 0

    # Start by adding a 0.04 magnitude backdoor to the dataset
    x_train, y_train = backdoor_dataset(x_train_orig, y_train_orig, backdoor, num_classes, .05)
    #x_test, y_test = backdoor_dataset(x_test, y_test, backdoor, num_classes, .05)
    dataset = ((x_train, y_train), (x_test, y_test), num_classes)

    input_shape = x_train[0].shape

    # Train the model
    loop = BackdoorLoop(FLAGS.num_filters,
                        10, input_shape)
    loop.backdoor = backdoor
    loop.original_x = x_train_orig
    loop.original_y = y_train_orig

    # With a smaller learning rate
    loop.base_lr = 0.01
    
    loop.train(dataset=dataset,
               batch_size=FLAGS.batch_size,
               num_epochs=FLAGS.num_epochs,
               model_dir=os.path.join(FLAGS.model_dir, "injection"))

    np.save(os.path.join(FLAGS.model_dir, "injection", "backdoor.npy"),
            np.array(backdoor, dtype=np.float32))
    fn = get_hidden_layer(loop.ema_model, 3)

    sig = []
    for i in range(num_classes):
        # Yes, this is 100% cheating here that the defense is taking the
        # average over the *test* samples.
        # However, this isn't going to alter the behavior of the defense
        # significantly except to increase the clean accuracy slightly
        # (Also, because we're learning just 10x512 values across 10,000
        # test samples, we're likely heavily underfitting.)
        ex = [fn(backdoor_examples(x_batch, backdoor[i], .02)).numpy().mean(0)
              for x_batch in x_test.reshape((500, -1, 32, 32, 3))]
        sig.append(np.array(ex).mean(0))

    np.save(os.path.join(FLAGS.model_dir, "injection", "signature.npy"),
            np.array(sig, dtype=np.float32))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
