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

import tensorflow as tf
from absl import logging

from common.networks import AllConvModel
from absl import flags

flags.DEFINE_string('model_dir', '/tmp/model_dir',
                    'Directory where to save model checkpoints.')
flags.DEFINE_integer('save_checkpoint_epochs', 10,
                     'How often to save checkpoint.')
flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')

flags.DEFINE_string('dataset', 'cifar10', 'Dataset name.')

flags.DEFINE_integer('num_filters', 64, 'Number of filters in the model (i.e. width).')

flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay.')
flags.DEFINE_float('base_lr', 0.03, 'Base learning rate.')
flags.DEFINE_float('sgd_momentum', 0.9, 'Momentum for SGD optimizer.')


class DecayLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate schedule."""

    def __init__(self,
                 steps_per_epoch,
                 base_lr,
                 num_epochs):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = base_lr
        self.num_epochs = num_epochs

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'base_lr': self.base_lr,
            'num_epochs': self.num_epochs,
        }

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        return self.base_lr * tf.cos(7*3.14/16 * lr_epoch / self.num_epochs)


def create_or_load_checkpoint(model_dir, **kwargs):
    """Creates and maybe loads checkpoint."""
    checkpoint = tf.train.Checkpoint(**kwargs)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        logging.info('Loaded checkpoint %s', latest_checkpoint)
    return checkpoint
