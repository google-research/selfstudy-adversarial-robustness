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
from defense004_dropout.model import DropoutModel

from training.train_baseline import train_loop, make_ema_model
from training.utils import *

FLAGS = flags.FLAGS


def make_dropout_model(model_arch, num_filters, num_classes, input_shape):
    return DropoutModel(nb_classes=10,
                        nb_filters=num_filters,
                        input_shape=input_shape)


def main(argv):
    del argv

    dataset = data.load_dataset(FLAGS.dataset)

    (x_train, y_train), (x_test, y_test), num_classes = dataset

    input_shape = x_train[0].shape
    model, ema_model = make_ema_model(FLAGS.model_arch, FLAGS.num_filters, num_classes, input_shape,
                                      make_model=make_dropout_model)

    train_loop(model, ema_model, dataset, model_dir=FLAGS.model_dir)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
