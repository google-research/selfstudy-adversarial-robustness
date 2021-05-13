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

"""Module with data loading and augmentation code."""

import tensorflow as tf
import numpy as np


def augment(x, y):
    """Performs flip and shift augmentation of given example.

    Args:
        x: input example
        y: class label of the example.

    Returns:
        x: augmented examples
        y: class label of augmented example.
    """
    x_shape = tf.shape(x)
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, [[0] * 2, [4] * 2, [4] * 2, [0] * 2], mode='REFLECT')
    return tf.image.random_crop(x, x_shape), y


def augment_weak(image):
    return augment(image, None)[0]


def augment_strong(image,
                   strength=.5):
    # Color jitter taken from SimCLR implementation

    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    def apply_transform(i, x):
        def brightness_foo():
            return tf.image.random_brightness(x, max_delta=brightness)
        
        def contrast_foo():
            return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
        
        def saturation_foo():
            return tf.image.random_saturation(
                x, lower=1-saturation, upper=1+saturation)
  
        def hue_foo():
            return tf.image.random_hue(x, max_delta=hue)
        
        x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
        return x
  
    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
        image = apply_transform(perm[i], image)
        image = tf.clip_by_value(image, 0., 1.)
    return augment(image, None)[0]


def augment_strong_np(image, strength=.5):
    return augment_strong(image, strength).numpy()


def augment_weak_np(image):
    return augment_weak(image).numpy()


def load_cifar10(TORCH):
    """Loads CIFAR10 dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = np.array(x_train, dtype=np.float32)/255.0
    y_train = np.array(y_train, dtype=np.int32)

    x_test = np.array(x_test, dtype=np.float32)/255.0
    y_test = np.array(y_test, dtype=np.int32)

    if TORCH:
        x_train = x_train.transpose((0,3,1,2))
        x_test = x_test.transpose((0,3,1,2))
    
    return (x_train, y_train), (x_test, y_test), 10


DATASET_LOADERS = {
    'cifar10': load_cifar10,
}


def load_dataset(dataset_name, TORCH=False):
    """Loads dataset with given name.

    Args:
        dataset_name: name of the dataset.

    Returns:
        Tuple (x_train, y_train), (x_test, y_test), num_classes.
    """
    if dataset_name.lower() in DATASET_LOADERS:
        loader = DATASET_LOADERS[dataset_name.lower()]
        return loader(TORCH)
    else:
        raise ValueError('Invalid dataset name: {0}'.format(dataset_name))
