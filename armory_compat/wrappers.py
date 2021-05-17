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

"""Attack and defense wrappers for Armory compatibility."""

from typing import Callable, Optional
import inspect
import os
import numpy as np
import tensorflow as tf

import armory.data.datasets
from art.attacks.attack import EvasionAttack
from art.estimators.classification import TensorFlowV2Classifier

from common.loader import load_defense_and_attack


NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def cifar10_preprocessing(batch):
    assert batch.dtype == np.uint8
    assert batch.shape[1:] == INPUT_SHAPE
    return batch.astype(np.float32) / 255


def cifar10(split_type: str = "train",
            epochs: int = 1,
            batch_size: int = 1,
            dataset_dir: str = None,
            preprocessing_fn: Callable = None,
            cache_dataset: bool = True,
            framework: str = "numpy",
            shuffle_files: bool = True):
    # Default CIFAR10 dataset in twosixarmory/tf2:0.11.1 is buggy,
    # so have to re-implement it.
    return armory.data.datasets._generator_from_tfds(
        "cifar10:3.0.2",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=cifar10_preprocessing,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files)


def get_defense_path(defense_subdir):
    """Returns full defense path, given subdirectory.

    Args:
        defense_subdir: name of defense subdirectory, for example "defense_baseline"

    Returns:
        Full path of the defense directory.
    """
    module_filename = inspect.getfile(inspect.currentframe())
    module_dirname = os.path.dirname(os.path.abspath(module_filename))
    root_dir = os.path.abspath(os.path.join(module_dirname, '..'))
    return os.path.join(root_dir, defense_subdir)


class AttackWrapper(EvasionAttack):
    """ART wrapper for attacks."""

    _estimator_requirements = ()

    def __init__(self, *args, **kwargs):
        super().__init__(estimator='placeholder')
        self._targeted = None
        defense_path = get_defense_path(kwargs['defense'])
        attack_name = kwargs['attack']
        if not attack_name.endswith('.py'):
            attack_name = attack_name + '.py'
        defense_model, attack_cls, task_def, _ = load_defense_and_attack(
            defense_path, attack_name, torch_model=None)
        self._model = defense_model
        self._attack_cls = attack_cls
        self._task_def = task_def

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        assert y is not None, 'Labels has to be provided.'
        return self._attack_cls(self._task_def).attack(self._model, x, y)


class DefenseWrapper(TensorFlowV2Classifier):
    """ART wrapper for defenses."""

    def __init__(self, model):
        super().__init__(
            lambda x: model.classify(x),
            nb_classes=NUM_CLASSES,
            input_shape=INPUT_SHAPE,
            clip_values=(0., 1.),
            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def set_learning_phase(self, train):
        # Without overloading this method, armory does not work
        pass


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    del wrapper_kwargs
    del weights_file
    defense_name = model_kwargs['defense']
    defense_path = get_defense_path(defense_name)
    model, _, _, dataset_name = load_defense_and_attack(
        defense_path, 'attack_linf.py', torch_model=None)
    assert dataset_name == 'cifar10', 'Only CIFAR10 dataset is supported for Armory.'
    # TODO: maybe add support for adversarial detector
    return DefenseWrapper(model)
