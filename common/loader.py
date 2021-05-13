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

"""Code for loading attacks and defenses."""

import importlib.util
import inspect
import os

from common.framework import DefenseModel, Attack


def load_module(module_name, module_path):
    """Loads module from specific path and imports it as specified name."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_subclasses_in_module(module, base_cls):
    """Finds subclasses of given class in the module."""
    name_values = inspect.getmembers(
        module,
        lambda x: inspect.isclass(x) and not inspect.isabstract(x) and (base_cls in inspect.getmro(x)))
    return [v for _, v in name_values]


def load_defense_and_attack(defense_path, attack_name, torch_model):
    """Load modules with defense and attack.

    Args:
        defense_path: path to directory with defense,
            defense code expected to be in model.py file in this directory.
        attack_name: name of the attack file in the defense directory.

    Returns:
        defense_model: instance of defense model.
        attack_cls: python class with attack.
        task_def: task definition.
        dataset_name: name of the dataset to use.
    """
    # Load modules
    task_def_module = load_module('defense.task_definition',
                                  os.path.join(defense_path, 'task_definition.py'))
    defense_model_module = load_module('defense.model',
                                       os.path.join(defense_path, 'model.py'))
    attack_module = load_module('defense.attack',
                                os.path.join(defense_path, attack_name))

    # Finding task definition
    if attack_name not in task_def_module.TASKS:
        print('Task definition not found for attack {0}'.format(attack_name))
        exit(1)
    task_def = task_def_module.TASKS[attack_name]

    # Loading defense model
    defense_classes = find_subclasses_in_module(defense_model_module, DefenseModel)
    if torch_model:
        defense_classes = [x for x in defense_classes if 'Torch' in x.__name__]
    else:
        defense_classes = [x for x in defense_classes if 'Torch' not in x.__name__]
    if not defense_classes or len(defense_classes) > 1:
        print('Defense {0} must have exactly one class implementing DefenseModel'.format(defense_path))
        print('Found: ', defense_classes)
        exit(1)
    defense_model = defense_classes[0]()

    # Loading attack
    attack_classes = find_subclasses_in_module(attack_module, Attack)
    if not attack_classes or len(attack_classes) > 1:
        print('Attack {0} must have exactly one class implementing Attack'.format(
            os.path.join(defense_path, attack_name)))
        print('Found: ', attack_classes)
        exit(1)
    attack_cls = attack_classes[0]

    return defense_model, attack_cls, task_def, task_def_module.DATASET
