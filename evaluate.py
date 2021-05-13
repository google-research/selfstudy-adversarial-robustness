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

"""Evaluation script.

Following command runs evaluation
on `attack_linf.py` of `defense_baseline` task:

    python evaluate.py defense_baseline/attack_linf.py

By default evaluation is run on entire test set.
To limit number of example (which might be useful for debugging)
use flat --num_examples:

    python evaluate.py --num_examples=100 defense_baseline/attack_linf.py

"""

import importlib.util
import inspect
import os

import numpy as np
from absl import app
from absl import flags

import common.data as data
from common.framework import DefenseModel, Attack
from common.loader import load_defense_and_attack


def show_usage():
    """Shows usage of the script."""
    print('Usage:')
    print('  python [--num_examples=N] [--example_list=2,3,8] [--batch_size=N] [--verbose] [--test] DEFENSE/ATTACK.py')
    print()
    print('Example:')
    print('  python evaluate.py defense_baseline/attack_linf.py')
    print()


def parse_argv(argv):
    """Parses command line arguments and extracts defense path and name of the attack."""
    if len(argv) != 2:
        print("Did not receive an attack path to run.")
        return False, None, None
    root_dir = os.path.dirname(os.path.realpath(argv[0]))
    defense_path, attack_name = os.path.split(argv[1])
    _, defense_name = os.path.split(defense_path)
    if not defense_name:
        print("Attack path must be of the form defenseXXX_YYY/attack_ZZZ.py.")
        return False, None, None
    defense_path = os.path.join(root_dir, defense_name)
    attack_path = os.path.join(defense_path, attack_name)
    if not os.path.isdir(defense_path):
        print("The defense path", defense_path, "does not exist.")
        return False, None, None
    if not os.path.exists(attack_path):
        print("The file", attack_path, "does not exist.")
        return False, None, None

    return True, defense_path, attack_name


def evaluate_defense(x_test,
                     y_test,
                     batch_size,
                     attack_cls,
                     defense_model,
                     task_def,
                     verbose=False):
    """Evaluate defense on given test set.

    Args:
        x_test: numpy array with images.
        y_test: numpy array with labels.
        batch_size: batch size.
        attack_cls: attack class.
        defense_model: instance of defense model.
        task_def: instance of task definition.
        verbose: if True then print about succeeded/failed attack on each example.

    Returns:
        List of indices of all examples where attack failed to construct adversarial example.
    """
    failed_examples = []

    attack_model = attack_cls(task_def)

    for i in range(0, x_test.shape[0], batch_size):
        x_batch = np.array(x_test[i:i+batch_size])
        y_batch = np.array(y_test[i:i+batch_size])

        advx_batch = attack_model.attack(defense_model,
                                         x_batch,
                                         y_batch)

        assert isinstance(advx_batch, np.ndarray), \
            "Attack must return a NumPy Array; not Tensor."

        x_pred_batch, x_detector_batch = defense_model(x_batch)
        advx_pred_batch, advx_detector_batch = defense_model(advx_batch)

        for j in range(len(x_batch)):

            assert advx_batch[j].shape == x_batch[j].shape
            assert len(advx_pred_batch[j]) == 10
            
            if np.any(np.isnan(advx_batch[j])):
                result, why = False, "contained NaN"
            else:
                result, why = task_def.evaluate(defense=defense_model,
                                                example_idx=i + j,
                                                true_label=y_batch[j],
                                                src_example=x_batch[j],
                                                adv_example=advx_batch[j],
                                                src_pred=x_pred_batch[j],
                                                adv_pred=advx_pred_batch[j],
                                                src_detector=x_detector_batch[j],
                                                adv_detector=advx_detector_batch[j])
            if result:
                if verbose:
                    print('Example', i + j, 'succeeded')
            else:
                if verbose:
                    print('Example', i + j, 'failed:', why)
                failed_examples.append(i + j)

    return failed_examples

def evaluate_clean(defense_model, x_test, y_test, batch_size):
    preds = []
    is_ok = []
    for i in range(0, len(x_test), batch_size):
        preds.extend(np.argmax(defense_model.classify(x_test[i:i+batch_size]),axis=1))
        is_ok.extend(defense_model.detect(x_test[i:i+batch_size]) < defense_model.threshold)
    print('Defense test accuracy', np.mean((preds==y_test) * np.array(is_ok)))

def tune_fpr(defense_model, x_test, y_test, batch_size, fpr):
    detect_scores = []
    for i in range(0, len(x_test), batch_size):
        detect_scores.extend(defense_model.detect(x_test[i:i+batch_size]))
    print("Set threshold to",
          sorted(detect_scores)[-int(fpr*len(detect_scores))],
          "to achieve a FPR of",
          fpr)
    

def main(argv):
    # Parse arguments
    success_parse_argv, defense_path, attack_name = parse_argv(argv)
    if not success_parse_argv:
        show_usage()
        return
    print('Evaluation parameters:')
    print('  Defense path: ', defense_path)
    print('  Attack name: ', attack_name)

    TORCH = 'torch' in attack_name

    defense_model, attack_cls, task_def, dataset_name = load_defense_and_attack(
        defense_path, attack_name, TORCH)

    if FLAGS.ignore_threshold:
        task_def.threshold = np.inf

    # Loading dataset
    print('  Dataset: ', dataset_name)
    _, (x_test, y_test), _ = data.load_dataset(dataset_name, TORCH)

    use_examples = np.arange(len(x_test))
    if FLAGS.example_list is not None:
        # We've got a specific set of examples to attack
        use_examples = list(map(int, FLAGS.example_list.split(",")))
    else:
        # Attack a sequential set of examples
        if FLAGS.num_examples > 0:
            use_examples = np.arange(FLAGS.num_examples)

    x_test = x_test[use_examples]
    y_test = y_test[use_examples]

    print('  Number of examples:', len(use_examples))

    batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else len(x_test)

    if FLAGS.test:
        evaluate_clean(defense_model, x_test, y_test, batch_size)
        exit(0)

    if FLAGS.tune_fpr is not None:
        tune_fpr(defense_model, x_test, y_test, batch_size, FLAGS.tune_fpr)
        exit(0)
        
    failed_examples = evaluate_defense(
        x_test, y_test, batch_size, attack_cls, defense_model, task_def, FLAGS.verbose)

    if len(failed_examples) == 0:
        print('SUCCESS!')
    else:
        print('FAIL')
        print('{0} out of {1} examples failed task'.format(len(failed_examples), len(x_test)))
        print('Indices of failed examples: ', [use_examples[x] for x in failed_examples])
        print('To re-run the attack on just these examples pass',
              '--example_list='+",".join(str(use_examples[x]) for x in failed_examples))
        if not FLAGS.verbose:
            print("Run with --verbose for more information on why each adversarial example failed.")


if __name__ == '__main__':
    flags.DEFINE_integer('num_examples', 100,
                         'Number of examples to use for eval. -1 means use entire test set')
    
    flags.DEFINE_string('example_list', None,
                        'List of examples indices to attack.')
    
    flags.DEFINE_integer('batch_size', -1,
                         'Batch size used to run evaluation. '
                         'Negative value means use all images from the dataset at once.')
    
    flags.DEFINE_bool('test', False,
                      'If true then only evaluate defense accuracy on clean test examples.')

    flags.DEFINE_float('tune_fpr', None,
                      'If true then compute the threshold that would set the false positive rate to the given value.')
    
    flags.DEFINE_bool('verbose', False,
                      'Generate verbose logging')
    
    flags.DEFINE_bool('ignore_threshold', False,
                      'Ignore the distortion bound threshold (for debugging).')
    
    FLAGS = flags.FLAGS

    app.run(main)
