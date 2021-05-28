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

"""Shared code for attacks and defenses."""

import abc
import inspect
import os
import numpy as np
import tensorflow as tf


class DefenseModel(abc.ABC):
    """Base class which defines interface for all defense models.

    This base class supports two options for defense model:
    - defense model which only returns predictions of target class.
    - defense model with detector of adversarial examples. Such model
      returns both output of detector and predictions of target class.

    If subclass need to implement model with detector then it should override
    `detect` method. Otherwise `detect` should be kept as is.

    Note that DefenseModel only provides method for forward pass and does
    not provide method to compute gradients. This is intentional and it's done
    because gradients for most of the defenses in the tutorials are not useful
    for adversarial examples search.
    """

    @abc.abstractmethod
    def classify(self, x):
        """Performs forward pass on the model.

        Args:
            x: input as either numpy array or Tensorflow Tensor with shape [B, H, W, C],
              each value of the input should be float in [0, 1] range.

        Returns:
            NumPy array with normalized probabilities returned by the model.
            Shape of return value is [B, NUM_LABELS].

        This method has to be implemented in the subclass.
        """
        pass

    """
    Threshold for detecting if an input is adversarial, by default 1/2.
    """
    threshold = 0.5

    def detect(self, x):
        """Perform forward pass on the adversarial detector.

        Args:
            x: input as either numpy array or Tensorflow Tensor with shape [B, H, W, C],
              each value of the input should be float in [0, 1] range.

        Returns:
            NumPy array with shape [B] of floats from [-inf,inf] with a higher value
            when the input is more likely to be adversarial, and should be rejected.

        If your model does not provide a detector then leave it as is.
        If your model provides detector then override this method in the subclass.
        """
        return np.zeros(x.shape[0])

    def __call__(self, x):
        """Performs both inference and detection classification on the model.

        Args:
            x: input as numpy array with shape [B, H, W, C] or [H, W, C]
              each value of the input should be float in [0, 1] range.

        Returns:
            numpy array with normalized predictions returned by the model.
            Shape of return value is [B, NUM_LABELS] if input is batched or [NUM_LABELS] otherwise.
            All predictions are in [0, 1] range.

        This is a convenience method which performs some basic argument checks
        and defers work to `forward`.
        Thus subclasses should implement forward pass logic should be in `forward` method instead.
        """
        assert len(x.shape) == 3 or len(x.shape) == 4
        assert isinstance(x, np.ndarray)
        if len(x.shape) == 3:
            # x is a single example
            x = np.expand_dims(x, axis=0)
            return self.classify(x), self.detect(x)
        else:
            # x is a batch of examples
            return self.classify(x), self.detect(x)


class Attack(abc.ABC):
    """Base class which defines interface for all attacks.

    Environment will create new instance of attack for each adversarial example.
    """

    def __init__(self, task):
        """
        Constructs an Attack class that generates adversarial examples
        for a particular TaskDefinition.
        """
        self.task = task

    @abc.abstractmethod
    def attack(self, model, x, y):
        """Peforms adversarial attack on batch of examples.

        Args:
            model: instance of DefenseModel.
            x: numpy array with batch of input examples.
               Shape is [B, H, W, C], values are in [0, 1] range.
            y: numpy array with shape [B], which contains true labels of examples.

        Returns:
            numpy array with adversarial examples, same shape and type as input x.
        """
        pass


class NonBatchedAttack(Attack):
    """Base class for attacks which are implemented for single example instead batch.

    This is a subclass of `Attack` which simplifies process of writing attacks which can not
    be batched easily. Instead of implementing attack on a batch of examples, subclasses of
    this class need to implement attack on single example.

    Nevertherless, to make evaluation faster, it's recommended to subclass directly
    from `Attack` and implement attack on the batch whenever it's possible and easy to do.
    """

    def attack(self, model, x, y):
        result = []
        for i in range(len(x)):
            adv_ex = self.attack_one_example(model, x[i], y[i])
            result.append(np.expand_dims(adv_ex, axis=0))
        return np.concatenate(result, axis=0)

    @abc.abstractmethod
    def attack_one_example(self, model, x, y):
        """Peforms adversarial attack on a single example.

        Args:
            model: instance of DefenseModel.
            x: numpy array with input example.
               Shape is [H, W, C], values are in [0, 1] range.
            y: true label of the input example.

        Returns:
            numpy array with adversarial example, same shape and type as input x.
        """
        pass


class TaskDefinition(abc.ABC):
    """Base class for all task definitions."""

    @abc.abstractmethod
    def evaluate(self, example_idx, true_label,
                 src_example, adv_example,
                 src_pred, adv_pred,
                 src_detector, adv_detector):
        """Evaluate attack on one example.

        Args:
            example_idx: index of the example in the dataset.
                Could be useful if certain examples in the dataset need to be skipped or
                require special handling.
            true_label: true label of the example.
            src_example: numpy array with source example, shape [H, W, C].
            adv_example: numpy array with adversarial example, shape [H, W, C].
            src_pred: probabilities vector with model prediction on source example.
            adv_pred: probabilities vector with model prediction on adversarial example.
            src_detector: probability that input is adversarial as returned by detector on source example.
            adv_detector: probability that input is adversarial as returned by detector on adversarial example.

        Returns:
            success: True is task passed on the example, False otherwise.
            reason: If task failed then the reason why.

        This method should verify following:
        - generated adversarial example is within required distance from the original example
        - class label was flipped on adversarial example compared to original example
        - detector returns low probability of adversarial example being adversarial (if detector is available)
        If all conditions are met then method should return True.
        If any condition is not satisfied then method should return False.
        """
        pass


class LpTask(TaskDefinition):
    """Task definition for adversarial examples in L_{p} ball."""

    def __init__(self, norm, threshold):
        """Initializes Lp task definition.

        Args:
            norm: norm type to use. Interpretation is same as `ord` argument of `np.linalg.norm` function
              when `np.linalg.norm` is used for vectors.
        """
        self.threshold = threshold
        self.norm = norm

    def evaluate(self, defense, example_idx, true_label,
                 src_example, adv_example,
                 src_pred, adv_pred,
                 src_detector, adv_detector):
        # Verify that the label is now incorrect
        if np.argmax(adv_pred) == true_label:
            return False, "Label {} matches true label {}".format(np.argmax(adv_pred), true_label)

        # Verify that example is within the allowed Lp norm
        distortion = np.linalg.norm((src_example - adv_example).flatten(), ord=self.norm)
        if distortion > self.threshold + 1e-3:
            return False, "Distortion {} exceeds bound {}".format(distortion, self.threshold)

        # Verify that example is within [0, 1] domain of input images
        min_val = np.min(adv_example)
        max_val = np.max(adv_example)
        if min_val < 0.0:
            return False, "Adversarial example min value {} is outside of [0, 1] range".format(min_val)
        if max_val > 1.0:
            return False, "Adversarial example max value {} is outside of [0, 1] range".format(max_val)

        # Verify that it's not detected as adversarial
        if adv_detector > defense.threshold:
            return False, "Adversarial example rejected by detector with score {}.".format(adv_detector)

        return True, None


def get_checkpoint_abs_path(rel_path):
    """Converts relative path of checkpoint-related resource to absolute path.

    Args:
        rel_path: path of the file relative to repository root.

    Returns:
        Absolute path of the file.

    Let's say that tutorials are located in `/path/to/tutorials/` directory,
    which means that full path of this file is `/path/to/tutorials/common/defense.py`.
    Then following call to this method:
        `get_checkpoint_abs_path('checkpoints/model-1')`
    will return `/path/to/tutorials/checkpoints/model-1`
    """
    module_filename = inspect.getfile(inspect.currentframe())
    module_dirname = os.path.dirname(os.path.abspath(module_filename))
    tutorials_root = os.path.abspath(os.path.join(module_dirname, '..'))
    return os.path.join(tutorials_root, rel_path)
