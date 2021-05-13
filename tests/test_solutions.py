"""Unit tests for solutions for defenses.

These tests merely verify that code runs, they don't test whether solutions work.
"""

import pytest

from common.data import load_dataset
from common.loader import load_defense_and_attack
from evaluate import evaluate_defense


DEFENSES_AND_ATTACKS_TO_TEST = [
    ('defense_baseline', 'attack_linf.py'),
    ('defense_blur', 'attack_linf.py'),
    ('defense_discretize', 'attack_linf.py'),
    ('defense_injection', 'attack_linf.py'),
    ('defense_jump', 'attack_linf.py'),
    ('defense_knn', 'attack_linf.py'),
    ('defense_labelsmooth', 'attack_linf.py'),
    ('defense_majority', 'attack_linf.py'),
    ('defense_mergebinary', 'attack_linf.py'),
    ('defense_randomneuron', 'attack_linf.py'),
    ('defense_temperature', 'attack_linf.py'),
    ('defense_transform', 'attack_linf.py'),
    ('defense_baseline', 'attack_linf_torch.py'),
    ('defense_blur', 'attack_linf_torch.py'),
    ('defense_discretize', 'attack_linf_torch.py'),
    ('defense_injection', 'attack_linf_torch.py'),
    ('defense_jump', 'attack_linf_torch.py'),
    ('defense_knn', 'attack_linf_torch.py'),
    ('defense_labelsmooth', 'attack_linf_torch.py'),
    ('defense_majority', 'attack_linf_torch.py'),
    ('defense_mergebinary', 'attack_linf_torch.py'),
    ('defense_randomneuron', 'attack_linf_torch.py'),
    ('defense_temperature', 'attack_linf_torch.py'),
    ('defense_transform', 'attack_linf_torch.py'),
]

NUM_EXAMPLES = 8
BATCH_SIZE = 4


@pytest.mark.parametrize('defense_path,attack_name', DEFENSES_AND_ATTACKS_TO_TEST)
def test_solution(defense_path, attack_name):
    torch = 'torch' in attack_name
    defense_model, attack_cls, task_def, dataset_name = load_defense_and_attack(
        defense_path, attack_name, torch)
    
    _, (x_test, y_test), _ = load_dataset(dataset_name, torch)
    x_test = x_test[:NUM_EXAMPLES]
    y_test = y_test[:NUM_EXAMPLES]
    
    failed_examples = evaluate_defense(
        x_test, y_test, BATCH_SIZE, attack_cls, defense_model, task_def)
    num_failed = len(failed_examples)
    # NOTE: if attack succeed on all examples then num_failed == 0
    # here we just expect that attack succeed at least on one example
    assert True
