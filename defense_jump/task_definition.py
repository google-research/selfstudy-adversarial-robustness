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

"""Task definition for evaluator code."""

import numpy as np
from common.framework import LpTask


DATASET = 'cifar10'
L2_THRESHOLD = 0.5
LINF_THRESHOLD = 4.0 / 255.0


TASKS = {
    'attack_linf.py': LpTask(np.inf, LINF_THRESHOLD),
    'attack_linf_torch.py': LpTask(np.inf, LINF_THRESHOLD),
    'attack_linf_soln.py': LpTask(np.inf, LINF_THRESHOLD),
    'attack_linf_torch_soln.py': LpTask(np.inf, LINF_THRESHOLD),
    'attack_l2.py': LpTask(2, L2_THRESHOLD),
    'attack_l2_torch.py': LpTask(2, L2_THRESHOLD),
    'attack_l2_soln.py': LpTask(2, L2_THRESHOLD),
    'attack_l2_torch_soln.py': LpTask(2, L2_THRESHOLD),
}
