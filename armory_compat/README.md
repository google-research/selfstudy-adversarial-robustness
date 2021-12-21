# Integration with Armory

This directory contains scripts for integration with [Armory](https://github.com/twosixlabs/armory).

## Usage

Installation:

```
pip install armory-testbed
```

Run armory on specific defense and attack:

```
./armory_compat/run_armory.sh defense_baseline attack_linf.py
```

```
./armory_compat/run_armory.sh defense_baseline attack_linf_torch.py
```

Note that for Pytorch attacks, 'torch' must be present the attack filename
for the armory compatibility wrapper. Similarly for 'l2' attacks,
'l2' must be present in the attack filename.


Additional parameters could be passed to armory in a following way:

```
# Pass --num-eval-batches=2 to armory
./armory_compat/run_armory.sh defense_baseline attack_linf.py --num-eval-batches=2
```

Common additional parameter include: --no-gpu for systems without a gpu.
