# Integration with Armory

This directory contains scripts for integration with [Armory](https://github.com/twosixlabs/armory).

## Usage

Installation:

```
pip install armory-testbed==0.11.1
```

Run armory on specific defense and attack:

```
./armory_compat/run_armory.sh defense_baseline attack_linf_soln.py
```

Additional parameters could be passed to armory in a following way:

```
# Pass --num-eval-batches=2 to armory
./armory_compat/run_armory.sh defense_baseline attack_linf_soln.py --num-eval-batches=2
```
