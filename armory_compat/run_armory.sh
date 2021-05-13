#!/bin/bash
#
# Script which runs Armory with given attack and defense
#
# Usage:
#    armory_compat/run_armory.sh DEFENSE_NAME ATTACK_NAME [EXTRA_ARMORY_ARGS]
#
# Where:
#   DEFENSE_NAME - name of the defense
#   ATTACK_NAME - name of the attack file
#   EXTRA_ARMORY_ARGS - optional, additional args which will be passed to armory
#
# Examples:
#    armory_compat/run_armory.sh defense_baseline attack_linf_soln.py
#    armory_compat/run_armory.sh defense_baseline attack_linf_soln.py --num-eval-batches=2
#

# Change directory to root (in case this script is run from somewhere else)
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Process args
DEFENSE_NAME=$1
ATTACK_NAME=$2
shift 2

if [[ "${ATTACK_NAME}" == *"l2"* ]]; then
  METRIC="l2"
else
  METRIC="linf"
fi

# Create a json config
config_file=$(mktemp)

ATTACK="${ATTACK_NAME}" DEFENSE="${DEFENSE_NAME}" METRIC="${METRIC}" \
    envsubst < armory_compat/config_template.json > "${config_file}"

echo "Generated Armory config saved in ${config_file}"

# Run armory
armory run "${config_file}" "$@"
