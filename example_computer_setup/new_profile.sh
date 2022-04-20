#!/bin/bash
set -e

profile=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if ! command -v verdi &> /dev/null
then
    echo "'verdi' command not found: did you activate the Conda environment where Aiida is installed?"
    exit 1
fi

if [ -z "$profile" ]
then
    echo "Usage: new_profile.sh <profile_name>"
    exit 1
fi

# Ensure profile is lowercase only

function lowered () {
    echo $1 | tr '[:upper:]' '[:lower:]'
}

if [ "$profile" != "$(lowered $profile)" ]
then
    echo "Profile name '$profile' is not lowercase"
    exit 1
fi

verdi quicksetup --profile $profile

for config_file in "$SCRIPT_DIR"/computers/*.yaml; do
    computer=$(basename $config_file .yaml)

    # -n to use default values that are not included
    # in the config file (this includes "username").
    verdi --profile $profile computer setup -n --config $config_file

    if [ $computer = localhost ]; then
        verdi --profile $profile computer configure core.local $computer -n --safe-interval 0
    else
        verdi --profile $profile computer configure core.ssh $computer -n --config "$SCRIPT_DIR/computers/ssh_transport.yaml"
    fi

    "$SCRIPT_DIR/add_extras.py" --profile $profile --config $config_file

done
