#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import aiida
from aiida.cmdline.utils import echo
import aiida.orm
import click
import yaml


@click.command()
@click.option("--profile", help="Aiida profile")
@click.option("--config", required=True, help="Config file for computer")
def main(profile, config):
    """Add extra properties to the computer defined in 'config'."""
    aiida.load_profile(profile)

    with open(config) as f:
        config = yaml.safe_load(f)

    label = config["label"]

    echo.echo_info(f"Adding extra properties to computer {label}")

    extras = config.get("extras", dict())

    computer = aiida.orm.load_computer(label)
    for k, v in extras.items():
        computer.set_property(k, str(v))
    computer.store()

    echo.echo_success(f"Added the following properties to {label}: {extras}")

    if not "conda_dir" in extras:
        echo.echo_info(f"Setting the conda directory for computer {label}")
        conda_dir = get_conda_dir(computer)
        computer.set_property("conda_dir", conda_dir)
        computer.store()

        echo.echo_success(f"Set the Conda directory on {label} to '{conda_dir}'")


def get_conda_dir(computer):
    """Return the Conda directory for the given computer.

    First we try to determine the Conda directory automatically by
    activating the "base" environment and getting $CONDA_PREFIX.

    If that fails we simply prompt the user.
    """
    label = computer.label
    with computer.get_transport() as t:
        rv, stdout, stderr = t.exec_command_wait(
            "set -e; conda activate base; echo $CONDA_PREFIX"
        )
        conda_dir = stdout.strip() or None
        if not conda_dir:
            echo.echo_warning(
                "Failed to automatically determine Conda directory "
                f"for {label} (the computer said: '{stderr}')"
            )

            while not conda_dir:
                x = click.prompt(f"Enter your conda directory for {label}")
                if t.isdir(x):
                    conda_dir = x
                else:
                    echo.echo_warning(f"'{x}' is not a directory on {label}")
        return conda_dir


if __name__ == "__main__":
    main()
