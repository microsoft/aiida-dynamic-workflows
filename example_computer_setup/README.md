# Aiida profiles for use with aiida-dynamic-workflows

aiida-dynamic-workflows assumes Conda is used to manage the Python environments
on the Computers

To get started, modify the `hostname` and `work_dir` keys in `computers/cluster.yaml`
to point to a Slurm cluster.
Then run run `./new_profile.sh <profile_name>` to create a new Aiida profile with that
Computer set up.
