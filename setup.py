# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from setuptools import setup


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    from importlib.util import module_from_spec, spec_from_file_location
    import os

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("aiida_dynamic_workflows")

# All other options are specified in 'setup.cfg'; the version has to be
# determined dynamically from git tags (using 'miniver'), so it needs
# to be done here.
setup(
    version=version, cmdclass=cmdclass,
)
