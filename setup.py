# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from setuptools import find_packages, setup


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


plugin = "dynamic_workflows"
pkg = f"aiida_{plugin}"

classifiers = """\
    Development Status :: 2 - Pre-Alpha
    Intended Audience :: Science/Research
    Intended Audience :: Developers
    Programming Language :: Python :: 3 :: Only
    Topic :: Software Development
    Topic :: Scientific/Engineering
    Operating System :: Linux"""

version, cmdclass = get_version_and_cmdclass(pkg)

setup(
    name=pkg,
    author="Microsoft Quantum",
    description=(
        "Aiida plugin for dynamically defining workflows that execute Python functions"
    ),
    license="MIT",
    packages=find_packages("."),
    version=version,
    cmdclass=cmdclass,
    entry_points={
        "aiida.calculations": [
            f"{plugin}.PyCalcJob = {pkg}.calculations:PyCalcJob",
            f"{plugin}.PyMapJob = {pkg}.calculations:PyMapJob",
            f"{plugin}.merge_remote_arrays = {pkg}.calculations:merge_remote_arrays",
        ],
        "aiida.parsers": [
            f"{plugin}.PyCalcParser = {pkg}.parsers:PyCalcParser",
            f"{plugin}.PyMapParser = {pkg}.parsers:PyMapParser",
        ],
        "aiida.data": [
            f"{plugin}.PyData = {pkg}.data:PyData",
            f"{plugin}.PyArray= {pkg}.data:PyArray",
            f"{plugin}.PyRemoteData = {pkg}.data:PyRemoteData",
            f"{plugin}.PyRemoteArray = {pkg}.data:PyRemoteArray",
            f"{plugin}.PyOutline = {pkg}.data:PyOutline",
            f"{plugin}.PyFunction = {pkg}.data:PyFunction",
            f"{plugin}.Nil = {pkg}.data:Nil",
            f"{plugin}.PyException = {pkg}.data:PyException",
        ],
        "aiida.node": [
            f"process.workflow.{plugin}.WorkChainNode = {pkg}.workchains:WorkChainNode",
        ],
        "aiida.schedulers": [
            f"{plugin}.slurm = {pkg}.schedulers:SlurmSchedulerWithJobArray",
        ],
        "aiida.workflows": [
            f"{plugin}.PyWorkChain = {pkg}.workflow:PyWorkChain",
            f"{plugin}.RestartedPyMapJob = {pkg}.workchains:RestartedPyMapJob",
            f"{plugin}.RestartedPyCalcJob = {pkg}.workchains:RestartedPyCalcJob",
        ],
    },
    setup_requires=["reentry"],
    install_requires=[
        "aiida-core >=2.0.0.a1,<3.0.0",
        "toolz >=0.11.0,<1.0.0",
        "cloudpickle >=2.0.0,<3.0.0",
        "numpy",
        "graphviz",
    ],
    reentry_register=True,
    classifiers=[c.strip() for c in classifiers.split("\n")],
)
