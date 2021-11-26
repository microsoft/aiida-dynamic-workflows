# aiida-dynamic-workflows
An AiiDA plugin for dynamically composing workflows from Python functions that run as CalcJobs.

**This is experimental, pre-alpha software**.


## Prerequisites
An environment where the _development_ version of AiiDA is installed.
This plugin makes use of a bugfix on the development branch, which will
not be included in an AiiDA release until v2.0.


## Installing
As pre-alpha software, this package is **not** released on PyPI.
Currently the only way to install the plugin is to clone the
repository and use `pip`:
```bash
pip install -e .
```


## Initialization
This plugin uses Conda for managing Python environments on remote computers.
Any Computers that you use with this plugin must have a `conda_dir` property
that contains an absolute path to the Conda directory on the machine
(typically something like `/home/{username}/miniconda3`.
The `add_extras.py` script in `example_cluster_setup/` can help you with this


## Examples
The [`examples/`](./examples) directory contains Jupyter notebooks that illustrate the main
features of `aiida-dynamic-workflows`. The notebooks are in Markdown format, and so require
the Jupyter plugin [jupytext](https://jupytext.readthedocs.io/en/latest/) in order to run them.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
