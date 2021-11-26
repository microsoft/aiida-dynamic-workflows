# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Aiida Parsers for interpreting the output of arbitrary Python functions."""

import os.path

import aiida.engine
import aiida.parsers

from . import common
from .common import MapSpec
from .data import PyRemoteArray, PyRemoteData, array_shape

# TODO: unify 'PyCalcParser' and 'PyMapParser': they are identical except
#       for the type of the outputs (PyRemoteData vs. PyRemoteArray).


class PyCalcParser(aiida.parsers.Parser):
    """Parser for a PyCalcJob."""

    def parse(self, **kwargs):  # noqa: D102

        calc = self.node

        def retrieve(value_file):
            # No actual retrieval occurs; we just store a reference
            # to the remote value.
            return PyRemoteData.from_remote_data(
                calc.outputs.remote_folder, value_file,
            )

        exception_file = "__exception__.pickle"
        remote_folder = calc.outputs["remote_folder"]
        remote_files = remote_folder.listdir()
        has_exception = exception_file in remote_files

        exit_code = None

        # If any data was produced we create the appropriate outputs.
        # If something went wrong the exit code will still be non-zero.
        output_folder = remote_folder.listdir("__return_values__")
        for r in calc.inputs.func.returns:
            filename = f"{r}.pickle"
            path = os.path.join("__return_values__", filename)
            if filename in output_folder:
                self.out(f"return_values.{r}", retrieve(path))
            else:
                exit_code = self.exit_codes.MISSING_OUTPUT

        try:
            job_infos = calc.computer.get_scheduler().parse_detailed_job_info(
                calc.get_detailed_job_info()
            )
        except AttributeError:
            pass
        else:
            (job_info,) = job_infos
            if job_info["State"] == "FAILED":
                exit_code = self.exit_codes.NONZERO_EXIT_CODE

        if has_exception:
            self.out("exception", retrieve(exception_file))
            exit_code = self.exit_codes.USER_CODE_RAISED

        if exit_code is not None:
            calc.set_exit_status(exit_code.status)
            calc.set_exit_message(exit_code.message)
            return exit_code


class PyMapParser(aiida.parsers.Parser):
    """Parser for a PyMapJob."""

    def parse(self, **kwargs):  # noqa: D102

        calc = self.node

        mapspec = MapSpec.from_string(calc.get_option("mapspec"))
        mapped_parameter_shapes = {
            k: array_shape(v)
            for k, v in calc.inputs.kwargs.items()
            if k in mapspec.parameters
        }
        expected_shape = mapspec.shape(mapped_parameter_shapes)
        remote_folder = calc.outputs["remote_folder"]
        has_exceptions = bool(remote_folder.listdir("__exceptions__"))

        def retrieve(return_value_name):
            return PyRemoteArray(
                computer=calc.computer,
                remote_path=os.path.join(
                    calc.outputs.remote_folder.get_remote_path(), return_value_name,
                ),
                shape=expected_shape,
                filename_template=common.array.filename_template,
            )

        exit_code = None

        # If any data was produced we create the appropriate outputs.
        # Users can still tell something went wrong from the exit code.
        for r in calc.inputs.func.returns:
            path = os.path.join("__return_values__", r)
            has_data = remote_folder.listdir(path)
            if has_data:
                self.out(f"return_values.{r}", retrieve(path))
            else:
                exit_code = self.exit_codes.MISSING_OUTPUT

        try:
            job_infos = calc.computer.get_scheduler().parse_detailed_job_info(
                calc.get_detailed_job_info()
            )
        except AttributeError:
            pass
        else:
            if any(j["State"] == "FAILED" for j in job_infos):
                exit_code = self.exit_codes.NONZERO_EXIT_CODE

        if has_exceptions:
            self.out("exception", retrieve("__exceptions__"))
            exit_code = self.exit_codes.USER_CODE_RAISED

        if exit_code is not None:
            calc.set_exit_status(exit_code.status)
            calc.set_exit_message(exit_code.message)
            return exit_code
