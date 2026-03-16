# Copyright (C) 2026 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from git_project import GitProject
from options import options
from utils.logger import log
from utils.result import Result
from utils.utils import run

from .base import Benchmark, Suite, TracingType


class PyTorchSuite(Suite):
    def __init__(self) -> None:
        self._project = None

    def name(self) -> str:
        return "PyTorch benchmarks"

    def git_url(self) -> str:
        return "https://github.com/lslusarczyk/pytorch.git"

    def git_ref(self) -> str:
        return "graph-native-integration"

    def setup(self) -> None:
        if options.sycl is None:
            return

    def benchmarks(self) -> list[Benchmark]:
        if options.pytorch is not None:
            return [
                PyTorchBenchmark(self, "KernelSubmitSingleQueue", "us"),
                PyTorchBenchmark(self, "KernelSubmitMultiQueue", "us"),
                PyTorchBenchmark(self, "KernelSubmitMemoryReuse", "us"),
            ]


class PyTorchBenchmark(Benchmark):
    """PyTorch benchmark that runs benchmark scripts from a pre-built PyTorch repository."""

    def __init__(self, suite: PyTorchSuite, bench_name: str, unit: str = "us"):
        super().__init__(suite)
        self._bench_name = bench_name
        self._unit = unit

    def name(self) -> str:
        return f"pytorch {self._bench_name}"

    def display_name(self) -> str:
        return self._bench_name

    def get_tags(self) -> list[str]:
        return ["pytorch"]

    def lower_is_better(self) -> bool:
        return True

    def enabled(self) -> bool:
        return options.sycl is not None and options.pytorch is not None

    @property
    def _pytorch_root(self) -> Path:
        return Path(options.pytorch)

    @property
    def _python(self) -> str:
        venv_python = self._pytorch_root / "venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else "python3"

    @property
    def _bench_script(self) -> Path:
        return self._pytorch_root / "benchmarks" / f"{self._bench_name}.py"

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        command = [self._python, str(self._bench_script)]
        result_str = self.run_bench(
            command, env_vars or {}, run_trace=run_trace, force_trace=force_trace
        )
        value = float(result_str.strip())
        return [
            Result(
                label=self.name(),
                value=value,
                command=command,
                env=env_vars or {},
                unit=self._unit,
            )
        ]
