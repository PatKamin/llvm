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

        if self._project is None:
            self._project = GitProject(
                self.git_url(),
                self.git_ref(),
                Path(options.workdir),
                "pytorch-benchmarks",
                use_installdir=False,
            )

        venv_dir = self._project.build_dir / "venv"
        python = str(venv_dir / "bin" / "python")
        if not venv_dir.exists():
            self._project.build_dir.mkdir(parents=True, exist_ok=True)
            run(["python3", "-m", "venv", str(venv_dir)])
            run([python, "-m", "pip", "install", "pytest"])
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements-build.txt",
                    "-r",
                    "requirements.txt",
                ],
                cwd=self._project.src_dir,
            )

        env_vars = {
            "USE_ONEMKL_XPU": "OFF",
            "TORCH_XPU_ARCH_LIST": "bmg",
            "CC": "gcc",
            "CXX": "g++",
            "USE_KINETO": "OFF",
            "USE_XPU": "1",
            "USE_CUDA": "0",
            "BUILD_TEST": "0",
            "CMPLR_ROOT": options.sycl,
            "CPATH": f"{options.sycl}/include",
            "OPENCLROOT": options.sycl,
        }
        run(
            [python, "-m", "pip", "install", "--no-build-isolation", "-v", "-e", "."],
            env_vars=env_vars,
            cwd=self._project.src_dir,
            add_sycl=True,
        )

    def benchmarks(self) -> list[Benchmark]:
        return [
            MockPyTorchBenchmark(self, "KernelSubmitSingleQueue", "us"),
            MockPyTorchBenchmark(self, "KernelSubmitMultiQueue", "us"),
            MockPyTorchBenchmark(self, "KernelSubmitMemoryReuse", "us"),
        ]


class MockPyTorchBenchmark(Benchmark):
    """Mock PyTorch benchmark that returns fixed fake results without requiring
    the actual PyTorch project to be cloned or built."""

    _MOCK_VALUE = 42.0

    def __init__(self, suite: PyTorchSuite, bench_name: str, unit: str = "us"):
        super().__init__(suite)
        self._bench_name = bench_name
        self._unit = unit

    def name(self) -> str:
        return f"pytorch mock {self._bench_name}"

    def display_name(self) -> str:
        return f"Mock {self._bench_name}"

    def description(self) -> str:
        return (
            f"Mock PyTorch benchmark for {self._bench_name}. "
            "Returns a fixed value and is used for testing the benchmark infrastructure "
            "without requiring a real PyTorch build."
        )

    def get_tags(self) -> list[str]:
        return ["pytorch", "mock"]

    def lower_is_better(self) -> bool:
        return True

    def enabled(self) -> bool:
        return options.sycl is not None

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        return [
            Result(
                label=self.name(),
                value=self._MOCK_VALUE,
                command=["mock"],
                env=env_vars or {},
                unit=self._unit,
            )
        ]
