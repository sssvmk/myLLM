"""Subprocess code sandbox (SB-1..SB-3). Network isolation is the host's job (SB-2)."""
from __future__ import annotations

import os
import resource
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ExecResult:
    passed: bool
    status: str          # pass | fail | timeout | error
    returncode: int
    stderr_tail: str


class Sandbox:
    def __init__(self, python_executable: str, timeout_s: float, memory_mb: int, max_open_files: int,
                 max_processes: int, max_parallel: int, env_allowlist: Sequence[str], isolated_host_confirmed: bool):
        if not isolated_host_confirmed:
            raise PermissionError("code_execution.isolated_host_confirmed must be true to run generated code (SB-2)")
        self.python, self.timeout_s, self.memory_mb = python_executable, timeout_s, memory_mb
        self.max_open_files, self.max_processes, self.max_parallel = max_open_files, max_processes, max_parallel
        self.env = {k: os.environ[k] for k in env_allowlist if k in os.environ}

    @classmethod
    def from_config(cls, c) -> "Sandbox":
        return cls(c.python_executable, c.timeout_s, c.memory_mb, c.max_open_files, c.max_processes,
                   c.max_parallel, c.env_allowlist, c.isolated_host_confirmed)

    def _limits(self):
        cpu = int(self.timeout_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        mem = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NOFILE, (self.max_open_files, self.max_open_files))
        resource.setrlimit(resource.RLIMIT_NPROC, (self.max_processes, self.max_processes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    def run(self, program: str) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="pf_sbx_") as d:
            path = os.path.join(d, "prog.py")
            with open(path, "w") as f:
                f.write(program)
            try:
                p = subprocess.run([self.python, "-I", path], cwd=d, env=self.env, capture_output=True,
                                   timeout=self.timeout_s, preexec_fn=self._limits, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return ExecResult(False, "timeout", -9, "")
            except OSError as e:
                return ExecResult(False, "error", -1, str(e))
            ok = p.returncode == 0                                     # SB-3
            return ExecResult(ok, "pass" if ok else "fail", p.returncode, p.stderr.decode(errors="replace")[-500:])

    def run_many(self, programs: List[str]) -> List[ExecResult]:
        with ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
            return list(ex.map(self.run, programs))
