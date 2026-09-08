# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""Verify pvapy.cli.hpcConsumer resolves from the same interpreter DashPVA launches."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pvapy", reason="pvapy not installed (standalone edition)")


def _run_probe(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result.stdout.strip()


def test_pvapy_cli_hpcconsumer_importable_from_sys_executable():
    """workflow.py invokes `sys.executable -m pvapy.cli.hpcConsumer`; this verifies it resolves."""
    out = _run_probe("import pvapy.cli.hpcConsumer as m; print(m.__file__)")
    assert out, "expected pvapy.cli.hpcConsumer.__file__ to be non-empty"
    assert Path(out).exists(), f"resolved path does not exist: {out}"


def test_pvapy_lives_under_sys_executable_prefix():
    """Catches the case where pvapy is on a different env than the launching interpreter."""
    exe_prefix = _run_probe("import sys; print(sys.prefix)")
    pvapy_file = _run_probe("import pvapy; print(pvapy.__file__)")

    exe_prefix_resolved = Path(exe_prefix).resolve()
    pvapy_resolved = Path(pvapy_file).resolve()

    assert str(pvapy_resolved).startswith(str(exe_prefix_resolved)), (
        f"pvapy at {pvapy_resolved} is not under interpreter prefix {exe_prefix_resolved} — "
        f"DashPVA's `sys.executable -m pvapy.cli.hpcConsumer` will fail at runtime."
    )
