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

"""Compare-and-swap contracts for TOML and database profile sources."""

import threading

import toml

from dashpva.utils.config.source import (
    ConfigSaveStatus,
    ConfigSource,
    DbProfileConfigSource,
)


def test_toml_stale_revision_cannot_overwrite_newer_save(tmp_path):
    path = tmp_path / "profile.toml"
    path.write_text(toml.dumps({"owner": "initial"}))
    source = ConfigSource(str(path))
    _, first_revision = source.load_snapshot()
    _, stale_revision = source.load_snapshot()

    first = source.replace_if_revision({"owner": "first"}, first_revision)
    stale = source.replace_if_revision({"owner": "stale"}, stale_revision)

    assert first.status is ConfigSaveStatus.SAVED
    assert first.revision
    assert stale.status is ConfigSaveStatus.CONFLICT
    assert toml.load(path) == {"owner": "first"}


def test_toml_concurrent_writers_have_exactly_one_winner(tmp_path):
    path = tmp_path / "profile.toml"
    path.write_text(toml.dumps({"owner": "initial"}))
    source = ConfigSource(str(path))
    _, revision = source.load_snapshot()
    barrier = threading.Barrier(3)
    results = []

    def write(owner):
        barrier.wait()
        results.append(
            source.replace_if_revision({"owner": owner}, revision).status
        )

    threads = [
        threading.Thread(target=write, args=("left",)),
        threading.Thread(target=write, args=("right",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert results.count(ConfigSaveStatus.SAVED) == 1
    assert results.count(ConfigSaveStatus.CONFLICT) == 1
    assert toml.load(path)["owner"] in {"left", "right"}


def test_toml_failed_atomic_replace_preserves_original(tmp_path, monkeypatch):
    import dashpva.utils.config.source as source_module

    path = tmp_path / "profile.toml"
    original = toml.dumps({"owner": "initial"}).encode()
    path.write_bytes(original)
    source = ConfigSource(str(path))
    _, revision = source.load_snapshot()

    def fail_replace(*_args):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(source_module.os, "replace", fail_replace)
    result = source.replace_if_revision({"owner": "replacement"}, revision)

    assert result.status is ConfigSaveStatus.ERROR
    assert "injected replacement failure" in result.error
    assert path.read_bytes() == original


def test_toml_snapshot_read_does_not_require_a_writable_lock_location(
    tmp_path, monkeypatch
):
    import dashpva.utils.config.source as source_module

    path = tmp_path / "profile.toml"
    path.write_text(toml.dumps({"owner": "read-only"}))

    def fail_if_locked(*_args, **_kwargs):
        raise AssertionError("read snapshots must not create a sidecar lock")

    monkeypatch.setattr(source_module, "_config_lock", fail_if_locked)

    config, revision = ConfigSource(str(path)).load_snapshot()

    assert config == {"owner": "read-only"}
    assert revision


def test_database_stale_revision_cannot_overwrite_newer_save(tmp_db):
    profile = tmp_db.create_profile("cas_profile")
    assert tmp_db.import_toml_to_profile(profile.id, {"owner": "initial"})
    source = DbProfileConfigSource(tmp_db, profile.id)
    _, first_revision = source.load_snapshot()
    _, stale_revision = source.load_snapshot()

    first = source.replace_if_revision({"owner": "first"}, first_revision)
    stale = source.replace_if_revision({"owner": "stale"}, stale_revision)

    assert first.status is ConfigSaveStatus.SAVED
    assert stale.status is ConfigSaveStatus.CONFLICT
    assert tmp_db.export_profile_to_toml(profile.id) == {"owner": "first"}


def test_database_concurrent_writers_have_exactly_one_winner(tmp_db):
    profile = tmp_db.create_profile("concurrent_cas_profile")
    assert tmp_db.import_toml_to_profile(profile.id, {"owner": "initial"})
    source = DbProfileConfigSource(tmp_db, profile.id)
    _, revision = source.load_snapshot()
    barrier = threading.Barrier(3)
    results = []

    def write(owner):
        barrier.wait()
        results.append(
            source.replace_if_revision({"owner": owner}, revision).status
        )

    threads = [
        threading.Thread(target=write, args=("left",)),
        threading.Thread(target=write, args=("right",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert results.count(ConfigSaveStatus.SAVED) == 1
    assert results.count(ConfigSaveStatus.CONFLICT) == 1
    assert tmp_db.export_profile_to_toml(profile.id)["owner"] in {"left", "right"}
