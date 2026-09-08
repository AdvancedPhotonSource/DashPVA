#!/usr/bin/env python3
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

"""
One-time cleanup: strip stray '*' UI edit-tracking markers left in the
database by a since-fixed bug, and write the cleaned values back.

Safe to run multiple times on an existing database — already-clean rows
are left untouched.
"""
import json
import sqlite3


def clean_database() -> None:
    import dashpva.settings as _settings
    from dashpva.database.managers.base import BaseManager

    db_file = _settings.PROJECT_ROOT / "dashpva.db"
    if not db_file.exists():
        return
    cleaner = BaseManager()
    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.cursor()

        for row_id, key in cur.execute(
            "SELECT id, config_key FROM profile_configs WHERE config_type != '__toml__'"
        ).fetchall():
            cleaned = cleaner.clean(key)
            if cleaned != key:
                cur.execute("UPDATE profile_configs SET config_key=? WHERE id=?", (cleaned, row_id))

        for row_id, raw in cur.execute(
            "SELECT id, config_value FROM profile_configs WHERE config_type='__toml__' AND config_key='__data__'"
        ).fetchall():
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue
            cleaned = cleaner.clean(data)
            if cleaned != data:
                cur.execute("UPDATE profile_configs SET config_value=? WHERE id=?", (json.dumps(cleaned), row_id))

        for row_id, key in cur.execute("SELECT id, key FROM setting_values").fetchall():
            cleaned = cleaner.clean(key)
            if cleaned != key:
                cur.execute("UPDATE setting_values SET key=? WHERE id=?", (cleaned, row_id))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    clean_database()
