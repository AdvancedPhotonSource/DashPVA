#!/usr/bin/env python3
# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
