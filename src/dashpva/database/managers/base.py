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

from typing import Any


class BaseManager:
    """Shared helpers for database managers."""

    def clean(self, value):
        """Strip stray '*' UI edit-tracking markers before persisting."""
        return self._strip_trailing_asterisks(value)

    def _strip_trailing_asterisks(self, obj: Any) -> Any:
        """Recursively strip trailing '*' UI edit-tracking markers from strings, dict keys, and list items.

        If both a clean key and its '*'-suffixed duplicate exist in the same dict,
        the clean key's value always wins, regardless of dict iteration order.
        """
        if isinstance(obj, dict):
            cleaned = [
                (key, key.rstrip('*') if isinstance(key, str) else key, self._strip_trailing_asterisks(value))
                for key, value in obj.items()
            ]
            result = {clean_key: value for key, clean_key, value in cleaned if key == clean_key}
            for key, clean_key, value in cleaned:
                if key != clean_key:
                    result.setdefault(clean_key, value)
            return result
        if isinstance(obj, list):
            return [self._strip_trailing_asterisks(item) for item in obj]
        if isinstance(obj, str):
            return obj.rstrip('*')
        return obj
