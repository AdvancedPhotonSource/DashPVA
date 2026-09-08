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

"""Configuration loading, resolution, and persistence for DashPVA."""

from .hkl import (
    HklAxisSection,
    HklChannel,
    axis_field_channels,
    axis_group_parts,
    get_hkl_section,
    iter_hkl_axes,
    iter_semantic_hkl_channels,
    numbered_axis_group_names,
    required_rsm_channels,
    section_field_channels,
    semantic_hkl_channels,
)
from .resolver import resolve_profile_config
from .source import ConfigSaveResult, ConfigSaveStatus, ConfigSource, ConfigSourceError

__all__ = [
    "ConfigSaveResult",
    "ConfigSaveStatus",
    "ConfigSource",
    "ConfigSourceError",
    "HklAxisSection",
    "HklChannel",
    "axis_group_parts",
    "axis_field_channels",
    "get_hkl_section",
    "iter_hkl_axes",
    "iter_semantic_hkl_channels",
    "numbered_axis_group_names",
    "required_rsm_channels",
    "resolve_profile_config",
    "section_field_channels",
    "semantic_hkl_channels",
]
