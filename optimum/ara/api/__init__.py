# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import platform
import importlib.util
from pathlib import Path

sdk_root = "/usr/share/rt-sdk-ara240"
if not os.path.exists(sdk_root):
    sdk_root = os.environ.get("DV_TGT_ROOT")
    if not sdk_root:
        raise EnvironmentError(
            "Default SDK path does not exist and DV_TGT_ROOT environment variable not set"
        )

# Detect platform architecture
machine = platform.machine().lower()
if machine in ["x86_64", "amd64"]:
    arch = "x86"
elif machine in ["aarch64", "arm64"]:
    arch = "aarch64"
else:
    raise EnvironmentError(
        f"Unsupported architecture: {machine}. Only x86_64 and aarch64 are supported."
    )

# Path to external dvapi.py based on architecture
if Path(sdk_root).is_symlink():
    dvapi_path = Path(sdk_root) / "include" / "dvapi.py"
else:
    dvapi_path = Path(sdk_root) / "art" / "linux" / arch / "include" / "dvapi.py"

if not dvapi_path.exists():
    raise FileNotFoundError(f"dvapi.py not found at {dvapi_path}")

# Dynamically load dvapi.py as optimum.ara.api.dvapi
spec = importlib.util.spec_from_file_location("optimum.ara.api.dvapi", dvapi_path)
if spec is None:
    raise ImportError(f"Could not create spec for dvapi at {dvapi_path}")
if spec.loader is None:
    raise ImportError(f"Could not get loader for dvapi at {dvapi_path}")
dvapi_module = importlib.util.module_from_spec(spec)
sys.modules["optimum.ara.api.dvapi"] = dvapi_module
sys.modules["dvapi"] = dvapi_module
spec.loader.exec_module(dvapi_module)
