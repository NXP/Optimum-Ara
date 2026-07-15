# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, IntEnum
import os


class InterfaceType(str, Enum):
    """
    Types of sessions currently supported.
    Proxy must also be running using same connection type to actually be able to connect.


    Default interface type for Linux is SOCKET.
    Default interface type for Windows is NAMED_PIPE.
    IPV4 can be used for both Linux and Windows.
    """

    IPV4 = "IPV4"
    NAMED_PIPE = "NAMED_PIPE"
    SOCKET = "SOCKET"


class CoreConfig(IntEnum):
    TIMEOUT_MS = 150000  # 150 seconds


# sets interface_type as NAMED_PIPE if user is on windows else SOCKET is used
DEFAULT_INTERFACE_TYPE = (
    InterfaceType.NAMED_PIPE
    if os.name == "nt"
    else InterfaceType.SOCKET
    if os.name == "posix"
    else InterfaceType.IPV4  # fallback for other platforms
)
