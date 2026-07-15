# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_GENERATION_CONFIG_NAME = "generation_config.json"
DEFAULT_SOCKET_PATH = "/var/run/dvproxy.sock"
DEFAULT_INFERENCE_DEVICE = "ARA2"
DEFAULT_IP_ADDRESS = "127.0.0.1"
DEFAULT_NAMED_PIPE = "//./pipe/proxy_pipe"
DEFAULT_PORT = 5000
DVM_DDR_BASE_ADDR = 0x90000000
ADDITION_SPACE_REQUIRED_TO_LOAD_MODEL = (
    600 * 1024 * 1024
)  # 600 MB addition memory requird (rough estimate)
