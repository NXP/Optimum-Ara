# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from ...modeling_decoder import AraModelForCausalLM
from .configuration_qwen import AraQwenConfig


class AraQwenForCausalLM(AraModelForCausalLM):
    config_class = AraQwenConfig
    pass
