# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_qwen import AraQwenConfig
from .modeling_qwen import AraQwenForCausalLM

AutoConfig.register("ara_qwen", AraQwenConfig)
AutoModelForCausalLM.register(AraQwenConfig, AraQwenForCausalLM)

__all__ = ["AraQwenConfig", "AraQwenForCausalLM"]
