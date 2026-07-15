# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_llama import AraLlamaConfig
from .modeling_llama import AraLlamaForCausalLM

AutoConfig.register("ara_llama", AraLlamaConfig)
AutoModelForCausalLM.register(AraLlamaConfig, AraLlamaForCausalLM)
