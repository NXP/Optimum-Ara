# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForVision2Seq

from .configuration_llava import AraLlava, AraLlavaConfig
from .modeling_llava import AraLlavaForConditionalGeneration

AutoConfig.register("ara_llava", AraLlavaConfig)
AutoModelForImageTextToText.register(AraLlavaConfig, AraLlavaForConditionalGeneration)
AutoModelForVision2Seq.register(AraLlavaConfig, AraLlavaForConditionalGeneration)
__all__ = [
    "AraLlavaConfig",
    "AraLlavaForConditionalGeneration",
]
