# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForVision2Seq

from .configuration_qwen2_5_vl import (
    AraQwen2_5ImageConfig,
    AraQwen2_5VLConfig,
    AraQwen2_5MultiModalConfig,
)
from .modeling_qwen2_5_vl import (
    AraQwen2_5_ImageForConditionalGeneration,
    AraQwen2_5_VLForConditionalGeneration,
    AraQwen2_5_MultiModalForConditionalGeneration,
)
from .processing_qwen2_5_vl import QwenVLProcessor

# Register with transformers library for seamless HF integration
AutoConfig.register("ara_qwen_vl", AraQwen2_5VLConfig)
AutoModelForImageTextToText.register(
    AraQwen2_5VLConfig, AraQwen2_5_VLForConditionalGeneration
)
AutoModelForVision2Seq.register(
    AraQwen2_5VLConfig, AraQwen2_5_VLForConditionalGeneration
)

AutoConfig.register("ara_qwen_image", AraQwen2_5ImageConfig)
AutoModelForImageTextToText.register(
    AraQwen2_5ImageConfig, AraQwen2_5_ImageForConditionalGeneration
)

AutoConfig.register("ara_qwen_multimodal", AraQwen2_5MultiModalConfig)
AutoModelForImageTextToText.register(
    AraQwen2_5MultiModalConfig, AraQwen2_5_MultiModalForConditionalGeneration
)
AutoModelForVision2Seq.register(
    AraQwen2_5MultiModalConfig, AraQwen2_5_MultiModalForConditionalGeneration
)

__all__ = [
    "AraQwen2_5VLConfig",
    "AraQwen2_5ImageConfig",
    "AraQwen2_5_ImageForConditionalGeneration",
    "AraQwen2_5_VLForConditionalGeneration",
    "AraQwen2_5_MultiModalForConditionalGeneration",
    "AraQwen2_5MultiModalConfig",
    "QwenVLProcessor",
]
