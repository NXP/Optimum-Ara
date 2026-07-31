# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForVision2Seq

from .configuration_qwen3_vl import (
    AraQwen3ImageConfig,
    AraQwen3VLConfig,
    AraQwen3MultiModalConfig,
)
from .modeling_qwen3_vl import (
    AraQwen3ImageForConditionalGeneration,
    AraQwen3VLForConditionalGeneration,
    AraQwen3MultiModalForConditionalGeneration,
)
from .processing_qwen3_vl import Qwen3VLProcessor

# Register with transformers library for seamless HF integration
AutoConfig.register("ara_qwen3_vl", AraQwen3VLConfig)
AutoModelForImageTextToText.register(
    AraQwen3VLConfig, AraQwen3VLForConditionalGeneration
)
AutoModelForVision2Seq.register(AraQwen3VLConfig, AraQwen3VLForConditionalGeneration)

AutoConfig.register("ara_qwen3_image", AraQwen3ImageConfig)
AutoModelForImageTextToText.register(
    AraQwen3ImageConfig, AraQwen3ImageForConditionalGeneration
)

AutoConfig.register("ara_qwen3_multimodal", AraQwen3MultiModalConfig)
AutoModelForImageTextToText.register(
    AraQwen3MultiModalConfig, AraQwen3MultiModalForConditionalGeneration
)
AutoModelForVision2Seq.register(
    AraQwen3MultiModalConfig, AraQwen3MultiModalForConditionalGeneration
)

__all__ = [
    "AraQwen3VLConfig",
    "AraQwen3ImageConfig",
    "AraQwen3ImageForConditionalGeneration",
    "AraQwen3VLForConditionalGeneration",
    "AraQwen3MultiModalForConditionalGeneration",
    "AraQwen3MultiModalConfig",
    "Qwen3VLProcessor",
]
