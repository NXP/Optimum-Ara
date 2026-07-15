# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar, Optional
from ...configuration_utils import AraConfig, AraPretrainedConfig

# We only support Qwen2_5_VL models in following modalities
# Each dvm is tied to a specific modality due to its inherent
# rope table.
# image: This dvm can only infer on image correctly
# video: This dvm can only infer on video of specific number of frames correctly (16 frames for now)
# multimodal: Latest addtion. This dvm can support image inference, video inference (of any duration, only limited by context_size of dvm, usually 4096) and text only inference. It also supports arbitrary system prompt.


class AraQwen2_5_VL(AraConfig):
    vision_model_path: str | None = None
    input_scale_path: str | None = None
    rope_table_path: str | None = None


class AraQwen2_5VLConfig(AraPretrainedConfig):
    model_type = "ara_qwen_vl"
    ara_class = AraQwen2_5_VL
    video_token_id: int = 151656
    image_token_id: int = 151655


class AraQwen2_5ImageConfig(AraPretrainedConfig):
    model_type = "ara_qwen_image"
    ara_class = AraQwen2_5_VL
    video_token_id: int = 151656
    image_token_id: int = 151655


class AraQwen2_5MultiModalConfig(AraPretrainedConfig):
    model_type = "ara_qwen_multimodal"
    ara_class = AraQwen2_5_VL
    video_token_id: int = 151656
    image_token_id: int = 151655
    sys_pad_token_id: int = 220
