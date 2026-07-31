# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from ...configuration_utils import AraConfig, AraPretrainedConfig


class AraLlava(AraConfig):
    """AraConfig for LLaVA models with LLaVA-specific fields."""

    vision_model_path: str | None = None
    vision_input_path: str | None = None
    vision_input_scale_path: str | None = None
    vision_add_input_path: str | None = None
    vision_add_input_scale_path: str | None = None
    vision_concat_input_path: str | None = None
    vision_concat_input_scale_path: str | None = None

    assets_dir_path: str | None = None


class AraLlavaConfig(AraPretrainedConfig):
    """
    Configuration class for LLaVA models.

    Inherits from AraPretrainedConfig and adds LLaVA-specific parameters.
    All configuration values should be specified in the model's config.json file.
    """

    model_type = "ara_llava"
    ara_class = AraLlava
