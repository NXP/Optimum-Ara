# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from ...configuration_utils import AraPretrainedConfig


class AraWhisperConfig(AraPretrainedConfig):
    model_type = "ara_whisper"
    config_class = AraPretrainedConfig
    pass
