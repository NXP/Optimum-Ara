# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

from .configuration_whisper import AraWhisperConfig
from .modeling_whisper import AraModelForWhisper

# Register with transformers library for seamless HF integration
AutoConfig.register("ara_whisper", AraWhisperConfig)
AutoModelForSpeechSeq2Seq.register(AraWhisperConfig, AraModelForWhisper)

__all__ = [
    "AraWhisperConfig",
    "AraModelForWhisper",
]
