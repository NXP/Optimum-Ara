# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Any, cast

from transformers import TextStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput

from optimum.ara import AraQwen2_5_VLForConditionalGeneration, QwenVLProcessor
from optimum.ara.generation.configuration_utils import AraGenerationConfig

# Configuration
VIDEO_PATH = "./examples/assets/personFalling_8.mp4"
PROMPT = "Describe this video briefly"
MODEL_PATH = "./models/qwen2.5-vl-3B_variable_length"

# Initialize components
model = AraQwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = QwenVLProcessor()

# Process inputs
processed_inputs = processor(prompt=PROMPT, video_path=VIDEO_PATH)

streamer = TextStreamer(processor.processor.tokenizer, stream=sys.stdout)  # type: ignore
# Generate
result = model.generate(
    **processed_inputs,
    max_new_tokens=512,
    temperature=1.0,
    do_sample=False,
    stream=False,
    streamer=streamer,
)

if isinstance(result, GenerateDecoderOnlyOutput):
    result = result.sequences

# Decode the generated tokens to text
generated_text = processor.processor.decode(result.flatten(), skip_special_tokens=True)

print("\n" + "=" * 50)
print("GENERATED TEXT:")
print("=" * 50)
print(generated_text)
print("=" * 50)
model.display_perf_statistics()

del model
