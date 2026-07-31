# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Any, List, cast

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import TextStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput

from optimum.ara import AraQwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# Configuration
VIDEO_PATH = "./examples/assets/personFalling_8.mp4"
PROMPT = "Describe this video briefly"
MODEL_PATH = "./models/qwen2.5-vl-3B_variable_length"

# Initialize components
model = AraQwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                "fps": 2,
                "resized_height": 336,
                "resized_width": 336,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
if image_inputs is None:
    image_inputs = []
if video_inputs is None:
    video_inputs = []

processed_inputs = processor(
    text=[text],
    videos=cast(List[Any], video_inputs if video_inputs else []),
    padding=True,  # pyrefly: ignore [unexpected-keyword]
    return_tensors="pt",  # pyrefly: ignore [unexpected-keyword]
)

streamer = TextStreamer(processor.tokenizer, stream=sys.stdout)  # type: ignore
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
generated_text = processor.decode(result.flatten(), skip_special_tokens=True)  # type: ignore

print("\n" + "=" * 50)
print("GENERATED TEXT:")
print("=" * 50)
print(generated_text)
print("=" * 50)
model.display_perf_statistics()

del model
