# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from pickle import NONE
from typing import Any, List, cast

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLProcessor, TextStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput

from optimum.ara import AraGenerationConfig, AraQwen2_5_ImageForConditionalGeneration

# Configuration
IMAGE_PATH = "examples/assets/test.jpg"
PROMPT = "Describe this image briefly"
MODEL_PATH = "models/qwen2.5-image-3B"


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
streamer = TextStreamer(processor.tokenizer)  # type: ignore

model = AraQwen2_5_ImageForConditionalGeneration.from_pretrained(MODEL_PATH)
generation_config = AraGenerationConfig.from_pretrained(
    MODEL_PATH + "/generation_config.json"
)

image = Image.open(IMAGE_PATH)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
                "resized_height": 336,
                "resized_width": 336,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = cast(Qwen2_5_VLProcessor, processor).apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
if image_inputs is None:
    image_inputs = []
if video_inputs is None:
    video_inputs = []

# Cast to satisfy processor type (process_vision_info return type differs from processor signature)
processed_inputs = cast(Qwen2_5_VLProcessor, processor)(
    text=[text],
    images=cast(List[Any], image_inputs if image_inputs else []),
    padding=True,
    return_tensors="pt",
)

# Generate
result = model.generate(
    **processed_inputs,
    generation_config=generation_config,
    max_new_tokens=512,
    temperature=1.0,
    do_sample=False,
    streamer=streamer,
)

if isinstance(result, GenerateDecoderOnlyOutput):
    result = result.sequences

# Decode the generated tokens to text
generated_text = cast(Qwen2_5_VLProcessor, processor).decode(
    result.flatten(), skip_special_tokens=True
)

print("\n" + "=" * 50)
print("GENERATED TEXT:")
print("=" * 50)
print(generated_text)
print("=" * 50)

del model
