# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, cast, List
from PIL import Image
from transformers import LlavaProcessor
from optimum.ara import AraLlavaForConditionalGeneration
from transformers import TextStreamer


MODEL_PATH = os.path.realpath("models/llava")
INPUT_PROMPT = "Describe the image."
IMAGE_PATH = "examples/assets/test.jpg"

# Load the image
sample_image = Image.open(IMAGE_PATH).convert("RGB")

# Load model
print("Loading LLaVA model...")
model = AraLlavaForConditionalGeneration.from_pretrained(MODEL_PATH)

# Load processor
print("Loading processor...")
processor = LlavaProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")

# Prepare conversation
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": INPUT_PROMPT},
            {"type": "image", "image": sample_image},
        ],
    }
]

# Process inputs
print("Processing inputs...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# Cast to satisfy processor signature (PIL.Image type mismatch between stubs)
# pyrefly: ignore [unexpected-keyword]
processed_inputs = processor(
    text=text, images=cast(Any, [sample_image]), padding=True, return_tensors="pt"
)

# Generate
print("\n" + "=" * 50)
print("GENERATED TEXT:")
print("-" * 50)
streamer = TextStreamer(cast(Any, processor).tokenizer, skip_prompt=True)
result = model.generate(**processed_inputs, streamer=streamer, max_new_tokens=512)
generated_text = processor.decode(result[0], skip_special_tokens=True)
# print(generated_text)
print("=" * 50)

del model
