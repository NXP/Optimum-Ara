# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, TextStreamer

from optimum.ara import (
    AraQwen2_5MultiModalConfig,
    AraQwen2_5_MultiModalForConditionalGeneration,
)
from optimum.ara.generation.configuration_utils import AraGenerationConfig

# Configuration
VIDEO_PATH = "/users/kapil/video_qwen/fire_bbq_cropped.mp4"
IMAGE_PATH = "/users/kapil/images_qwen/NXP_Logo_RGB_Colour.png"
PROMPT = "Describe this video briefly"
PROCESSOR_ID = (
    "Qwen/Qwen2.5-VL-3B-Instruct"  # "Qwen/Qwen2.5-VL-3B-Instruct" for 3b model Kapil
)
MODEL_PATH = Path(
    "./models/qwen2.5-vl-multimodal-3b"
)  # ./models/qwen2.5-vl-multimodal-7b for b model

config = AraQwen2_5MultiModalConfig.from_pretrained(MODEL_PATH / "config.json")
generation_config = AraGenerationConfig.from_pretrained(
    MODEL_PATH / "generation_config.json"
)

# Initialize components
model = AraQwen2_5_MultiModalForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    config=config,
)


def video_inference(video_path, processor):
    print(
        f"Doing video inference over video = {video_path}, prompt = Describe this video."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "resized_height": 336,
                    "resized_width": 336,
                    "fps": 2,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)  # pyrefly: ignore

    processed_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    streamer = TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Generate
    model.generate(
        **processed_inputs,
        generation_config=generation_config,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=False,
        stream=True,
        streamer=streamer,
    )


def image_inference(image_path, processor):
    print(
        f"Doing image inference over image = {image_path}, prompt = Describe this image."
    )
    image = Image.open(image_path)

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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)  # pyrefly: ignore

    processed_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    streamer = TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Generate
    model.generate(
        **processed_inputs,
        generation_config=generation_config,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=False,
        stream=True,
        streamer=streamer,
    )


def text_inference(prompt, processor, sys_prompt=None):
    print(f"Doing text inference  with prompt = {prompt} and sys_prompt = {sys_prompt}")
    messages = []
    if sys_prompt:
        messages.append(
            {"role": "system", "content": sys_prompt},
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    )

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)  # pyrefly: ignore

    processed_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    streamer = TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Generate
    model.generate(
        **processed_inputs,
        generation_config=generation_config,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=False,
        stream=True,
        streamer=streamer,
    )


processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
video_inference(VIDEO_PATH, processor)
image_inference(IMAGE_PATH, processor)
text_inference("Who are you?", processor, sys_prompt="You are Chat Assistant by NXP")
