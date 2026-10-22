# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig
from transformers import AutoModelForImageTextToText
from pathlib import Path
from transformers import AutoProcessor, TextStreamer

from optimum.ara import (
    AraQwen3MultiModalConfig,
    AraQwen3MultiModalForConditionalGeneration,
    Qwen3VLProcessor,
    AraGenerationConfig
)

# Configuration
VIDEO_PATH = "/users/kapil/video_qwen/fire_bbq_cropped.mp4"
#IMAGE_PATH = "/users/kapil/images_qwen/NXP_Logo_RGB_Colour.png"
IMAGE_PATH = "/users/jayant/demo.jpeg"

PROMPT = "Describe this video briefly"
PROCESSOR_ID = "Qwen/Qwen3-VL-2B-Instruct"  # "Qwen/Qwen2.5-VL-3B-Instruct" for 3b model Kapil
MODEL_PATH = Path("./models/qwen3-vl-multimodal")  # ./models/qwen2.5-vl-multimodal-7b for b model

config = AutoConfig.from_pretrained(MODEL_PATH / "config.json")
generation_config = AraGenerationConfig.from_pretrained(
    MODEL_PATH / "generation_config.json"
)

# Initialize components
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    config=config,
)

processor = AutoProcessor.from_pretrained(PROCESSOR_ID)

streamer = TextStreamer(
    processor.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

def video_inference(video_path):
    print(f"Doing video inference over video = {video_path}, prompt = Describe this video.")

    resized_video_path = Qwen3VLProcessor.resize_video(video_path)
    messages = [
        {"role": "system", "content": [ {"type": "text", "text": "You are a helpful assistant"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": resized_video_path,  # some sample video
                    "fps": 2,
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]

    processed_inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")

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
    

def image_inference(image_path):
    print(f"Doing image inference over image = {image_path}, prompt = Describe this image.")
    resized_image_path = Qwen3VLProcessor.resize_image(image_path)
    messages = [
        {"role": "system", "content": [ {"type": "text", "text": "You are a helpful assistant"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resized_image_path,  # some sample video
                },
                {"type": "text", "text": "Describe the image"},
            ],
        }
    ]

    processed_inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")

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


def text_inference(prompt, sys_prompt = None):
    print(f"Doing text inference  with prompt = {prompt} and sys_prompt = {sys_prompt}")
    messages = []
    if sys_prompt:
        messages.append(
            {"role": "system", "content": [ {"type": "text", "text": sys_prompt}]},
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    )

    processed_inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")

    # Generate
    model.generate(
        **processed_inputs,
        generation_config=generation_config,
        stream=True,
        streamer=streamer,
    )


# video_inference(VIDEO_PATH)
# model.display_perf_statistics()

image_inference(IMAGE_PATH)
model.display_perf_statistics()

text_inference("Tell me a Joke.", sys_prompt="You are a helpful assistant")
model.display_perf_statistics()

del model

