# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, cast
from threading import Thread

import gradio as gr
from PIL import Image
from optimum.ara import AraLlavaForConditionalGeneration
from transformers import LlavaProcessor, TextIteratorStreamer

# ---- Configuration ----
MODEL_PATH = os.path.realpath("models/llava")

# ---- Load model & processor once (startup) ----
print("Loading LLaVA model...")
model = AraLlavaForConditionalGeneration.from_pretrained(MODEL_PATH)
print("Loading processor...")
# Using the same processor as in the initial llava_demo.py
processor = LlavaProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")

tokenizer = cast(Any, processor).tokenizer


def describe_image(image: Image.Image, prompt: str):
    """
    Generator function that yields tokens as they are generated.
    Gradio automatically treats any function with `yield` as streaming output.
    """
    if image is None:
        yield "Please upload an image."
        return

    # Prepare conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Cast image to Any to satisfy processor signature stubs
    # pyrefly: ignore [unexpected-keyword]
    processed_inputs = processor(
        text=text, images=cast(Any, [image]), padding=True, return_tensors="pt"
    )

    # Create TextIteratorStreamer to capture tokens as they're generated
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Prepare generation arguments
    generation_kwargs = {
        **processed_inputs,
        "max_new_tokens": 512,
        "streamer": streamer,
    }

    # Run generation in a separate thread so we can stream tokens
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield tokens as they arrive from the streamer
    generated_text = ""
    for new_token in streamer:
        generated_text += new_token
        yield generated_text

    # Ensure thread completes
    thread.join()


with gr.Blocks() as demo:
    gr.Markdown("## LLaVA Image Description Demo (Streaming)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image", type="pil")
            prompt_input = gr.Textbox(
                label="Prompt",
                value="Describe this image briefly",
            )
            run_btn = gr.Button("Generate description")
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated description",
                lines=8,
            )

    run_btn.click(
        fn=describe_image,
        inputs=[image_input, prompt_input],
        outputs=[output_text],
    )

if __name__ == "__main__":
    demo.launch()
