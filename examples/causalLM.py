# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import optimum.ara
from transformers import TextStreamer

MODEL_PATH = os.path.realpath("models/llama2-7b")
# TOKENIZER_PATH = os.path.realpath("models/llama2-7b/assets/llama2_tokenizer")

# uncomment if you have access to tokenizer from hf hub
TOKENIZER_PATH = "meta-llama/Llama-2-7b-chat-hf"

INPUT_PROMPT = "Hello, how are you?"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": INPUT_PROMPT},
]

inputs = tokenizer.apply_chat_template(
    conversation=messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(inputs)
streamer = TextStreamer(tokenizer, stream=sys.stdout)
output = model.generate(**inputs, streamer=streamer)
generated_text = tokenizer.decode(output[0])

print("\n" + "=" * 50)
print("GENERATED TEXT:")
print("=" * 50)
print(generated_text)
print("=" * 50)

del model
