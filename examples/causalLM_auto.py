# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TextStreamer,
    GenerationConfig,
)
from optimum import ara
from optimum.ara import AraGenerationConfig, AraModelForCausalLM
import numpy as np

config = AutoConfig.from_pretrained("KinaraInc/llama2-7b")
config.ara.dvm_path = "./models/llama2-7b/assets/model.dvm"
config.ara.interface_type = "SOCKET"

model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("./models/llama2-7b/assets/llama2_tokenizer/")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a joke"},
]

# generation_config = AraGenerationConfig.from_pretrained('./models/llama2-7b/')
# print(generation_config)

inputs = tokenizer.apply_chat_template(
    conversation=messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(inputs)
# output = model.generate(**inputs, streamer=streamer, generation_config=generation_config)
output = model.generate(**inputs, streamer=streamer)

print("\n")
model.display_perf_statistics()

del model
