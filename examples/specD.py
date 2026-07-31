# Copyright (c) 2026, Kinara, Inc. All rights reserved.
# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    TextStreamer,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
    LogitNormalization,
)
from optimum.ara import AraModelForCausalLM, AraGenerationConfig
from optimum.ara.generation.logits_process import (
    SpeculativeDecodingLogitsProcessor,
    SampleLogitsProcessor,
)

MODEL_PATH = os.path.realpath("models/llama3.1-8b")
TOKENIZER_PATH = os.path.realpath("models/llama3.1-8b/assets/llama3_tokenizer")
INPUT_PROMPT = "Hello, how are you?"

# CLI argument for reading from binary file
model = AraModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": INPUT_PROMPT},
]

inputs_text = tokenizer.apply_chat_template(
    conversation=messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(inputs_text, return_tensors="pt")

# Construct the processor list
logits_processor = LogitsProcessorList()
# Adding HF built-in processors as requested
logits_processor.append(NoRepeatNGramLogitsProcessor(ngram_size=3))
logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))
logits_processor.append(TopKLogitsWarper(top_k=50))
logits_processor.append(TopPLogitsWarper(top_p=0.9))
# logits_processor.append(LogitNormalization())
# logits_processor.append(SpeculativeDecodingLogitsProcessor())
# logits_processor.append(SampleLogitsProcessor())

streamer = TextStreamer(tokenizer, stream=sys.stdout)
output = model.generate(
    **inputs,
    streamer=streamer,
    logits_processor=logits_processor,
    target_prompt_post_mcp=False,
    target_token_post_mcp=False,
    target_prompt_pre_mcp=False,
    target_token_pre_mcp=False,
)
generated_text = tokenizer.decode(output[0])

print("\n" + "=" * 50)
print("GENERATED TEXT:")
print("=" * 50)
print(generated_text)
print("=" * 50)
