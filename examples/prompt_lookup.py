# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TextStreamer,
    LogitsProcessor,
)
from transformers.generation.candidate_generator import PromptLookupCandidateGenerator
from optimum import ara
from optimum.ara import AraGenerationConfig, AraModelForCausalLM, CustomLogitsProcessor
import numpy as np
import torch
from typing import cast
import pdb


model = AraModelForCausalLM.from_pretrained("models/qwen2.5-instruct-7B/")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_config = AraGenerationConfig.from_pretrained("models/qwen2.5-instruct-7B/")
print(generation_config)

prompt = 'Below are python functions\n\ndef expand_center(s, left, right):\n    while left >= 0 and right < len(s) and s[left] == s[right]:\n        left -= 1\n        right += 1\n    return s[left+1:right]\n\ndef find_longest(s):\n    longest = ""\n    for i in range(len(s)):\n        for a, b in [(i, i), (i, i+1)]:  # odd & even centers\n            temp = expand_center(s, a, b)\n            if len(temp) > len(longest):\n                longest = temp\n    return longest\n\nfrom the above python functions what is the code for find_longest function, need only code, no description.\n'

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

inputs = tokenizer.apply_chat_template(
    conversation=messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(inputs)
generation_config = AraGenerationConfig.from_pretrained(
    "./local_models/qwen2.5-instruct-7B/"
)
print(generation_config)


class PromptLookupCustomProcessor(CustomLogitsProcessor):
    def __init__(self):
        self.candidate_generator = PromptLookupCandidateGenerator(
            eos_token_id=torch.tensor(151645),
            num_output_tokens=3,
            max_matching_ngram_size=10,
            max_length=generation_config.max_length,
            # logits_processor=logits_processor,
            # vocab_size=generation_config.vocab_size,
        )
        self.first_call = True
        self.candidate_new_tokens_prev = torch.tensor([0])
        self.count = 0
        self.acceptance_length = 0

    def __call__(
        self, input_ids: torch.LongTensor, new_logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.first_call:
            cur_len = input_ids.shape[1]
            self.first_call = False
            candidate_input_ids, _ = self.candidate_generator.get_candidates(input_ids)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]

            for i in range(0, candidate_new_tokens.size(1)):
                if candidate_new_tokens[0][i].numel() == 151645:
                    candidate_new_tokens = candidate_input_ids[:, :i]
                    break

            self.candidate_new_tokens_prev = candidate_new_tokens

            new_tokens = np.argmax(new_logits, axis=-1)
            return new_tokens[:1], candidate_new_tokens.numpy().flatten()

        new_tokens = np.argmax(new_logits, axis=-1)

        n_matches = 0
        for i in range(0, min(self.candidate_new_tokens_prev.numel(), new_tokens.size)):
            if self.candidate_new_tokens_prev[0][i].item() == new_tokens[i]:
                n_matches += 1
            else:
                break
        self.count += 1
        self.acceptance_length += n_matches

        if n_matches == self.candidate_new_tokens_prev.size():
            n_matches -= 1

        selected_tokens = new_tokens[: n_matches + 1]

        selected_tokens_tensor = torch.from_numpy(selected_tokens).unsqueeze(
            0
        )  # Shape: [1, num_tokens]
        cat_ids = torch.cat([input_ids, selected_tokens_tensor], dim=1)
        input_ids = cast(torch.LongTensor, cat_ids.long())

        candidate_input_ids, _ = self.candidate_generator.get_candidates(input_ids)

        cur_len = input_ids.shape[1]
        candidate_new_tokens = candidate_input_ids[:, cur_len:]

        self.candidate_new_tokens_prev = candidate_new_tokens

        return (selected_tokens, candidate_new_tokens.numpy().flatten())


processor = PromptLookupCustomProcessor()

output = model.generate(
    **inputs,
    streamer=streamer,
    generation_config=generation_config,
    custom_logits_processor=processor,
)
print("\n")
# print(tokenizer.decode(output))
if processor.count != 0:
    print("avg accepetance length : ", processor.acceptance_length / processor.count)
else:
    print("avg accepetance length : 1")

model.display_perf_statistics()

del model
