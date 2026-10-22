# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0
"""
stop_generation.py – interactive example of asynchronous generation stopping.

Usage
-----
    python examples/stop_generation.py

The script starts a long generation in a background thread and waits for the
user to press <Enter>.  As soon as <Enter> is pressed model.stop() is called,
the background thread receives the signal, breaks out of the generation loop,
and the partial output is decoded and printed.

Model / tokenizer paths
-----------------------
Set the environment variables before running:

    export ARA_MODEL_PATH=/path/to/llama2-7b
    export ARA_TOKENIZER_NAME=meta-llama/Llama-2-7b-chat-hf

Or edit MODEL_PATH / TOKENIZER_PATH below directly.
"""

import os
import sys
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import optimum.ara  # noqa: F401 
from optimum.ara import AraGenerationConfig

# ---------------------------------------------------------------------------
# Configuration: override via environment variables or edit directly
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.realpath("models/llama2-7b")
TOKENIZER_PATH = "meta-llama/Llama-2-7b-chat-hf"

INPUT_PROMPT = (
    "Explain in detail the history and evolution of artificial intelligence, "
    "covering all major milestones from the 1950s to the present day."
)

MAX_GENERATION_LENGTH = 2048  # intentionally long so stop() determines the length

print(f"Loading model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

generation_config = AraGenerationConfig.from_pretrained(MODEL_PATH)
generation_config.max_length = MAX_GENERATION_LENGTH

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": INPUT_PROMPT},
]
prompt_text = tokenizer.apply_chat_template(
    conversation=messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt_text, return_tensors="pt")

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_output = {}


def run_generation():
    print("\n[INFO] Generation started. Press <Enter> at any time to stop.\n")
    print("-" * 60)
    output = model.generate(
        **inputs,
        generation_config=generation_config,
        streamer=streamer,
    )
    generation_output["sequences"] = output
    print("\n" + "-" * 60)
    print(f"[INFO] Generation finished. Status: {model.generation_status}")


gen_thread = threading.Thread(target=run_generation, daemon=True)
gen_thread.start()

# ---------------------------------------------------------------------------
# Wait for user to press Enter, then call stop()
# ---------------------------------------------------------------------------
input("\nPress <Enter> to stop generation at any time...\n")

if gen_thread.is_alive():
    print("[INFO] Stopping generation…")
    model.stop()
    gen_thread.join(timeout=10)
    print(f"[INFO] Generation stopped. Status: {model.generation_status}")
else:
    print("[INFO] Generation had already finished naturally.")

# ---------------------------------------------------------------------------
# Decode and print the partial / full output
# ---------------------------------------------------------------------------
sequences = generation_output.get("sequences")
if sequences is not None:
    print("\n" + "=" * 60)
    print("GENERATED TEXT:")
    print("=" * 60)
    # sequences includes the prompt tokens; skip them for display
    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = sequences[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(generated_text)
    print("=" * 60)
    print(f"Generated {len(generated_ids)} token(s). Status: {model.generation_status}")
else:
    print("[WARN] No output was captured.")

del model
