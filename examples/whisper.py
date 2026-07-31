# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoProcessor
from typing import Any, cast
import soundfile as sf
import librosa
from optimum.ara import AraGenerationConfig
from transformers import (
    TextStreamer,
    AutoModelForSpeechSeq2Seq,
    AutoConfig,
)

# Load the tokenizer for the medium model
processor = AutoProcessor.from_pretrained("openai/whisper-medium")
generation_config = AraGenerationConfig.from_pretrained("models/whisper")

config = AutoConfig.from_pretrained("models/whisper")
forced_decoder_ids = [
    tid
    # pyrefly: ignore [missing-attribute]
    for _, tid in processor.get_decoder_prompt_ids(
        language="english", task="transcribe", no_timestamps=True
    )
]
forced_decoder_ids = [generation_config.decoder_start_token_id] + forced_decoder_ids
model = AutoModelForSpeechSeq2Seq.from_config(config)

audio_input, sr = sf.read("examples/assets/audio.mp3")
prompt = "transcribe the following audio "

# pyrefly: ignore [missing-attribute]
prompt_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids

if sr != 16000:
    audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
    sr = 16000

# pyrefly: ignore [not-callable, unexpected-keyword]
inputs = processor(audio_input, sampling_rate=sr, return_tensors="pt")
input_features = inputs.input_features
streamer = TextStreamer(getattr(processor, "tokenizer"))

generated_ids = model.generate(
    input_features=input_features,
    streamer=streamer,
    prompt_ids=prompt_ids,
    forced_decoder_ids=forced_decoder_ids,
    generation_config=generation_config,
)

del model
