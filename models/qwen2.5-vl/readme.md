# Qwen2.5 VLM Model for Ara Hardware

This directory contains the configuration files for the Qwen2.5 VLM model optimized for Ara hardware.

## Model Information

- **Model**: Qwen2.5-VL-3B
- **Architecture**: Vision-Language Model
- **Vision Encoder**: ViT (Vision Transformer)
- **Language Model**: Qwen 2.5 3B
- **Hardware**: Ara DVM
- **Supported input length**: 1-26 seconds

## Files

- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `readme.md` - This file

## Usage

```python
from optimum.ara import AraQwen2_5_VLForConditionalGeneration, QwenVLProcessor

# loading model and processor
model = AraQwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = QwenVLProcessor()

VIDEO_PATH = "/path/to/video.mp4"
PROMPT = "Describe this video briefly"
processed_inputs = processor(VIDEO_PATH, PROMPT)

# Generate response
result = model.generate(
    **processed_inputs,
    max_new_tokens=512,
    temperature=1.0,
    do_sample=False,
)

response = processor.processor.decode(result, skip_special_tokens=True)
print(response)
```

## Configuration

The model uses the same two-model architecture as LLaVA:
1. **Vision Model**: ViT DVM for image feature extraction
2. **Language Model**: Qwen 2.5 DVM for text generation

Both models are loaded from the paths specified in the configuration.

## Hardware Requirements

- Ara hardware with DVM support
- Vision and language model DVM files
- Required bin files for vision encoding constants 

## Input Requirements

- Video inputs allowed be upto 26 seconds long.

## Important Note
- Old QwenVL2.5-7B model of fixed video length will not work with the latest version.
- New version supports dynamic video length.
