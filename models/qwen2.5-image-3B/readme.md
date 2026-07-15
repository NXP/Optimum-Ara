# Qwen2.5 VLM Model for Ara Hardware

This directory contains the configuration files for the Qwen2.5 VLM model for Image only optimized for Ara hardware.

## Model Information

- **Model**: Qwen2.5-VL-3B(Image)
- **Architecture**: Vision-Language Model
- **Vision Encoder**: ViT (Vision Transformer)
- **Language Model**: Qwen 2.5 3B
- **Hardware**: Ara DVM
- **Supported input**: Image

## Files

- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `readme.md` - This file

## Usage

```python
from PIL import Image
from optimum.ara import AraQwen2_5_ImageForConditionalGeneration

# Load model and processor
model = AraQwen2_5_ImageForConditionalGeneration.from_pretrained("./models/qwen2.5-image-3B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


image = Image.open("path/to/image.jpg")
prompt = "Describe this image in detail."
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
image_inputs, video_inputs = process_vision_info(messages)

# preparing inputs
processed_inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Generate response
outputs = model.generate(
    **processed_inputs,
    max_new_tokens=512,
    temperature=1.0,
    do_sample=False,
    stream=False
)

response = processor.decode(result, skip_special_tokens=True)
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