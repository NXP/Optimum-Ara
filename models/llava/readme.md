# LLaVA Model Configuration

This directory contains the configuration for LLaVA (Large Language and Vision Assistant) model integration with Ara hardware.

## Model Components

- **LLM Model**: Llama 3.1 8B model (.dvm file)
- **Vision Model**: CLIP ViT-L/14 model (.dvm file) 
- **Multimodal Projector**: Connects vision features to LLM embedding space

## Required Files

### Model Files
- `llama3_1_model.dvm` - Llama 3.1 8B LLM model
- `vit_model.dvm` - CLIP ViT-L/14 vision model

### Assets Directory
- `embedding_table.pt` - Token embedding table (128320 x 4096)
- `proxy_ip_data1.bin` - Quantization constants
- `proxy_ip_data2.bin` - Quantization constants  
- `proxy_ip_data3.bin` - Quantization constants
- `proxy_ip_data4.bin` - Quantization constants
- `proxy_ip_data5.bin` - Quantization constants

## Configuration

The `config.json` file contains:
- `ara.dvm_path`: Path to the model directory (relative path)
- `vision_model_path`: Path to the vision model file (relative to model directory)

## Model Directory Structure

The LLaVA model should be organized as follows:

```
models/llava/
├── config.json                    # Model configuration
├── generation_config.json         # Generation configuration
├── llama3_1_model.dvm            # LLM model file (DVM format)
├── vit_model.dvm                 # Vision model file (DVM format)
└── assets/
    ├── embedding_table.pt         # Token embeddings
    ├── proxy_ip_data0.bin         # Quantization constants
    ├── proxy_ip_data1.bin         # Quantization constants
    ├── proxy_ip_data2.bin         # Quantization constants
    ├── proxy_ip_data3.bin         # Quantization constants
    ├── proxy_ip_data4.bin         # Quantization constants
    ├── proxy_ip_data5.bin         # Quantization constants
    └── llava_tokenizer/           # Tokenizer directory
```

## Usage

```python
from optimum.ara import AraLlavaForConditionalGeneration
from transformers import AutoTokenizer

model = AraLlavaForConditionalGeneration.from_pretrained("models/llava")
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# Process image and text
image = Image.open("image.jpg")
inputs = tokenizer("Describe this image", return_tensors="pt")
outputs = model.generate(image=image, **inputs)
``` 