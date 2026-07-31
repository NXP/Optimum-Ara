# Qwen3 VL Model Classes

Qwen3-VL is a multimodal large language model from the Qwen series that can understand text, images, and videos. It supports visual question answering, image captioning, document understanding, and arbitrary system prompts.

Qwen3-VL DVMs are available in three modalities, each tied to a specific rope table baked at compile time:

- **video** — inference on video with a fixed frame count (16 frames)
- **image** — inference on images only
- **multimodal** — inference on image, video (any duration, limited by context_size, usually 4096), and text-only; also supports arbitrary system prompts

## Classes

### Config classes

- [AraQwen3_VL](#araqwen3_vl)
- [AraQwen3VLConfig](#araqwen3vlconfig)
- [AraQwen3ImageConfig](#araqwen3imageconfig)
- [AraQwen3MultiModalConfig](#araqwen3multimodalconfig)

### Model classes

- [AraQwen3VLForConditionalGeneration](#araqwen3vlforconditionalgeneration)
- [AraQwen3ImageForConditionalGeneration](#araqwen3imageforconditionalgeneration)
- [AraQwen3MultiModalForConditionalGeneration](#araqwen3multimodalforconditionalgeneration)

### Processor class

- [Qwen3VLProcessor](#qwen3vlprocessor)

---

## AraQwen3_VL

**AraQwen3_VL** extends **AraConfig** with Qwen3-VL-specific config params for vision model inputs. It is referenced as the `ara_class` by all three Qwen3 config variants.

```python
class AraQwen3_VL(AraConfig):
    vision_model_path: str | None = None
    rope_table_path: str | None = None
    vision_pos_table_path: str | None = None
    vision_pos_table_scale_path: str | None = None
```

### Config Parameters:

| Parameter                     | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| `vision_model_path`           | Path to the Vision model `.dvm` file.                                 |
| `rope_table_path`             | Path to the rope table `.dat` file required for vision inference.     |
| `vision_pos_table_path`       | Path to the vision position table `.dat` file (Qwen3-specific).       |
| `vision_pos_table_scale_path` | Path to the vision position table scale `.dat` file (Qwen3-specific). |

Checkout `AraConfig` class for more info, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

---

## AraQwen3VLConfig

**AraQwen3VLConfig** extends **AraPretrainedConfig** for the base Qwen3-VL video model.

```json
{
  "ara": {
    "vision_model_path": "./models/qwen3-vl/vision_encoder/model.dvm",
    "rope_table_path": "./models/qwen3-vl/vision_encoder/inputs/rope_table.dat",
    "vision_pos_table_path": "./models/qwen3-vl/vision_encoder/inputs/vision_pos_table.dat",
    "vision_pos_table_scale_path": "./models/qwen3-vl/vision_encoder/inputs/vision_pos_table_scale.dat"
  },
  "model_type": "ara_qwen3_vl",
  "video_token_id": 151656,
  "image_token_id": 151655
}
```

### Config Parameters:

| Parameter        | Description                                         |
| ---------------- | --------------------------------------------------- |
| `model_type`     | `"ara_qwen3_vl"`                                    |
| `video_token_id` | Token ID for video placeholders (default: `151656`) |
| `image_token_id` | Token ID for image placeholders (default: `151655`) |

### Example:

```python
from optimum.ara import AraQwen3VLConfig

config = AraQwen3VLConfig.from_pretrained("./models/qwen3-vl/")
```

---

## AraQwen3ImageConfig

**AraQwen3ImageConfig** extends **AraPretrainedConfig** for the image-only Qwen3-VL model.

```json
{
  "model_type": "ara_qwen3_image",
  "video_token_id": 151656,
  "image_token_id": 151655
}
```

### Config Parameters:

| Parameter        | Description                                         |
| ---------------- | --------------------------------------------------- |
| `model_type`     | `"ara_qwen3_image"`                                 |
| `video_token_id` | Token ID for video placeholders (default: `151656`) |
| `image_token_id` | Token ID for image placeholders (default: `151655`) |

---

## AraQwen3MultiModalConfig

**AraQwen3MultiModalConfig** extends **AraPretrainedConfig** for the multimodal Qwen3-VL model supporting image, video (any duration), and text-only inference with arbitrary system prompts.

```json
{
  "model_type": "ara_qwen3_multimodal",
  "video_token_id": 151656,
  "image_token_id": 151655,
  "sys_pad_token_id": 220
}
```

### Config Parameters:

| Parameter          | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `model_type`       | `"ara_qwen3_multimodal"`                                                       |
| `video_token_id`   | Token ID for video placeholders (default: `151656`)                            |
| `image_token_id`   | Token ID for image placeholders (default: `151655`)                            |
| `sys_pad_token_id` | Token ID used to pad system prompt to rope alignment boundary (default: `220`) |

---

## AraQwen3VLForConditionalGeneration

**AraQwen3VLForConditionalGeneration** extends **AraModelForVisualCausalLM** for Qwen3-VL models. When using the model for vision-language tasks, the flow is:

1. Vision encoder processes image/video frames and outputs feature embeddings (main + ds0, ds1, ds2 deepstack features).
2. Language model handles text generation with merged vision+text embeddings.

Qwen3-VL differs from Qwen2.5-VL in:

- **5 vision DVM inputs** (pixels, scale, rope table, vision pos table, vision pos table scale) vs 3 in Qwen2.5
- **8 vision DVM outputs** (main features+scale + ds0/ds1/ds2 features+scale) vs 2 in Qwen2.5
- **Frame divisor** of 16 (vs 28 for Qwen2.5)
- **quantize()** accepts `torch.Tensor` directly (vs `np.ndarray` with manual `*32` scaling in Qwen2.5)

Key attributes:

- `PATCHES_ALONG_HEIGHT = 12`
- `PATCHES_ALONG_WIDTH = 12`
- `PATCH_SIZE = 16`
- `NUM_FRAMES = 16` (default, overridable)
- `NUM_TKNS_PER_FRAME = 144` (12 x 12)

User APIs are as parent class. Checkout [AraModelForVisualCausalLM](../base_classes/ara_model_for_visual_language.md#aramodelforvisualcausallm).

### Example:

```python
from optimum.ara import AraQwen3VLForConditionalGeneration

model = AraQwen3VLForConditionalGeneration.from_pretrained("./models/qwen3-vl/")
```

---

## AraQwen3ImageForConditionalGeneration

**AraQwen3ImageForConditionalGeneration** extends **AraQwen3VLForConditionalGeneration** for image-only inference. It sets `is_image_model=True` and `NUM_FRAMES=2`.

### Example:

```python
from optimum.ara import AraQwen3ImageForConditionalGeneration

model = AraQwen3ImageForConditionalGeneration.from_pretrained("./models/qwen3-image/")
```

---

## AraQwen3MultiModalForConditionalGeneration

**AraQwen3MultiModalForConditionalGeneration** extends **AraQwen3VLForConditionalGeneration** for full multimodal inference supporting image, video (any duration), and text-only. It sets `is_multimodal_model=True`. Vision tokens start at a multiple of 128, with system prompt padded using `sys_pad_token_id=220` for rope alignment.

### Example:

```python
from optimum.ara import AraQwen3MultiModalForConditionalGeneration

model = AraQwen3MultiModalForConditionalGeneration.from_pretrained(
    "./models/qwen3-vl-multimodal/"
)
```

### Full usage example (from `examples/qwen3_vl_multimodal.py`):

```python
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    TextStreamer,
)
from optimum.ara import (
    AraQwen3MultiModalConfig,
    AraQwen3MultiModalForConditionalGeneration,
    Qwen3VLProcessor,
    AraGenerationConfig,
)
from pathlib import Path

PROCESSOR_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_PATH = Path("./models/qwen3-vl-multimodal")

config = AutoConfig.from_pretrained(MODEL_PATH / "config.json")
generation_config = AraGenerationConfig.from_pretrained(
    MODEL_PATH / "generation_config.json"
)

model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, config=config)
processor = AutoProcessor.from_pretrained(PROCESSOR_ID)

streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

# --- Video inference ---
VIDEO_PATH = "path/to/video.mp4"
resized_video_path = Qwen3VLProcessor.resize_video(VIDEO_PATH)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": resized_video_path, "fps": 2},
            {"type": "text", "text": "Describe the video."},
        ],
    },
]
processed_inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
model.generate(
    **processed_inputs,
    generation_config=generation_config,
    max_new_tokens=512,
    do_sample=False,
    stream=True,
    streamer=streamer,
)

# --- Image inference ---
IMAGE_PATH = "path/to/image.jpg"
resized_image_path = Qwen3VLProcessor.resize_image(IMAGE_PATH)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": resized_image_path},
            {"type": "text", "text": "Describe the image."},
        ],
    },
]
processed_inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
model.generate(
    **processed_inputs,
    generation_config=generation_config,
    max_new_tokens=512,
    do_sample=False,
    stream=True,
    streamer=streamer,
)

# --- Text-only inference ---
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant"}],
    },
    {"role": "user", "content": [{"type": "text", "text": "Tell me a joke."}]},
]
processed_inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
model.generate(
    **processed_inputs,
    generation_config=generation_config,
    stream=True,
    streamer=streamer,
)
```

---

## Qwen3VLProcessor

**Qwen3VLProcessor** is a utility class providing image/video preprocessing and quantization for Qwen3-VL models.

### Methods:

| Method                                                          | Description                                                                                    |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `resize_image(input_path, target_width=384, target_height=384)` | Resizes an image to the target dimensions and returns a temp file path.                        |
| `resize_video(input_path, target_width=384, target_height=384)` | Resizes a video to the target dimensions and returns a temp file path.                         |
| `open_image(source)`                                            | Opens an image from a local path or URL, returns a PIL `Image`.                                |
| `quantize(inp_tensor: torch.Tensor)`                            | Quantizes a `torch.Tensor` to int8 with per-group-of-64 scales. Returns `(scales, quantized)`. |
| `dequantize(quantized, scales, shape)`                          | Dequantizes int8 data back to float using scales and target shape.                             |

### Processing constants:

| Constant   | Value     |
| ---------- | --------- |
| Image size | `384x384` |
| Video FPS  | `2`       |

### Example:

```python
from optimum.ara import Qwen3VLProcessor

# Resize image to 384x384
resized_path = Qwen3VLProcessor.resize_image("path/to/image.jpg")

# Resize video to 384x384
resized_path = Qwen3VLProcessor.resize_video("path/to/video.mp4")

# quantize/dequantize (used internally)
import torch

tensor = torch.randn(1024)
scales, quantized = Qwen3VLProcessor.quantize(tensor)
restored = Qwen3VLProcessor.dequantize(quantized.numpy(), scales.numpy(), tensor.shape)
```

---

## HF Auto* Registration

All Qwen3 classes are registered with the HuggingFace transformers library for seamless integration:

| model_type             | AutoConfig                 | AutoModelForImageTextToText                  | AutoModelForVision2Seq                       |
| ---------------------- | -------------------------- | -------------------------------------------- | -------------------------------------------- |
| `ara_qwen3_vl`         | `AraQwen3VLConfig`         | `AraQwen3VLForConditionalGeneration`         | `AraQwen3VLForConditionalGeneration`         |
| `ara_qwen3_image`      | `AraQwen3ImageConfig`      | `AraQwen3ImageForConditionalGeneration`      | —                                            |
| `ara_qwen3_multimodal` | `AraQwen3MultiModalConfig` | `AraQwen3MultiModalForConditionalGeneration` | `AraQwen3MultiModalForConditionalGeneration` |

This allows loading via `AutoConfig.from_pretrained()` and `AutoModelForImageTextToText.from_pretrained()` by specifying the correct model path.

---

## Class Hierarchy

```
AraConfig
  └── AraQwen3_VL  (vision-specific ara fields)

AraPretrainedConfig
  ├── AraQwen3VLConfig        (model_type: ara_qwen3_vl)
  ├── AraQwen3ImageConfig     (model_type: ara_qwen3_image)
  └── AraQwen3MultiModalConfig (model_type: ara_qwen3_multimodal)

AraModel (PreTrainedModel)
  └── AraModelForVisualCausalLM
        └── AraQwen3VLForConditionalGeneration
              ├── AraQwen3ImageForConditionalGeneration   (is_image_model=True, NUM_FRAMES=2)
              └── AraQwen3MultiModalForConditionalGeneration (is_multimodal_model=True)
```
