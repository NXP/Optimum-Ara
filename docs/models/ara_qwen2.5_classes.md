# Qwen 2.5VL Model Classes

Qwen2.5-VL is a multimodal large language model from Qwen series that can understand both text and images. It supports tasks like visual question answering, image captioning, and document understanding, handling complex visual inputs.

We have added three Qwen 2.5VL specific classes.

- [AraQwen2_5VLConfig](#araqwen2_5vlconfig)
- [AraQwen2_5_VLForConditionalGeneration](#araqwen2_5_vlforconditionalgeneration)
- [Qwen2_5_VLProcessor](#Qwen2_5_VLProcessor)

We have added two Qwen 2.5V Image specific classes.

- [AraQwen2_5ImageConfig](#araqwen2_5imageconfig)
- [AraQwen2_5_ImageForConditionalGeneration](#araqwen2_5_imageforconditionalgeneration)

## AraQwen2_5VLConfig

**AraQwen2_5VLConfig** extends **AraPretrainedConfig** with AraQwen2.5-vl model specific config params.
Following config params are addition to params already present in **AraPretrainedConfig**

```json
{
  "ara": {
    "vision_model_path": "./models/qwen2.5_vl/vision_encoder/model.dvm",
    "input_scale_path": "./models/qwen2.5_vl/vision_encoder/inputs/layer1-input_0_scale-1_input_dv.dat",
    "rope_table_path": "./models/qwen2.5_vl/vision_encoder/inputs/layer2-rope_table-2_input_dv.dat"
  },
  "model_type": "ara_qwen_vl"
}
```

### Config Parameters:

| Parameters          | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| `vision_model_path` | Path to Vision model dvm file we want to use.                                          |
| `input_scale_path`  | path to fixed input file layer1-input_0_scale-1_input_dv.dat required for vision model |
| `rope_table_path`   | path to fixed input file layer2-rope_table-2_input_dv.dat required for vision model    |

Checkout `AraPretrainedConfig` class for more info, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

### Example:

```python
from optimum.ara import AraQwen2_5VLConfig

config = AraQwen2_5VLConfig.from_pretrained("./models/qwen2.5-vl/")
```

## AraQwen2_5_VLForConditionalGeneration

**AraQwen2_5_VLForConditionalGeneration** extends **AraModelForVisualCausalLM** for Qwen 2.5VL models. When using Qwen VLM model for vision-language tasks, the model follows the below flow:

    1. Vision encoder processes video and outputs features.
    2. Language model handles text generation with merged video+text embeddings.

User APIs are as parent class.
checkout [AraModelForVisualCausalLM](../base_classes/ara_model_for_visual_language.md#aramodelforvisualcausallm)

### Example:

```python
from optimum.ara import AraQwen2_5_VLForConditionalGeneration

model = AraQwen2_5_VLForConditionalGeneration.from_pretrained("./models/qwen2.5-vl/")
```

## Qwen2_5_VLProcessor

The **Qwen2_5_VLProcessor** extends transformers `Qwen2_5_VLProcessor`.

**Qwen2_5_VLProcessor** overwrites `apply_chat_template` function to validate `fps` and `resized dimentions` against supported values.

```python
# Processing parameters with constants
RESIZED_DIM = 336
FPS = 2
```

### Example

```python
from optimum.ara import Qwen2_5_VLProcessor

VIDEO_PATH = "path to video file"
PROMPT = "Describe this video briefly"

# Initialize the QwenVLProcessor
processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                "fps": 2,
                "resized_height": 336,
                "resized_width": 336,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
if image_inputs is None:
    image_inputs = []
if video_inputs is None:
    video_inputs = []

processed_inputs = processor(
    text=[text],
    videos=cast(List[Any], video_inputs if video_inputs else []),
    padding=True,
    return_tensors="pt",
)
```

We can achieve same result with `AutoProcessor` as well.

```python
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
image = Image.open(IMAGE_PATH)
prompt = "Describe this image."
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
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

processed_inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
```

## AraQwen2_5ImageConfig

**AraQwen2_5ImageConfig** extends **AraPretrainedConfig** with Qwen2.5 Image specific config params.

```python
model_type = "ara_qwen_image"
```

To know more about the AraPretrainedConfig class, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

### AraQwen2_5_ImageForConditionalGeneration

**AraQwen2_5_ImageForConditionalGeneration** is created on top of **AraModelForVisualCausalLM** for Qwen 2.5VL models. When using Qwen VLM model for vision-language tasks, the model follows the below flow:

    1. Vision encoder processes image and outputs features.
    2. Language model handles text generation with merged image+text embeddings.

User APIs are as parent class.
checkout [AraModelForVisualCausalLM](../base_classes/ara_model_for_visual_language.md#aramodelforvisualcausallm)
