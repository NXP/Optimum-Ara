# LLAVA Model Classes

We have added two Qwen 2.5V Image specific classes.

- [AraLlavaConfig](#arallavaconfig)
- [AraLlavaForConditionalGeneration](#arallavaforconditionalgeneration)

## AraLlavaConfig

**AraLlavaConfig** extends **AraPretrainedConfig** with LLAVA specific config params.

```json
{
  "ara": {
    "vision_model_path": "models/llava/assets/llava_vit.dvm",
    "vision_input_path": "models/llava/assets/inputs/layer0-onnx__Cast_0-0_input_dv.dat",
    "vision_input_scale_path": "models/llava/assets/inputs/layer1-onnx__Cast_0_scale-1_input_dv.dat",
    "vision_add_input_path": "models/llava/assets/inputs/layer2-dv_Add_input-2_input_dv.dat",
    "vision_add_input_scale_path": "models/llava/assets/inputs/layer3-dv_Add_input_scale-3_input_dv.dat",
    "vision_concat_input_path": "models/llava/assets/inputs/layer4-dv_concat_input-4_input_dv.dat",
    "vision_concat_input_scale_path": "models/llava/assets/inputs/layer5-dv_concat_input_scale-5_input_dv.dat"
  },
  "model_type": "ara_llava"
}
```

### Config Parameters:

| Parameters                       | Description                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `vision_model_path`              | Path to Vision model dvm file we want to use.                                                  |
| `vision_input_path`              | path to fixed input file layer0-onnx__Cast_0-0_input_dv.dat required for vision model          |
| `vision_input_scale_path`        | path to fixed input file layer1-onnx__Cast_0_scale-1_input_dv.dat required for vision model    |
| `vision_add_input_path`          | path to fixed input file layer2-dv_Add_input-2_input_dv.dat required for vision model          |
| `vision_add_input_scale_path`    | path to fixed input file layer3-dv_Add_input_scale-3_input_dv.dat required for vision model    |
| `vision_concat_input_path`       | path to fixed input file layer4-dv_concat_input-4_input_dv.dat required for vision model       |
| `vision_concat_input_scale_path` | path to fixed input file layer5-dv_concat_input_scale-5_input_dv.dat required for vision model |

Checkout `AraPretrainedConfig` class for more info, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

### AraLlavaForConditionalGeneration

**AraLlavaForConditionalGeneration** is created on top of **AraModelForVisualCausalLM** for LLAVA models. When using LLAVA model for vision-language tasks, the model follows the below flow:

    1. Vision encoder processes image and outputs features.
    2. Language model handles text generation with merged image+text embeddings.

User APIs are as parent class.
checkout [AraModelForVisualCausalLM](../base_classes/ara_model_for_visual_language.md#aramodelforvisualcausallm)

```python
from optimum.ara import AraLlavaForConditionalGeneration

MODEL_PATH = "models/llava"

model = AraLlavaForConditionalGeneration.from_pretrained(MODEL_PATH)
```
