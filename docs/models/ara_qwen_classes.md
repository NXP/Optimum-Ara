# Ara Qwen Model Classes

Purpose of these classes is to run an `ara` model on `ara` device with the ease of hugging face apis.

We have added two LlaMA specific classes.

- [AraQwenConfig](#araqwenconfig)
- [AraQwenForCausalLM](#araqwenforcausallm)

## AraQwenConfig()

**AraQwenConfig** is a configuration class that extends `AraPretrainedConfig` (which extends transformers **PretrainedConfig** class) with `ara qwen` specific configuration parameters.

```json
model_type = "ara_qwen"
```

Checkout `AraPretrainedConfig` class for more info, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

### Example:

```python
from optimum.ara import AraQwenConfig

config = AraQwenConfig.from_pretrained("./models/qwen2_7b/")
```

## AraQwenForCausalLM()

**AraQwenForCausalLM** is a model specific class which extends `AraModelForCausalLM`

Checkout `AraModelForCausalLM` class for more info, refer [AraModelForCausalLM](../base_classes/ara_model_for_causallm.md).

### Example:

```python
from optimum.ara import AraQwenForCausalLM

model = AraQwenForCausalLM.from_pretrained("./models/qwen2_7b/")
```
