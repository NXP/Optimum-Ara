# Ara Llama Model Classes

Purpose of these classes is to run an `ara` model on `ara` device with the ease of hugging face apis. 

We have added two LlaMA specific classes.
- [Ara Llama Model Classes](#ara-llama-model-classes)
  - [AraLlamaConfig()](#arallamaconfig)
    - [Example:](#example)
  - [AraLlamaForCausalLM()](#arallamaforcausallm)
    - [Example:](#example-1)

## AraLlamaConfig()

**AraLlamaConfig** is a configuration class that extends `AraPretrainedConfig` (which extends transformers **PretrainedConfig** class) with `ara llama` specific configuration parameters.

```json
model_type = "ara_llama"
```

Checkout `AraPretrainedConfig` class for more info, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

### Example:

```python
from optimum.ara import AraLlamaConfig

config = AraLlamaConfig.from_pretrained("./models/llama2-7b/")
```

## AraLlamaForCausalLM()

**AraLlamaForCausalLM** is a model specific class which extends `AraModelForCausalLM`

Checkout `AraModelForCausalLM` class for more info, refer [AraModelForCausalLM](ara_model_for_causallm.mdx). 

### Example:

```python
from optimum.ara import AraLlamaForCausalLM

model = AraLlamaForCausalLM.from_pretrained("./models/llama2-7b/")
```