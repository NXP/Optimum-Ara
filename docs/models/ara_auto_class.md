# Auto Classes

Auto Classes are used to automatically load correct model class object based on `Model Task` and `model_type` in config file.

Config Classes:
- [AutoConfig](#autoconfig)

Task Based Model Classes:
- [Auto Classes](#auto-classes)
  - [AutoConfig](#autoconfig)
    - [Example](#example)
  - [AutoModelForCausalLM](#automodelforcausallm)
    - [from\_pretrained()](#from_pretrained)
      - [Arguments:](#arguments)
      - [Returns:](#returns)
      - [Example](#example-1)
    - [from\_config()](#from_config)
      - [Arguments:](#arguments-1)
      - [Returns:](#returns-1)
      - [Example](#example-2)
    - [generate()](#generate)
  - [AutoModelForImageTextToText](#automodelforimagetexttotext)
  - [AutoModelForVision2Seq](#automodelforvision2seq)
  - [AutoModelForSpeechSeq2Seq](#automodelforspeechseq2seq)

## AutoConfig

It takes a `config.json` file and returns model specific Config Class dependening upon `model_type` param set in `config.json` file.

### Example

Here is example on how to use **AutoConfig** to load `Ara` Configs:

```python
from transformers import AutoConfig
# `import optimum.ara` is required to register AraClasses with transformers
import optimum.ara

# provide path to local directory where config.json is present
config = AutoConfig.from_pretrained("<optimum-ara>/models/llama2-7b/")

# provide HF Hub model_id, downloads config.json from HF Hub
config = AutoConfig.from_pretrained("nxp/Qwen2.5-Coder-1.5B-Ara240")
```

## AutoModelForCausalLM

Below is the list of APIs:

- [`from_pretrained()`](#from_pretrained)
- [`from_config()`](#from_config)
- [`generate()`](#generate)

### from_pretrained()
```python
    .from_pretrained(
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
    ) -> SpecificPreTrainedModelType
 ``` 
#### Arguments:

| Arguments                        | Description                                                       |
| -------------------------------- | ----------------------------------------------------------------- |
| `pretrained_model_name_or_path ` | Path to directory containing config.json file or HF Hub model id. |

#### Returns:

Loads **SpecificPreTrainedModelType** model instance and returns a class object.

#### Example
```python

from transformers import AutoModelForCausalLM
from optimum import ara

# provide path to local directory where config.json is present
model = AutoModelForCausalLM.from_pretrained("<optimum-ara>/models/llama2-7b/")

config = AutoConfig.from_pretrained("<optimum-ara>/models/llama2-7b/")
config.ara.model_dvm = "<actual path>/model.dvm"
model = AutoModelForCausalLM.from_config(config)

```

### from_config()
```python
    .from_config(
        config: PreTrainedConfig,
    ) -> SpecificPreTrainedModelType
 ``` 
#### Arguments:

| Arguments | Description                                                                                                                                  |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `config ` | use config object to load model, useful if you want to change any config param from app before loading the model, for example model_dvm path |


#### Returns:
Loads **SpecificPreTrainedModelType** model instance and returns a class object.

#### Example
```python
from transformers import AutoModelForCausalLM
from optimum import ara

config = AutoConfig.from_pretrained("nxp/Qwen2.5-Coder-1.5B-Ara240")
config.ara.model_dvm = "<actual path>/model.dvm"
model = AutoModelForCausalLM.from_config(config)

```
### generate()

Checkout Specific Model Class documentation for generate.
- [AraLlamaForCausalLM](./models/ara_llama_classes.md)
- [AraQwenForCausalLM](./models/ara_qwen_classes.md)
- [AraQwen2_5_VLForConditionalGeneration](./models/ara_qwen2.5_classes.md)
- [AraQwen2_5_ImageForConditionalGeneration](./models/ara_qwen2.5_classes.md)

## AutoModelForImageTextToText
## AutoModelForVision2Seq
## AutoModelForSpeechSeq2Seq

APIs and their usage is same as [AutoModelForCausalLM](#automodelforcausallm)