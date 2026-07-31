# Auto Classes

Auto Classes are used to automatically load correct model class object based on `Model Task` and `model_type` in config file.

Config Classes:

- [AutoConfig](#autoconfig)

Task Based Model Classes:

- [AutoModelForCausalLM](#automodelforcausallm)
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

| Arguments                        | Description                                                                                                                                                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pretrained_model_name_or_path ` | Specifies the path to a directory containing config.json file or HF Hub model id.                                                                                                                                                                         |
| `device_map`                     | `None`: use "auto" <br> `"auto"`: check all endpoints one by one for available dram <br> `<int> or "ara:<int>`": load model on provided endpoint, space will not be checked in this case <br> `[0, 1]`: list of endpoint for MCS (Multi Chip Shard) Model |

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

- [AraLlamaForCausalLM](./ara_llama_classes.md)
- [AraQwenForCausalLM](./ara_qwen_classes.md)
- [AraLlavaForConditionalGeneration](./ara_llava_classes.md)
- [AraQwen2_5_VLForConditionalGeneration](./ara_qwen2.5_classes.md)
- [AraQwen2_5_ImageForConditionalGeneration](./ara_qwen2.5_classes.md)
- [AraModelForWhisper](./ara_whisper_classes.md)

## AutoModelForImageTextToText

APIs and their usage is same as [AutoModelForCausalLM](#automodelforcausallm)

#### Example

```python
from transformers import AutoModelForImageTextToText
from optimum import ara

# provide path to local directory where config.json is present
model = AutoModelForImageTextToText.from_pretrained(
    "<optimum-ara>/models/qwen2.5-image/"
)

config = AutoConfig.from_pretrained("<optimum-ara>/models/qwen2.5-image/")
config.ara.model_dvm = "<actual path>/model.dvm"
model = AutoModelForImageTextToText.from_config(config)
```

## AutoModelForVision2Seq

APIs and their usage is same as [AutoModelForCausalLM](#automodelforcausallm)

#### Example

```python
from transformers import AutoModelForVision2Seq
from optimum import ara

# provide path to local directory where config.json is present
model = AutoModelForVision2Seq.from_pretrained("<optimum-ara>/models/qwen2.5-vl/")

config = AutoConfig.from_pretrained("<optimum-ara>/models/qwen2.5-vl/")
config.ara.model_dvm = "<actual path>/model.dvm"
model = AutoModelForVision2Seq.from_config(config)
```

## AutoModelForSpeechSeq2Seq

APIs and their usage is same as [AutoModelForCausalLM](#automodelforcausallm)

#### Example

```python
from transformers import AutoModelForSpeechSeq2Seq
from optimum import ara

# provide path to local directory where config.json is present
model = AutoModelForSpeechSeq2Seq.from_pretrained("<optimum-ara>/models/whisper/")

config = AutoConfig.from_pretrained("<optimum-ara>/models/whisper/")
config.ara.model_dvm = "<actual path>/model.dvm"
model = AutoModelForSpeechSeq2Seq.from_config(config)
```
