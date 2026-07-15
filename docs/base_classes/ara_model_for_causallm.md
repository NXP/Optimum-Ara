# AraModelForCausalLM APIs

**AraModelForCausalLM** is a base class used by model specific class to build on top of it. This class is specific for LLM models.

**Note:** This base class is useful to add support for custom LLM model that is not already supported in optimum-ara.

Below is the list of APIs:

- [`from_pretrained()`](#from_pretrained)
- [`from_config()`](#from_config)
- [`generate()`](#generate)

## from_pretrained()

Establishes a connection with the Ara device and loads the model on it.

```python
def from_pretrained(
    cls,
    pretrained_model_name_or_path: Union[str, Path, os.PathLike],
    config: Optional[Union[PretrainedConfig, AraPretrainedConfig]] = None,
    *args,
    **kwargs,
) -> "AraModelForCausalLM":
 ``` 
### Arguments:

| Arguments                                                                                        | Description                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pretrained_model_name_or_path `                                                                 | Path to directory containing config.json                                                                                                                                                                  |
| `config `                                                                                        | Optional configuration object to initialize the model object, if config is provide `pretrained_model_name_or_path ` is used only as backup to find for model.dvm file if config doesn't have correct path |
| `args`                                                                                           | Forward addition arguments to parent class.                                                                                                                                                               |
| `kwargs : following configs are support through kwargs`                                          |                                                                                                                                                                                                           |
| `device_map (Optional[Union[str, dict[str, Union[int, str, torch.device]], int, torch.device]])` | In case of multiple endpoint, specify which endpoint to load the model on.                                                                                                                                |
| `max_memory (Optional[dict])` **(unsupported for now)**                                          | specify memory limit on individual endpoints.                                                                                                                                                             |
| `file_name (Optional[str])`                                                                      | Specific model file name.                                                                                                                                                                                 |
| `use_cache (Optional[bool])` **(unsupported for now)**                                           | Indicates whether the model should use previously computed key/value attention states to accelerate decoding, if supported.                                                                               |

### Raises:

- `ValueError`:
    - Raised when the parameters combination use_cache=False, use_merged=True" is not supported. To use a merged decoder, past key values must be used.
    - 	All of `model_id`, `file_name`, and `config` are missing.

- `FileNotFoundError`:
    - Raised when none of the methods could find any DVM model file.

- `NotImplementedError`:
    - Currently, handling multiple .dvm files in the same folder is unsupported. So this is raised when more than one .dvm file found in the folder. 

### Returns:

returns **AraModelForCausalLM** class object.

## from_config()

Instantiates the model from a model configuration object.
Its specially useful when config parameters need to updated from script at runtime, like mode_dvm path etc.

```python
def from_config(
    cls, 
    config: AraPretrainedConfig, 
    **kwargs
):
```


### Arguments:

| Arguments                       | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| ` config (AraPretrainedConfig)` | provided config object used to create model object.       |
| `**kwargs`                      | Additional parameters used to initialize the model class. |

### Returns:

returns **AraModelForCausalLM** class object.

### Example:

Test loading a causalLM model from local config.json using AraModelForCausalLM.from_config(). Assumes model is compiled into dvm format.

```python
import os
from optimum.ara import AraModelForCausalLM, AraPretrainedConfig

# Load the configuration from a JSON file using its full file path
config = AraPretrainedConfig.from_json_file(
    os.path.realpath("models/llama2-7b/config.json")
)

# Load the model using the configuration object
model = AraModelForCausalLM.from_config(config)
```

## generate()

Generates text sequences for given input tokens.

```python
def generate(  # pyrefly: ignore[bad-override]
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[Union[GenerationConfig, AraGenerationConfig]] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):
```

### Arguments:

| Arguments                                           | Description                                                                                                                                                                                                                                   |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs (Optional[torch.Tensor])`                   | The input tensor used as a prompt for text generation or passed to the model's encoder.                                                                                                                                                       |
| `generation_config (Optional[AraGenerationConfig])` | Defines generation settings like maximum length and sampling behavior. Any matching **kwargs will override its values. If not supplied, it's loaded from generation_config.json if present, otherwise from the model's default configuration. |
| `logits_processor (Optional[LogitsProcessorList])`  | Custom logits processors.                                                                                                                                                                                                                     |
| `streamer (Optional[BaseStreamer])`                 | A streamer object used to handle streaming of generated tokens. Tokens are passed using `streamer.put(token_ids)`, and the streamer handles output processing.                                                                                |
| `**kwargs`                                          | Additional parameters for generation or model-specific settings.                                                                                                                                                                              |

### Raises:

- `DvApiException`:
    - Raised when `_generate_first_token()` method fails to generate first token.

### Returns:

Generates token IDs for the provided tensor/prompt.

### Example

```python
from optimum.ara import AraModelForCausalLM, AraGenerationConfig  
from transformers import AutoTokenizer  

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "how are you?"},
]

inputs = self.tokenizer.apply_chat_template(
    conversation=messages, tokenize=False, add_generation_prompt=True
)
inputs = self.tokenizer(inputs)

output = self.model.generate(**inputs)
```
