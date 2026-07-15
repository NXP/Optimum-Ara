# AraModelForVisualCausalLM

**AraModelForVisualCausalLM** is a base class used by Vision Specific Model classes.

**Note:** This base class is useful to add support for custom Vision models that is not already supported in optimum-ara.

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
    ) -> "AraModelForVisualCausalLM":
 ``` 
### Arguments:

| Arguments                                                                                        | Description                                                                                                                                                |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pretrained_model_name_or_path (Union[str, Path])`                                               | It can be a simple model ID in which case the default model is used. Alternatively, it can be a full path or repository name, such as **user_name/model**. |
| `config (Optional[AraPretrainedConfig])`                                                         | Optional configuration object to load instead of the default one. This also determines the tokenizer class to instantiate.                                 |
| `kwargs : following configs are support through kwargs`                                          |                                                                                                                                                            |
| `device_map (Optional[Union[str, dict[str, Union[int, str, torch.device]], int, torch.device]])` | In case of multiple endpoint, specify which endpoint to load the model on.                                                                                 |
| `max_memory (Optional[dict])` **(unsupported for now)**                                          | specify memory limit on individual endpoints.                                                                                                              |
| `file_name (Optional[str])`                                                                      | Specific model file name.                                                                                                                                  |
| `use_cache (Optional[bool])` **(unsupported for now)**                                           | Indicates whether the model should use previously computed key/value attention states to accelerate decoding, if supported.                                |


### Raises:

- `ValueError`:
    - Raised when the parameters combination use_cache=False, use_merged=True" is not supported. To use a merged decoder, past key values must be used.
    - 	All of `model_id`, `file_name`, and `config` are missing.

- `FileNotFoundError`:
    - Raised when none of the methods could find any DVM model file.

- `NotImplementedError`:
    - Currently, handling multiple .dvm files in the same folder is unsupported. So this is raised when more than one .dvm file found in the folder. 

### Returns:

returns **AraModelForVisualCausalLM** class object.

### Example:

In the below example, loading a causalLM model from local directory using AraModelForVisualCausalLM.from_pretrained(). Assumes model is compiled into dvm format.

```python
import os
from optimum.ara import AraModelForVisualCausalLM

model = AraModelForVisualCausalLM.from_pretrained(
    os.path.realpath("models/qwen2.5-vl/config.json")
)
```

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

| Arguments                       | Description                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| ` config (AraPretrainedConfig)` | loads class object using provided config object.                                                 |
| `**kwargs`                      | Additional parameters used to modify the configuration after loading or to initialize the model. |

### Returns:

Returns **AraModelForVisualCausalLM** class object.


### Example:

```python
import os
from optimum.ara import AraModelForVisualCausalLM, AraPretrainedConfig

config = AraPretrainedConfig.from_json_file(
    os.path.realpath("models/qwen2.5-vl/config.json")
)

model = AraModelForVisualCausalLM.from_config(config)
```

## generate()

Generates text sequences for given input tokens and feature condition i.e. Image or Audio.

```python
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[Union[GenerationConfig, AraGenerationConfig]] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    streamer: Optional[BaseStreamer] = None,
    **kwargs,
) -> Union[GenerateEncoderDecoderOutput, torch.LongTensor, Any]:
```

### Arguments:

| Arguments                                           | Description                                                                                                                                                                                                                                   |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs (Optional[torch.Tensor])`                   | The input tensor used as a Condition for Condition generation or passed to the model's encoder.                                                                                                                                               |
| `generation_config (Optional[AraGenerationConfig])` | Defines generation settings like maximum length and sampling behavior. Any matching **kwargs will override its values. If not supplied, it's loaded from generation_config.json if present, otherwise from the model's default configuration. |
| `logits_processor (Optional[LogitsProcessorList])`  | TCustom logits processors.                                                                                                                                                                                                                    |
| `streamer (Optional[BaseStreamer])`                 | A streamer object used to handle streaming of generated tokens. Tokens are passed using `streamer.put(token_ids)`, and the streamer handles output processing.                                                                                |
| `**kwargs`                                          | Additional parameters for generation or model-specific settings.                                                                                                                                                                              |
| `input_ids (Optional[torch.LongTensor])`            | tokenized prompt input                                                                                                                                                                                                                        |
| `pixel_values (Optional[torch.Tensor])`             | list of token ids used to control model behaviour.                                                                                                                                                                                            |
| `pixel_values_videos (Optional[torch.Tensor])`      | list of token ids used to control model behaviour.                                                                                                                                                                                            |

### Returns:

Generates token IDs for the provided tensor/prompt.

### Example

```python
from optimum.ara import AraQwen2_5_VLForConditionalGeneration, QwenVLProcessor, AraGenerationConfig
from transformers import TextStreamer

# Configuration
VIDEO_PATH = "./examples/assets/personFalling_8.mp4"
PROMPT = "Describe this video briefly"
MODEL_PATH = "./models/qwen2.5-vl-3B_variable_length"

# Initialize components
model = AraQwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH)
generation_config = AraGenerationConfig.from_pretrained( MODEL_PATH + "/generation_config.json")
streamer = TextStreamer(processor.processor.tokenizer, stream=sys.stdout)
processor = QwenVLProcessor()
processed_inputs = processor(prompt=PROMPT, video_path=VIDEO_PATH)

result = model.generate(
    **processed_inputs,
    generation_config=generation_config,
    max_new_tokens=512,
    temperature=1.0,
    do_sample=False,
    stream=False,
    streamer=streamer
)

generated_text = processor.processor.decode(result.flatten(), skip_special_tokens=True)
```
