# AraGenerationConfig APIs

To use **AraGenerationConfig** class, we import it from `optimum.ara` in the below mentioned way.

```python
from optimum.ara import AraGenerationConfig
```

**AraGenerationConfig** extends transformers `GenerationConfig` class and adds `ara` specific generation config prameters.

Model Specific config classes can extent **AraGenerationConfig**  class to add more model specific config parameters.

Following is ara specific config params which are required for `optimum.ara` and present in  **AraGenerationConfig**.

We have choosen an approch which keeps `ara` specifc config seprate from transformer config pramas for clarity.
```python
{
  "ara": {
    "target_prompt_post_mcp": 1,
    "target_prompt_pre_mcp":  1,
    "target_token_post_mcp":  1,
    "target_token_pre_mcp":   1
  },
  "bos_token_id": 1,
  "do_sample": false,
}
```

### Generation Config Parameters:

| Parameters               | Description                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `target_prompt_post_mcp` | If value is 1, first token return from `ara` device is token_id , If value is 0, first token return from `ara` device is a logit vector                                                                            |
| `target_prompt_pre_mcp`  | If value is 1, prompt token_ids are directly provided to  `ara` device, If value is 0, prompt token_ids are converted to embeddings on host device and embeddings are passed to `ara` device                       |
| `target_token_post_mcp`  | If value is 1, (second onwards) token selection processing happend on `ara` device , If value is 0, (second onwards) tokens `ara` device returns logit vectors and token selection process happends on host device |
| `target_token_pre_mcp`   | If value is 1, (second onwards) token_ids are directly provided to `ara` device, If value is 0, (second onwards) token_ids are converted to embeddings on host device and embeddings are passed to `ara` device    |

Below is the list of APIs

- [`from_pretrained()`](#from_pretrained)
- [`to_json_string()`](#to_json_string)

## from_pretrained()

Loads an **AraGenerationConfig** from a pretrained model directory or file.

```python
def from_pretrained(
    cls,
    pretrained_model_name: Union[str, os.PathLike],
    config_file_name: Optional[Union[str, os.PathLike]] = None,
    **kwargs,
) -> "AraGenerationConfig":
```

### Arguments:

| Arguments                                         | Description                                                                                                                                                                                         |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pretrained_model_name_or_path (str or PathLike)` | Can be a string, the model id of a pretrained model configuration, a directory path with a saved configuration, for example, using `save_pretrained()`, or a direct path/URL to a JSON config file. |
| `config_file_name (str or PathLike)`              | Can be a string, the model id of a pretrained model configuration, a directory path with a saved configuration, for example, using `save_pretrained()`, or a direct path/URL to a JSON config file. |
| `**kwargs`                                        | Additional configuration values that override existing settings when loading the model.                                                                                                             |

### Returns:

Returns an instance of **AraGenerationConfig**.

### Example:

In the below eaxmple, a causalLM config is loaded from local directory using AraGenerationConfig.from_pretrained(). Assumes model is compiled into dvm format.

```python
from optimum.ara import AraGenerationConfig

config = AraGenerationConfig.from_pretrained("models/llama2-7b")
config = AraGenerationConfig.from_pretrained("models/llama3.1-8b/generation_config.json")
```

## to_json_string()

Serializes the configuration to a JSON-formatted string.

```python

def to_json_string(
    self, 
    use_diff: bool = False, 
    ignore_metadata=False
) -> str:

```
### Arguments:

| Arguments                                                    | Description                                                                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `use_diff (bool, optional)` **(unsupported for now)**        | If True, only the changes from the default AraPretrainedConfig() are saved to the JSON file. Default is True. |
| `ignore_metadata (bool, optional)` **(unsupported for now)** | Whether or not to ignore metadata.                                                                            |

### Returns:

Provides complete configuration including Ara-specific settings as a clean, indented JSON string.


### Example

```python
from optimum.ara import AraGenerationConfig

config = AraGenerationConfig.from_pretrained("models/llama2-7b")  
json_string = config.to_json_string()
print(json_string)
```