# AraPretrainedConfig APIs

To use **AraPretrainedConfig** class, we import it from `optimum.ara` in the below mentioned way.

```python

from optimum.ara import AraPretrainedConfig

```
**AraPretrainedConfig** extends `PretrainedConfig` class with `ara` specific config parameters.

Model Specific config classes can extent **AraPretrainedConfig**  class to add more model specific config parameters.

Following is ara specific config params which are required for `optimum.ara` and present in  **AraPretrainedConfig**.

We have choosen an approch which keeps `ara` specifc config seprate from transformer config pramas for clarity.

```json
{
  "ara": {
    "dvm_path": "models/llama2-7b/assets/model.dvm",
    "interface_ip_address": "127.0.0.1",
    "interface_named_pipe": "//./pipe/proxy_pipe",
    "interface_port": 5000,
    "interface_socket_file": "/var/run/proxy.sock",
    "interface_type": "SOCKET"
  },
  "model_type": "ara_llama"
}

```

### Config Parameters:

| Parameters                                 | Description                                                                                                               |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `dvm_path`                                 | Path to LLM model dvm file we want to use.                                                                                |
| `interface_ip_address`                     | IP Address set in proxy_config.yaml (IP Address use by proxy service), Its used only if `interface_type` is set to `IPV4` |
| `interface_port`                           | named_pipe set in proxy_config.yaml, Its used only if `interface_type` is set to `IPV4`                                   |
| `interface_socket_file`                    | (Only Linux) socket_file set in proxy_config.yaml, Its used only if `interface_type` is set to `SOCKET`                   |
| `interface_named_pipe (unsupport for now)` | (Only Windows) named_pipe set in proxy_config.yaml, Its used only if `interface_type` is set to `NAMED_PIPE`              |
| `interface_type`                           | interface_type set in proxy_config.yaml (interface on which proxy is running), options `IPV4`, `SOCKET`, `NAMED_PIPE`     |


Below is the list of APIs

- [`from_pretrained()`](#from_pretrained)
- [`from_dict()`](#from_dict)

## from_pretrained()

Provides the directory containing `config.json` and optionally config parameters and returns an instance of **AraPretrainedConfig**.

```python
 def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "AraPretrainedConfig":
```
  
### Arguments:

| Arguments                                                | Description                                                                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `pretrained_model_name_or_path(Union[str, os.PathLike])` | Can be a string, the model id of a pretrained model configuration, a directory path with a JSON config file. |
| `**kwargs`                                               | config values can provided in function call itself to overwite values read from config.json                  |

### Returns:

This returns an instance of **AraPretrainedConfig**.

### Example:

Test loading a causalLM config from local directory using
AraPretrainedConfig.from_pretrained(). Assumes model is compiled into dvm format.

```python
import os
from optimum.ara import AraPretrainedConfig

# Path to directory containing config.json
config = AraPretrainedConfig.from_pretrained("models/llama2-7b")
# Complete Path to config.json file
config = AraPretrainedConfig.from_json_file("models/llama2-7b/config.json")
# Hugging Face Hub model id, config.json will be downloaded from huggingface hub
config = AraPretrainedConfig.from_pretrained("nxp/Qwen2.5-Coder-1.5B-Ara240")

```

## from_dict()

Loads **AraPretrainedConfig** directly from the dictionary passed in the function. It builds a config object from a dictionary instead of a file.

```python
def from_dict(
    cls, 
    config_dict, 
    **unused_kwargs
    )
```
### Arguments:

| Arguments                                   | Description                                                                                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config_dict`                               | Dictionary that is used to instantiate the configuration object.                                                                                                    |
| `**unused_kwargs` **(unsupported for now)** | It is a dictionary containing key values from kwargs that were not used to update the configuration, representing keys that are not recognized as config attributes |

### Returns:

This returns an instance of **AraPretrainedConfig**.

### Example:

```python

from optimum.ara import AraPretrainedConfig

config_dict = {
  "ara": {
    "dvm_path": "models/llama2-7b/assets/model.dvm",
    "interface_socket_file": "/var/run/proxy.sock",
    "interface_type": "SOCKET"
  },
  "model_type": "ara_llama"
}

config = AraPretrainedConfig.from_dict(config_dict)
```
