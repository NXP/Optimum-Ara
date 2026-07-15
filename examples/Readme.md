# Run Examples

Before running examples, follow setup steps here [readme](../README.md)

## HuggingFace Hub Setup
```bash
# this provides huggingface-cli or huggingface-hub commands.
pip install -U huggingface_hub
#Create a token: Go to https://huggingface.co/settings/tokens, generate a “Read” token.
huggingface-cli login
# paste your token when prompted
#Verify with
huggingface-cli whoami 
```

>Note: llama tokenizer require agreeing to license on huggingface hub.
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

After this you will be able to use following in example script:
```python
TOKENIZER_PATH = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
```

### examples/causalLM.py

This example script can be used to run any LLM Model.
like: llama2-7b, llama3.1, qwen2, qwen2.5 etc

The default script look for following files at this relative location.

```bash
optimum-ara
├── models
│   ├── llama2-7b
│   │   ├── assets
│   │   │   └── model.dvm
│   │   ├── config.json
│   │   ├── generation_config.json

# you can use sym link of model.dvm here instead of copying model.dvm here or you can change the 'dvm_path' in config.json file.
```


```bash
# Steps to run Examples

# 1. Update 'dvm_path' path in models/<model name>/config.json to point to model.dvm
# 2. Update 'MODEL_PATH' model directory path in examples/casualLM.py to models/<model name>
# 3. Update tokenizer assests path or use model id in examples/casualLM.py
# 4. Run from optimum-ara (top folder)

python examples/casualLM.py

```

### examples/qwen_vl.py
This example runs Qwen2.5 Video model.

```bash
# Steps to run Examples

# 1. Update 'dvm_path' path in models/qwen2.5-vl-3B_variable_length/config.json to point to model.dvm
# 2. Run from optimum-ara (top folder)

python examples/qwen_vl.py

```

### examples/prompt_lookup.py

This example shows how to use custom_logits_processor with prompt_loopkup decoding.

```bash
# Steps to run Examples

# 1. Update 'dvm_path' path in models/llava/config.json to point to model.dvm
# 2. Run from optimum-ara (top folder)

python examples/prompt_lookup.py

```
