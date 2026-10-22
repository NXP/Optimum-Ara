
## Run Example with Optimum-Ara

This section shows you how to run an example using Optimum-Ara after installation. THis example uses `Qwen2.5-Coder-1.5B` model, fetched from [nxp/Qwen2.5-Coder-1.5B-Ara240](https://huggingface.co/nxp/Qwen2.5-Coder-1.5B-Ara240).

1. Fetch the model from Hugging Face:

   ```bash
   uv run hf download nxp/Qwen2.5-Coder-1.5B-Ara240 --local-dir /usr/share/llm/Qwen2.5-Coder-1.5B
   ```

1. Create a Python script with below example code. You can name it something like `qwen2_5_coder_1.5b.py`:

   > Note: You might need to change the `MODEL_PATH` and `TOKENIZER_PATH` to match your setup. This one works on i.MX platforms with the default BSP setup.

   ```python
   # Copyright 2026 NXP
   # SPDX-License-Identifier: Apache-2.0

   import os
   import sys

   from transformers import Qwen2TokenizerFast, AutoModelForCausalLM
   import optimum.ara
   from transformers import TextStreamer

   MODEL_PATH = os.path.realpath("/usr/share/llm/Qwen2.5-Coder-1.5B")
   TOKENIZER_PATH = "/usr/share/llm/Qwen2.5-Coder-1.5B/tokenizer"

   INPUT_PROMPT = "Write a Python script that computes the factorial of a number."

   tokenizer = Qwen2TokenizerFast.from_pretrained(TOKENIZER_PATH)
   model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

   messages = [
       {"role": "system", "content": "You are a helpful python coding assistant."},
       {"role": "user", "content": INPUT_PROMPT},
   ]

   inputs = tokenizer.apply_chat_template(
       conversation=messages, tokenize=False, add_generation_prompt=True
   )
   inputs = tokenizer(inputs, return_tensors="pt")
   streamer = TextStreamer(tokenizer, stream=sys.stdout)

   print("\n" + "=" * 50)
   print("USER PROMPT:")
   print("=" * 50)
   print(INPUT_PROMPT)
   print("=" * 50 + "\n")

   output = model.generate(**inputs, streamer=streamer, max_new_tokens=256)

   del model
   ```

1. Run the example with below command:

   ```bash
   uv run qwen2_5_coder_1.5b.py
   ```

   ````plain
   [2026-07-14 17:11:03] [INFO] [modeling_base.py:_get_llm_dvm_path] Found dvm_file, /usr/share/llm/Qwen2.5-Coder-1.5B/model.dvm
   [INFO] - DVAPI: loaded dvinfclient lib: /usr/lib/libaraclient_aarch64.so
   [2026-07-14 17:11:03] [INFO] [modeling_base.py:_handle_device_map] Model size: 1.73 GB, Additional required memory: 600.00 MB
   [2026-07-14 17:11:03] [INFO] [modeling_base.py:_handle_device_map] Total size required for Model: 2.32 GB
   [2026-07-14 17:11:03] [INFO] [modeling_base.py:_handle_device_map] Endpoint : 0 has free space : 15.77 GB
   I:DVPULB[260714171103] model type is ara2 llm
   I:DVPULB[260714171103] Successfully read llm_params_t from model file
   I:DVPULB[260714171103] model found to be dyn quant v2 model

   ==================================================
   USER PROMPT:
   ==================================================
   Write a Python script that computes the factorial of a number.
   ==================================================

       ```python
       def factorial(n):
           # Base case: factorial of 0 or 1 is 1
           if n == 0 or n == 1:
               return 1
           # Recursive case: n * factorial of (n-1)
           else:
               return n * factorial(n-1)

       # Example usage
       number = int(input("Enter a number to calculate its factorial: "))
       result = factorial(number)
       print(f"The factorial of {number} is {result}")
       ```

   In this solution, the `factorial` function calculates the factorial of a given number `n`. The function uses recursion to multiply each number from `n` down to `1`, effectively computing the product. The base cases are when `n` is either 0 or 1, in which case it returns 1. For all other values of `n`, it calls itself with `n-1`, multiplying the result by the current value of `n`. This recursive approach allows us to efficiently compute factorials without explicitly looping through all numbers up to and including that value.<|im_end|>
   ````

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

### examples/llava_demo.py
This example runs LLAVA Image model.

```bash
# Steps to run Examples

# 1. Update 'dvm_path' path in models/llava/config.json to point to model.dvm
# 2. Run from optimum-ara (top folder)

python examples/llava_demo.py

```

### examples/prompt_lookup.py

This example shows how to use custom_logits_procrssor with prompt_loopkup decoding.

```bash
# Steps to run Examples

# 1. Update 'dvm_path' path in models/llava/config.json to point to model.dvm
# 2. Run from optimum-ara (top folder)

python examples/prompt_lookup.py

```
