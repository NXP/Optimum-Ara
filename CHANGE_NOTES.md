Ticket: [HAWK-388](https://kinara.atlassian.net/browse/HAWK-388) , PR: [#68](https://gitea.kinara.ai/eng/optimum-ara/pulls/68)
Date: 24 April, 2026, Author: [Shayaan Jamil](https://gitea.kinara.ai/xshayaan), [xshess](https://gitea.kinara.ai/xshess)

### `QwenVLProcessor` renamed to `Qwen2_5_VLProcessor`

`QwenVLProcessor` has been renamed to `Qwen2_5_VLProcessor` to better reflect the model family it supports and to align with the upstream Transformers naming convention.

- **New name**: `Qwen2_5_VLProcessor` — a thin subclass of the Transformers `Qwen2_5_VLProcessor` that enforces fixed image dimensions (336×336) and video FPS (2) via `apply_chat_template`.
- **Deprecated alias**: `QwenVLProcessor` is still importable from `optimum.ara` and `optimum.ara.models.qwen2_5_vl.processing_qwen2_5_vl` but will emit a `logging.WARNING` on instantiation and **will be removed in a future release**.
- **Migration**: Replace all usages of `QwenVLProcessor` with `Qwen2_5_VLProcessor`.

```python
# Before (deprecated)
from optimum.ara import QwenVLProcessor
processor = QwenVLProcessor()

# After
from optimum.ara import Qwen2_5_VLProcessor
processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

----
Ticket: [HAWK-417](https://kinara.atlassian.net/browse/HAWK-417) , PR: [#63](https://gitea.kinara.ai/eng/optimum-ara/pulls/63)
Date 24 April, 2026, Author: [Naveen Mittal](https://gitea.kinara.ai/naveen)

1. `generate` now streams input prompt token ids as well, as expected by streamer APIs.
2. If optimum-ara failed to connect to proxy using provided interface_type, it will try to connect using other available interface_types with availabe config values


Ticket: [HAWK-322](https://kinara.atlassian.net/browse/HAWK-322) , PR: [#35](https://gitea.kinara.ai/eng/optimum-ara/pulls/35) , Commit ID [8e1df40314](https://gitea.kinara.ai/eng/optimum-ara/commit/8e1df4031495580717624e46a12aa228414a0538): , Date 25th Feb, 2026: , Author: [Shayaan Jamil](https://gitea.kinara.ai/xshayaan) 

1. `generate` return output changes for `AraModelForVisualCausalLM`

- Keeping in line with Hugging face, now generate function will return either a `torch.LongTensor` containing inputs + the generated output
- If `return_dict_in_generate` key in generation config is set to `True`, generate function will return a `GenerateDecoderOnlyOutput` which is a transformers `ModelOutput` class which will contain outputs in the `sequences` key.

```python
#before
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[Union[GenerationConfig, AraGenerationConfig]] = None,
    ...) -> Union[torch.Tensor, Any]:
         generated_list = torch.tensor(generated_list, dtype=torch.int32)
        return generated_list
#now
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[Union[GenerationConfig, AraGenerationConfig]] = None,
    ...) ->  Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        generated_list = torch.LongTensor(generated_list)
        output_torch = torch.cat(
            [torch.from_numpy(numpy_inputs), generated_list.unsqueeze(0)], dim=1
        )
        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=output_torch)
        return output_torch
```

2. `generate` return output changes for `AraModelForCausalLM`
 
- Keeping in line with Hugging face, now generate function will return a `torch.LongTensor` or `GenerateDecoderOnlyOutput` containing inputs + the generated output.

----
<br>
