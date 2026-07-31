# Optimum Ara APIs

Optimum-Ara is an extenstion of HuggingFace Transformers library with support for Ara Accelarator to enable High Performance AI at Edge.

Below is the list of Enabled Classes:

Task Based Classes:

- [AutoModelForCausalLM](#automodelforcausallm)
- [AutoModelForImageTextToText](#automodelforimagetexttotext)
- [AutoModelForVision2Seq](#automodelforvision2seq)
- [AutoModelForSpeechSeq2Seq](#automodelforspeechseq2seq)

Other Classes:

- [AutoConfig](#autoconfig)
- [AraGenerationConfig](#aragenerationconfig)

## AutoModelForCausalLM

**AutoModelForCausalLM** class is imported from `transformers`.
when importing `optimum.ara` we will register following classes with **AutoModelForCausalLM**

The **AutoModelForCausalLM** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraLlamaForCausalLM](./models/ara_llama_classes.md)
- [AraQwenForCausalLM](./models/ara_qwen_classes.md)

For more details see [AutoModelForCausalLM](./models/ara_auto_class.md#automodelforcausallm)

## AutoModelForImageTextToText

**AutoModelForImageTextToText** class is imported from `transformers`.
when importing `optimum.ara` we will register following classes with **AutoModelForImageTextToText**

The **AutoModelForImageTextToText** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraLlavaForConditionalGeneration](./models/ara_llava_classes.md#arallavaforconditionalgeneration)
- [AraQwen2_5_VLForConditionalGeneration](./models/ara_qwen2.5_classes.md#araqwen2_5_vlforconditionalgeneration)
- [AraQwen2_5_ImageForConditionalGeneration](./models/ara_qwen2.5_classes.md#araqwen2_5_imageforconditionalgeneration)

For more details see [AutoModelForImageTextToText](./models/ara_auto_class.md#automodelforimagetexttotext)

## AutoModelForVision2Seq

**AutoModelForVision2Seq** class is imported from `transformers`.
when importing `optimum.ara` we will register following classes with **AutoModelForVision2Seq**

The **AutoModelForVision2Seq** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraLlavaForConditionalGeneration](./models/ara_llava_classes.md#arallavaforconditionalgeneration)
- [AraQwen2_5_VLForConditionalGeneration](./models/ara_qwen2.5_classes.md#araqwen2_5_vlforconditionalgeneration)

For more details see [AutoModelForVision2Seq](./models/ara_auto_class.md#automodelforvision2seq)

## AutoModelForSpeechSeq2Seq

**AutoModelForSpeechSeq2Seq** class is imported from `transformers`.
when importing `optimum.ara` we will register following classes with **AutoModelForSpeechSeq2Seq**

The **AutoModelForSpeechSeq2Seq** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraModelForWhisper](./models/ara_whisper_classes.md#aramodelforwhisper)

For more details see [AutoModelForSpeechSeq2Seq](./models/ara_auto_class.md#automodelforspeechseq2seq)

## AutoConfig

**AutoConfig** class is imported from `transformers`.
when importing `optimum.ara` we will register following classes with **AutoConfig**

The **AutoConfig** class will automatically select correct `AraConfig` class based on `model_type` in `config.json` file.

- [AraLlamaConfig](./models/ara_llama_classes.md#arallamaconfig)
- [AraQwenConfig](./models/ara_qwen_classes.md#araqwenconfig)
- [AraLlavaConfig](./models/ara_llava_classes.md#arallavaconfig)
- [AraQwen2_5VLConfig](./models/ara_qwen2.5_classes.md#araqwen2_5vlconfig)
- [AraQwen2_5ImageConfig](./models/ara_qwen2.5_classes.md#araqwen2_5imageconfig)
- [AraWhisperConfig](./models/ara_whisper_classes.md)

For more details see [AutoConfig](./models/ara_auto_class.md#autoconfig)

## AraGenerationConfig

**AraGenerationConfig** class is imported from `optimum.ara`. It extends transformers `GenerationConfig` class to handle `Ara` specific config params.

To know more about APIs usage, refer [AraGenerationConfig](./base_classes/ara_generation_config.md).
