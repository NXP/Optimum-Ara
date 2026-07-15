# Optimum Ara APIs

Optimum-Ara is an extenstion of HuggingFace Transformers library with support for Ara Accelarator to enable High Performance AI at Edge.


Below is the list of Enabled Classes: 

Task Based Classes:
- [Optimum Ara APIs](#optimum-ara-apis)
  - [AutoModelForCausalLM](#automodelforcausallm)
  - [AutoModelForImageTextToText](#automodelforimagetexttotext)
  - [AutoModelForVision2Seq](#automodelforvision2seq)
  - [AutoConfig](#autoconfig)
  - [AraGenerationConfig](#aragenerationconfig)

Other Classes:
- [AutoConfig](#autoconfig)
- [AraGenerationConfig](#aragenerationconfig)


## AutoModelForCausalLM

**AutoModelForCausalLM** class is imported from `transformers`. 
when importing `optimum.ara` we will register following classes with **AutoModelForCausalLM**

The **AutoModelForCausalLM** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraLlamaForCausalLM](./models/ara_llama_classes.md)
- [AraQwenForCausalLM](./models/ara_qwen_classes.md)

more details on using **AutoModelForCausalLM** to load **Ara** classes [click here]()

## AutoModelForImageTextToText

**AutoModelForImageTextToText** class is imported from `transformers`. 
when importing `optimum.ara` we will register following classes with **AutoModelForImageTextToText**

The **AutoModelForImageTextToText** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraQwen2_5_VLForConditionalGeneration](./models/ara_qwen2.5_classes.md)
- [AraQwen2_5_ImageForConditionalGeneration](./models/ara_qwen2.5_classes.md)

more details on using **AutoModelForImageTextToText** to load **Ara** classes [click here]()

## AutoModelForVision2Seq

**AutoModelForVision2Seq** class is imported from `transformers`. 
when importing `optimum.ara` we will register following classes with **AutoModelForVision2Seq**

The **AutoModelForVision2Seq** class will automatically select correct `AraModel` class based on `model_type` in `config.json` file.

- [AraLlavaForConditionalGeneration]()
- [AraQwen2_5_VLForConditionalGeneration](./models/ara_qwen2.5_classes.md)

more details on using **AutoModelForVision2Seq** to load **Ara** classes [click here]()

## AutoConfig

**AutoConfig** class is imported from `transformers`. 
when importing `optimum.ara` we will register following classes with **AutoConfig**

The **AutoConfig** class will automatically select correct `AraConfig` class based on `model_type` in `config.json` file.

- [AraLlamaConfig](./models/ara_llama_classes.md#arallamaconfig)
- [AraQwenConfig](./models/ara_qwen_classes.md#araqwenconfig)
- [AraLlavaConfig]()
- [AraQwen2_5VLConfig]()
- [AraQwen2_5ImageConfig]()

more details on using **AutoConfig** to load **Ara** classes [click here]()

## AraGenerationConfig

**AraGenerationConfig** class is imported from `optimum.ara`. It extends transformers `GenerationConfig` class to handle `Ara` specific config params.

To know more about APIs usage, refer [AraGenerationConfig](./base_classes/ara_generation_config.md).
