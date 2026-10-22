# Base Classes

Base classes serve as the foundational building blocks in `optimum-ara`, providing shared functionality and standardized configurations across different model types.

## Configuration Classes

### [AraGenerationConfig](ara_generation_config.md)
Dedicated configuration classes designed to handle `ara`-specific parameters within configuration files. They ensure consistent hyperparameter management during text generation.

### [AraPretrainedConfig](ara_pretrained_config.md)
Dedicated configuration classes designed to handle `ara`-specific parameters within configuration files. They ensure consistent hyperparameter management during model loading.

---

## Task Based Classes

### [AraModelForCausalLM](ara_model_for_causallm.md)
The foundational base class for all text-based models. It provides the core functionalities required to load, initialize, and run causal language models. This class is extended by specific architectures, including:
* LLaMa
* Qwen

### [AraModelForVisualLanguage](ara_model_for_visual_language.md)
The core base class tailored for multimodal vision models. It handles the underlying logic for processing both **image** and **video** inputs. This class is extended by specific architectures, including:
* LLaVa
* Qwen2.5-VL

### [AraModelForSpeechSeq2Seq](ara_model_for_seq2seq.md)
The dedicated base class for audio and speech models, managing the sequence-to-sequence workflows required for speech processing and recognition tasks. This class is extended by specific architectures, including:
* Whisper