# Whisper Model Classes

Whisper is a series of large language models (LLMs) based on encoder–decoder architecture. Whisper is designed for efficient inference, using an encoder–decoder architecture that separates audio processing from text generation.

We have added two Whisper specific classes.
- [AraWhisperConfig](#arawhisperconfig)
- [AraModelForWhisper](#aramodelforwhisper)

## AraWhisperConfig

**AraWhisperConfig** is a configuration class that extends **AraPretrainedConfig**.

```json
"model_type": "ara_whisper"
```

To know more about the AraPretrainedConfig class, refer [AraPretrainedConfig](../base_classes/ara_pretrained_config.md).

### Example:

```python
from optimum.ara import AraWhisperConfig

configuration = AraWhisperConfig.from_pretrained("./models/whisper/")
```
## AraModelForWhisper

**AraModelForWhisper** class extends **AraModelForSpeechSeq2Seq**.

Client APIs are same a parent class.
To know more about **AraModelForSpeechSeq2Seq**, refer [AraModelForSpeechSeq2Seq](../base_classes/ara_model_for_seq2seq.md). 

### Example:

```python
from optimum.ara import AraWhisperConfig

configuration = AraModelForWhisper.from_pretrained("./models/whisper/")
```