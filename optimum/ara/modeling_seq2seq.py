# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
import logging
from pathlib import Path
from typing import Optional, Union, cast, List, Dict, Any, Tuple
import torch
import numpy as np

from transformers import (
    AutoModelForSpeechSeq2Seq,
    Cache,
    PretrainedConfig,
    GenerationConfig,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
)
from transformers.generation import (
    GenerationMixin,
    LogitsProcessorList,
)
from transformers.generation.utils import GenerateEncoderDecoderOutput
from transformers.generation.streamers import BaseStreamer
import time

from .configuration_utils import AraPretrainedConfig
from .generation.configuration_utils import AraGenerationConfig, AraHostConfig
from .modeling_base import AraModel


class AraModelForConditionalGeneration(AraModel, GenerationMixin):
    """
    This class provides functionality for text generation using Ara models, including prompt processing
    and token generation. It handles both local and device-based model configuration parameter (MCP) processing.
    TODO:
        - Add complete conditional generation logic to support:
            * Beam search decoding
            * Diverse beam search
            * Contrastive search
            * Constrained beam search
            * Prefix-constrained beam search
        - Implement proper handling of generation strategies based on model configuration
        - Add support for different sampling methods in conditional generation
        - Implement proper attention masking for conditional generation
    Main functionalities:
        - Model loading and initialization
        - Text generation with various parameters
        - Performance monitoring and statistics
        - Device temperature monitoring
        - Configuration management
        - Token processing and logits processing

    AraModelForConditionalGeneration is a wrapper for conditional language modeling using Ara models.
    """

    stateful = False
    generation_config: AraGenerationConfig  # type: ignore[override]

    def __init__(
        self,
        model_path: Union[Path, str],
        config: "AraPretrainedConfig",
        generation_config: "AraGenerationConfig",
        preprocessors: Optional[list] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        """
        Initialize the AraModelForConditionalGeneration.

        Args:
            session: DVSession object for model execution.
            model: DVModel object representing the loaded model.
            endpoint: Endpoint for model inference.
            model_path (Union[Path, str]): Path to the model file.
            config (AraPretrainedConfig): Model configuration.
            preprocessors (Optional[list]): List of preprocessors.
            use_cache (Optional[bool]): Whether to use KV caching.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            model_path=model_path,
            config=config,
            generation_config=generation_config,
            preprocessors=preprocessors,
        )
        self.normalized_config = None
        self.num_pkv = 2
        self.use_cache = use_cache
        self.next_token_idx = 0

        self.dvm_path = model_path
        self.model_type = config.model_type

        self.key_value_input_names = []
        self.key_value_output_names = []

        self.use_merged = False
        self.use_fp16 = False

    def prepare_inputs_for_generation(
        self,
        input_ids: Union[torch.LongTensor, np.ndarray],
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepares input and output arrays for generation.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            generation_config (AraGenerationConfig): Generation configuration.

        Returns:
            Dict[str, Any]: Dictionary with input and output arrays.
        """
        if generation_config is None:
            raise ValueError("generation_config cannot be None")

        input_ids_numpy = (
            input_ids.numpy()
            if isinstance(input_ids, torch.Tensor)
            else np.array(input_ids)
        )

        vocab_size = self.core.get_vocab_size()
        num_max_tokens = self.core.get_max_num_tokens()
        pad_token_id = self.core.get_pad_token_id()
        is_speculative = self.core.is_speculative()

        if generation_config.ara.target_prompt_post_mcp == 1:
            # post processing on device
            if is_speculative:
                output_size = 4
            else:
                output_size = 1
        else:
            if is_speculative:
                output_size = 4 * vocab_size
            else:
                output_size = vocab_size

        output_array = np.zeros(output_size, dtype=np.int32)

        num_tokens = input_ids_numpy.size
        padded = np.full(num_max_tokens, pad_token_id, dtype=np.int32)
        padded[:num_tokens] = input_ids_numpy
        input_array = padded

        return {
            "input_array": input_array,
            "output_array": output_array,
        }

    def _encode(
        self,
        input_array: np.ndarray,
        input_features: torch.Tensor,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> Any:
        """Device/backend-specific prompt-processing entrypoint.

        Subclasses should implement encoding / prompt processing logic that
        prepares the model for token generation (e.g. perform prompt
        processing MCP or host-side processing). This method is invoked by the
        high-level `generate()` flow to produce the initial device state and
        first-token logits.

        The return value and exact signature are backend-specific but
        implementations typically accept (input_array, input_features,
        output_array, num_valid_tokens, num_active_tokens, generation_config,
        **kwargs) and return a dv_status_code indicating success or failure.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _decode(
        self,
        input_array: np.ndarray,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> Tuple[Any, float]:
        """Device/backend-specific token-generation entrypoint.

        Subclasses should implement token-generation logic that accepts the
        current input buffer and writes logits (or token ids) into
        `output_array`. The method should return (inf_req_obj, infer_time).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def get_generation_timing(self):
        """
        Get timing information from the last generation.

        Returns:
            dict: Dictionary containing ttft, total_time, and calculated token_rate
        """
        if not hasattr(self, "_ttft") or not hasattr(self, "_subsequent_start"):
            return None

        return {
            "ttft": self._ttft,
            "total_time": self._token_generation_time + self._ttft,
            "subsequent_time": self._token_generation_time,
        }

    @classmethod
    def from_config(cls, config: AraPretrainedConfig, **kwargs):
        """
        Instantiates the model from a model configuration object.

        Args:
            config (AraPretrainedConfig): Model configuration.
            **kwargs: Additional arguments.

        Returns:
            AraModelForConditionalGeneration: Loaded model instance.
        """
        pretrained_model_name_or_path = config.ara.dvm_path

        gen_cfg = AraGenerationConfig.from_model_config(config)

        return cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            generation_config=gen_cfg,
            **kwargs,
        )

    @classmethod
    def _from_config(cls, config: AraPretrainedConfig, **kwargs):
        """
        Required for AutoConfig Integration
        """
        return cls.from_config(config, **kwargs)

    def save_config(self, save_directory):
        """
        Saves the model configuration to the specified directory.

        Args:
            save_directory (str): Directory to save the configuration.
        """
        os.makedirs(save_directory, exist_ok=True)

        self.config.save_pretrained(save_directory)
        # TODO Add gen config saving option here as well


class AraModelForSpeechSeq2Seq(AraModelForConditionalGeneration):
    """Speech sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports whisper, speech_to_text."""

    main_input_name = "input_ids"
    auto_model_class = AutoModelForSpeechSeq2Seq

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Following a breaking change in transformers that relies directly on the mapping name and not on the
        # greedy model mapping (that can be extended), we need to hardcode the ortmodel in this dictionary.
        # Other pipelines do not seem to have controlflow depending on the mapping name.
        # See: https://github.com/huggingface/transformers/pull/24960/files
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES["ara_speechseq2seq"] = (
            self.__class__.__name__
        )

    def generate(  # pyrefly: ignore[bad-override]
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[
            Union[GenerationConfig, AraGenerationConfig]
        ] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Any] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional[Any] = None,
        streamer: Optional[BaseStreamer] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateEncoderDecoderOutput, torch.LongTensor, Any]:
        """
        Generates text sequences given input tokens.

        Args:
            inputs (Optional[torch.Tensor]): Input tensor.
            generation_config (Optional[Union[GenerationConfig, AraGenerationConfig]]): Generation configuration.
            logits_processor (Optional[LogitsProcessorList]): Custom logits processors.
            stopping_criteria (Optional[StoppingCriteriaList]): Custom stopping criteria.
            prefix_allowed_tokens_fn (Optional[Callable]): Function to constrain generation.
            synced_gpus (Optional[bool]): Whether to sync GPUs.
            assistant_model (Optional[Any]): Assistant model for speculative decoding.
            streamer (Optional[BaseStreamer]): Streamer for output tokens.
            negative_prompt_ids (Optional[torch.Tensor]): Negative prompt IDs.
            negative_prompt_attention_mask (Optional[torch.Tensor]): Negative prompt mask.
            **kwargs: Additional generation parameters (e.g., prompt_ids).

        Returns:
            Union[GenerateEncoderDecoderOutput, torch.LongTensor]: Generated token IDs.
        """
        self.fixed_config = kwargs.pop(
            "forced_decoder_ids", getattr(self, "fixed_config", None)
        )
        prompt_ids = kwargs.pop("prompt_ids", None)

        if prompt_ids is None:
            # create a tensor with one batch and zero tokens (empty prompt)
            prompt_ids = torch.LongTensor([[]])
        if inputs is None:
            raise ValueError("inputs cannot be None")

        if isinstance(inputs, torch.Tensor):
            numpy_inputs = inputs.detach().numpy().flatten()
        elif isinstance(inputs, list):
            numpy_inputs = np.array(inputs)
        else:
            numpy_inputs = inputs

        self._num_input_tokens = prompt_ids

        if numpy_inputs is None or numpy_inputs.size == 0:
            print("Error: input audio features are empty")
            # Return an empty tensor with shape (1, 0) to match the expected return type
            return torch.LongTensor([[]])

        # updates generation variables according to the generation config sent or kwargs
        generation_config, _ = self._prepare_generation_config(
            generation_config, **kwargs
        )
        mcp_dict = {
            "target_prompt_post_mcp": 1,
            "target_prompt_pre_mcp": 0,
            "target_token_post_mcp": 1,
            "target_token_pre_mcp": 1,
        }
        ara_cfg = generation_config.ara
        if (
            ara_cfg.target_prompt_post_mcp != mcp_dict["target_prompt_post_mcp"]
            or ara_cfg.target_prompt_pre_mcp != mcp_dict["target_token_pre_mcp"]
            or ara_cfg.target_token_post_mcp != mcp_dict["target_token_post_mcp"]
            or ara_cfg.target_token_pre_mcp != mcp_dict["target_token_pre_mcp"]
        ):
            logging.warning(f"Suggested mcp values for good results are {mcp_dict}")

        self.core.update_llm_params(generation_config)

        # llm_params_obj = self.core.dv_internal.model._model.llm_params.contents
        pad_token_id = self.core.get_pad_token_id()
        eos_token_id = self.core.get_eos_token_id()
        is_speculative = self.core.is_speculative()
        num_max_tokens = self.core.get_max_num_tokens()

        if prompt_ids.numel() >= num_max_tokens:
            print(
                f"input length is greater than or equal to max token length {num_max_tokens}, current input length: {prompt_ids.size}"
            )
            empty_tensor = torch.LongTensor([[]])
            if generation_config.return_dict_in_generate:
                return GenerateEncoderDecoderOutput(sequences=empty_tensor)
            return empty_tensor

        eos_token_ls = (
            generation_config.eos_token_id
            if isinstance(generation_config.eos_token_id, list)
            else [generation_config.eos_token_id]
        )
        eos_token_ls.append(eos_token_id)

        prepared_inputs = self.prepare_inputs_for_generation(
            prompt_ids, generation_config=generation_config
        )

        num_tokens = 0
        next_token_index = 0

        input_array = prepared_inputs["input_array"]
        output_array = prepared_inputs["output_array"]

        generated_list = []
        active_tokens = num_tokens
        num_valid_tokens = num_tokens

        infer_time = self._encode(
            input_array,
            inputs,
            output_array,
            active_tokens,
            num_valid_tokens,
            generation_config=generation_config,
        )

        self._ttft = infer_time
        self._token_generation_time = 0

        if (
            generation_config is not None
            and generation_config.ara.target_prompt_post_mcp
        ):
            valid_tokens = output_array[:1].copy().astype("int32")
        else:
            valid_tokens = np.argmax(output_array, axis=-1, keepdims=True).astype(
                "int32"
            )

        generated_list += valid_tokens[:1].flatten().tolist()

        num_valid_tokens = 1

        if streamer:
            streamer.put(valid_tokens)

        if generation_config is not None and generation_config.max_length > 0:
            user_max_length = generation_config.max_length
            if user_max_length > num_max_tokens:
                user_max_length = num_max_tokens
        else:
            user_max_length = num_max_tokens

        while len(generated_list) >= user_max_length or not any(
            token in eos_token_ls for token in valid_tokens
        ):
            if is_speculative:
                input_array = np.pad(
                    valid_tokens,
                    (0, num_max_tokens - len(valid_tokens)),
                    "constant",
                    constant_values=pad_token_id,
                )
            else:
                input_array = valid_tokens

            inf_req, infer_time = self._decode(
                input_array=input_array,
                output_array=output_array,
                num_valid_tokens=num_valid_tokens,
                num_active_tokens=1,
                generation_config=generation_config,
            )

            self._token_generation_time += infer_time

            assert inf_req is not None

            num_valid_tokens = (
                inf_req.contents.llm_infer_info.contents.llm_infer_resp_num_valid_tokens
            )

            # Check if we got any valid tokens
            if num_valid_tokens == 0:
                # No more tokens generated, break the loop
                print("Error : No more valid token generated!")
                break

            if generation_config.ara.target_token_post_mcp:
                valid_tokens = output_array[:num_valid_tokens]
            else:
                valid_tokens = np.argmax(output_array, axis=-1, keepdims=True)

            next_token_index += len(valid_tokens)
            if streamer is not None:
                streamer.put(valid_tokens)

            for token in valid_tokens:
                if token in eos_token_ls:
                    break
                generated_list.append(token)

        if streamer is not None:
            streamer.end()
        self._generated_tokens_count = len(generated_list)

        generated_list_torch = torch.LongTensor([generated_list])

        if generation_config.return_dict_in_generate:
            return GenerateEncoderDecoderOutput(sequences=generated_list_torch)
        return generated_list_torch

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        config: Optional[Union[PretrainedConfig, AraPretrainedConfig]] = None,
        *args,
        **kwargs,
    ) -> "AraModelForSpeechSeq2Seq":
        """
        Loads a pretrained AraModelForSpeechSeq2Seq from disk.

        Args:
            pretrained_model_name_or_path (Union[str, Path, os.PathLike]): Model identifier or path containig config.json and generation_config.json.
            config (Optional[Union[PretrainedConfig, AraPretrainedConfig]]): Model configuration.
            *args: forwarded to parent class.
            **kwargs: Additional arguments.

        Returns:
            AraModelForSpeechSeq2Seq: Loaded model instance.
        """
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        use_cache = kwargs.pop("use_cache", True)
        file_name = kwargs.pop("file_name", None)

        config_result = cls._get_config(
            pretrained_model_name_or_path, cast(Optional[AraPretrainedConfig], config)
        )
        if not isinstance(config_result, AraPretrainedConfig):
            raise TypeError(f"Expected AraPretrainedConfig, got {type(config_result)}")
        config_ara = config_result

        dvm_file = cls._get_llm_dvm_path(
            pretrained_model_name_or_path, config_ara, file_name
        )

        # Load generation config use parent class method
        gen_config = cls._get_generation_config(
            pretrained_model_name_or_path,
            config_ara,
            kwargs.pop("generation_config", None),
        )

        init_cls = cls(
            model_path=dvm_file,
            config=config_ara,
            generation_config=gen_config,
            use_cache=use_cache,
            **kwargs,
        )

        # Load model
        init_cls.load_model(dvm_file, device_map, max_memory)

        return init_cls
