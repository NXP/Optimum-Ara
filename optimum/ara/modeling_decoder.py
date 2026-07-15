# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
import logging
from pathlib import Path
from typing import Optional, Union, Any, List, Dict, cast, Tuple, TYPE_CHECKING
import torch
import numpy as np

# import ctypes as c
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    GenerationConfig,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerationMixin
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast
import time
import logging

from .configuration_utils import AraPretrainedConfig, AraConfig
from .generation.configuration_utils import AraGenerationConfig, AraHostConfig
from .generation import CustomLogitsProcessor
from .modeling_base import AraModel
from .generation.logits_process import LogitsDequantizer, SampleLogitsProcessor
from .utils.constants import DEFAULT_GENERATION_CONFIG_NAME
from .utils.file_utils import find_files_matching_pattern
from .utils.utils import get_logits_processor, get_stopping_criteria


class AraModelForCausalLM(AraModel, GenerationMixin):
    """
    AraModelForCausalLM is a wrapper for causal language modeling using Ara models.
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"
    stateful = False
    pad_token_id = 0
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
        Initialize the AraModelForCausalLM.

        Args:
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

        self.input_array_cached: np.ndarray = np.array([], dtype=np.int32)

    def prepare_inputs_for_generation(
        self,
        input_ids: Union[torch.LongTensor, np.ndarray],
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[
            Union[torch.FloatTensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
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
        eos_token_id = self.core.get_eos_token_id()
        is_speculative = self.core.is_speculative()
        is_host_specd = self.core.is_host_specd()
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

        if is_host_specd == 1:
            output_size = 4 * vocab_size

        logging.debug(f"output_size : {output_size}")

        output_array = np.zeros(output_size, dtype=np.int32)

        num_tokens = input_ids_numpy.size
        padded = np.full(num_max_tokens, pad_token_id, dtype=np.int32)
        padded[:num_tokens] = input_ids_numpy
        input_array = padded

        return {
            "input_array": input_array,
            "output_array": output_array,
        }

    def forward(
        self,
        input_ids: Optional[np.ndarray] = None,
        input_embeds: Optional[np.ndarray] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        input_array = input_ids if input_ids is not None else input_embeds
        if input_array is None:
            raise ValueError("Either input_ids or input_embeds must be provided")
        if isinstance(input_array, torch.Tensor):
            input_array = input_array.numpy()
        if input_array.ndim == 2:
            input_array = input_array.squeeze(0)

        # deciding if its first token generation or next token generation
        # if input_array is shorter than our cache, then it will be a new prompt
        # if the diff btw cache and input array is greater than 4, then new prompt
        # else if cache completely hits then, its old propmt else its new prompt
        last_non_pad = np.where(input_array != self.core.get_pad_token_id())[0][-1]
        actual_prompt = input_array[: last_non_pad + 1]
        if (
            len(self.input_array_cached) > len(actual_prompt)
            or len(actual_prompt) - len(self.input_array_cached) > 4
        ):
            first_token_gen = True
        else:
            possible_cache = actual_prompt[: len(self.input_array_cached)]
            first_token_gen = (
                False
                if np.array_equal(possible_cache, self.input_array_cached)
                else True
            )

        if first_token_gen:
            infer_type = self.core.infer_type.DV_INFER_TYPE_LLM_PROMPT_PROCESSING
            valid_tokens = len(actual_prompt)
            active_tokens = len(actual_prompt)
        else:
            infer_type = self.core.infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION
            valid_tokens = len(actual_prompt) - len(self.input_array_cached)
            active_tokens = len(self.input_array_cached)
            input_array = actual_prompt[len(self.input_array_cached) :]

        self.input_array_cached = actual_prompt

        ara_cfg = self.generation_config.ara
        if ara_cfg.target_prompt_pre_mcp == 0 or ara_cfg.target_token_pre_mcp == 0:
            if self.embedding is None or self.embedding_scales is None:
                raise ValueError(
                    "embedding & embedding_scales are required for host pre-processing"
                )
            input_ids_torch = torch.as_tensor(
                input_array.view(np.int32), dtype=torch.int32
            )
            output = self.embedding(input_ids_torch)
            output_scale = self.embedding_scales(input_ids_torch)
            input_array = np.concatenate(
                (
                    input_array.view(np.int8),
                    output.detach().flatten().numpy(),
                    output_scale.detach().flatten().numpy(),
                ),
                axis=0,
            )

        # setting output array here
        vocab_size = self.core.get_vocab_size()
        is_speculative = self.core.is_speculative()
        is_host_specd = self.core.is_host_specd()
        if ara_cfg.target_prompt_post_mcp == 1:
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

        if is_host_specd == 1:
            output_size = 4 * vocab_size

        logging.debug(f"output_size : {output_size}")

        output_array = np.zeros(output_size, dtype=np.int32)

        _, _, _ = self.core.forward(
            input_array,
            output_array,
            infer_type=infer_type,
            active_tokens=active_tokens,
            valid_tokens=valid_tokens,
            tokens_to_skip=0,
        )

        if ara_cfg.target_token_post_mcp or ara_cfg.target_prompt_post_mcp:
            logits = torch.full((vocab_size,), -100.0)
            logits[output_array[0]] = 0.0
            output_array = logits
        else:
            dequantizer = LogitsDequantizer()
            out_torch = cast(torch.FloatTensor, torch.tensor(output_array).float())
            empty_logits = cast(torch.LongTensor, torch.empty(0, dtype=torch.long))
            output_array = dequantizer(empty_logits, out_torch)

        ### defining logits for the entire prompt
        final_logits = torch.full(
            (1, len(actual_prompt), self.core.get_vocab_size()), fill_value=-100.0
        )
        for i, token_id in enumerate(actual_prompt):
            final_logits[0, i, token_id] = 0

        output_array = output_array.unsqueeze(0).unsqueeze(0)
        final_logits = torch.cat([final_logits, output_array], dim=1)
        return CausalLMOutputWithPast(logits=cast(torch.FloatTensor, final_logits))

    def _generate_first_token(
        self,
        input_array: np.ndarray,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ):
        """
        Generates the first token using prompt processing.

        Args:
            input_array (np.ndarray): Input array.
            output_array (np.ndarray): Output array.
            num_tokens (int): Number of tokens in the prompt.

        Returns:
            dv_status_code: Status code of the inference.
        """

        skip_tokens = kwargs.pop("tokens_to_skip", 0)
        if generation_config is None:
            raise ValueError(
                "generation_config cannot be None in _generate_first_token"
            )
        self.logits_processor = get_logits_processor(generation_config)
        if (
            not generation_config.ara.target_prompt_pre_mcp
        ):  # mcp = 0 means processing on host.
            input_array_block = input_array.view(np.int32)

            if self.embedding is None or self.embedding_scales is None:
                raise ValueError(
                    "Both embedding and embedding_scales are required for host post-processing"
                )

            _device = cast(
                Optional[Union[torch.device, str, int]], self.embedding.weight.device
            )
            input_ids_torch = torch.as_tensor(
                input_array_block, dtype=torch.int32, device=_device
            )
            output = self.embedding(input_ids_torch)
            output_scale = self.embedding_scales(input_ids_torch)

            input_array = np.concatenate(
                (
                    input_array.view(np.int8),
                    output.detach().flatten().numpy(),
                    output_scale.detach().flatten().numpy(),
                ),
                axis=0,
            )

        status, _, _ = self.core.forward(
            input_array,
            output_array,
            infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_PROMPT_PROCESSING,
            active_tokens=num_active_tokens,
            valid_tokens=num_valid_tokens,
            tokens_to_skip=skip_tokens,
        )

    def _generate_next_token(
        self,
        input_array: np.ndarray,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[AraGenerationConfig] = None,
    ) -> Tuple[Any, Any, float]:
        """
        Generates the next token(s) using token generation inference.

        Args:
            input_array (np.ndarray): Input array.
            output_array (np.ndarray): Output array.
            num_valid_tokens (int): Number of valid tokens.
            active_tokens (int): Index for the next token.

        Returns:
            Tuple of status code and inference request object.
        """
        if generation_config is None:
            raise ValueError("generation_config cannot be None")
        if generation_config.ara.target_token_pre_mcp == 0:
            if self.embedding is None or self.embedding_scales is None:
                raise ValueError(
                    "Both embedding and embedding_scales are required for host post-processing"
                )
            _device = cast(
                Optional[Union[torch.device, str, int]], self.embedding.weight.device
            )
            input_ids_torch = torch.as_tensor(
                input_array, dtype=torch.long, device=_device
            )
            # print(input_ids_torch[:10])
            output = self.embedding(input_ids_torch)
            output_scale = self.embedding_scales(input_ids_torch)
            input_array = np.concatenate(
                (
                    input_array,
                    output.detach().flatten().numpy().astype(np.int8),
                    output_scale.detach().flatten().numpy().astype(np.int8),
                ),
                axis=0,
            )

        status, inf_req_obj, infer_time = self.core.forward(
            input_array,
            output_array,
            infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION,
            active_tokens=num_active_tokens,
            valid_tokens=num_valid_tokens,
            tokens_to_skip=0,
        )

        return status, inf_req_obj, infer_time

    def _extract_next_tokens(
        self,
        output_array: np.ndarray,
        input_ids: np.ndarray,
        generation_config: "AraGenerationConfig",
        num_valid_tokens: Optional[int] = 1,
        is_first_token: bool = False,
    ) -> np.ndarray:
        """
        Extract the next tokens depending on MCP mode.

        - For first token: If target_prompt_post_mcp = 0 → device returned logits → apply local processors.
        - For first token: If target_prompt_post_mcp = 1 → device returned token IDs → read from output_array.
        - For next tokens: If target_token_post_mcp = 0 → device returned logits → apply local processors.
        - For next tokens: If target_token_post_mcp = 1 → device returned token IDs → read from output_array.

        Args:
            output_array (np.ndarray): Raw device output buffer.
            input_ids (np.ndarray): Current input IDs (for processors).
            generation_config (AraGenerationConfig): Current gen config.
            num_valid_tokens (int): Number of valid tokens when MCP=1.
            is_first_token (bool): Whether this is the first token generation.

        Returns:
            np.ndarray: Next tokens as int32 (shape: [num_tokens]).
        """
        vocab_size = self.core.get_vocab_size()
        if is_first_token:
            # For first token, check target_prompt_post_mcp
            if generation_config.ara.target_prompt_post_mcp == 0:
                # Case: device returned logits
                raw_logits = output_array[:vocab_size].copy()
                processed_logits = self._apply_logit_processing(
                    input_ids, raw_logits.astype(np.float32)
                )
                next_tokens = np.argmax(processed_logits, axis=-1, keepdims=True)
                return next_tokens
            else:
                # Case: device returned tokens
                return output_array[:num_valid_tokens].copy()
        else:
            # For next tokens, check target_token_post_mcp
            if generation_config.ara.target_token_post_mcp == 0:
                # Case: device returned logits
                raw_logits = output_array[:vocab_size].copy()
                processed_logits = self._apply_logit_processing(
                    input_ids, raw_logits.astype(np.float32)
                )
                next_tokens = np.argmax(processed_logits, axis=-1, keepdims=True)
                return next_tokens
            else:
                # Case: device returned tokens
                return output_array[:num_valid_tokens].copy()

    def _ensure_generation_operators(
        self, generation_config: Optional[AraGenerationConfig] = None
    ) -> None:
        """
        Ensure self.logits_processor and self.stopping_criteria are set up
        according to the (possibly updated) generation configuration.
        """
        if generation_config is None:
            raise ValueError("generation_config cannot be None")
        gc = generation_config
        # These two helpers mirror HF's design: a list of score processors and optional stopping rules.
        self.logits_processor = get_logits_processor(gc)
        self.stopping_criteria = get_stopping_criteria(gc)

    def _apply_logit_processing(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        raw_logits: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the configured logits processors to raw logits (shape: [vocab_size])
        using the provided input_ids (shape: [1, seq_len]). Returns a numpy array
        with the same shape as raw_logits.

        This centralizes 'local' logit processing so both LLMs and VLMs can use it
        when mcp=0 (i.e., the device returns raw logits and we must process locally).
        """
        # If no processors are configured, return raw logits as-is.
        if not hasattr(self, "logits_processor") or self.logits_processor is None:
            return raw_logits

        # Normalize input_ids to a torch.LongTensor of shape [1, seq_len]
        if isinstance(input_ids, np.ndarray):
            input_ids_torch = cast(torch.LongTensor, torch.from_numpy(input_ids).long())
        else:
            input_ids_torch = cast(torch.LongTensor, input_ids.long())
        if input_ids_torch.dim() == 1:
            input_ids_torch = cast(torch.LongTensor, input_ids_torch.unsqueeze(0))

        # Normalize logits to a torch.FloatTensor of shape [1, vocab_size]
        # (HF LogitsProcessorList expects (batch, vocab))
        logits_torch = cast(torch.FloatTensor, torch.from_numpy(raw_logits).float())
        if logits_torch.dim() == 1:
            logits_torch = cast(torch.FloatTensor, logits_torch.unsqueeze(0))

        # Apply the processors (e.g., repetition penalty, temperature/top-k/top-p, etc.)
        processed = self.logits_processor(input_ids_torch, logits_torch)

        # Return back to numpy 1D (vocab_size)
        return processed.squeeze(0).detach().cpu().numpy().astype(np.int32)

    def _handle_host_specd(
        self,
        inputs: Optional[np.ndarray] = None,
        generation_config: Optional[AraGenerationConfig] = None,
        streamer: Optional["BaseStreamer"] = None,
        custom_logits_processor: Optional[CustomLogitsProcessor] = None,
        **kwargs,
    ):
        pad_token_id = self.core.get_pad_token_id()
        eos_token_id = self.core.get_eos_token_id()
        num_max_tokens = self.core.get_max_num_tokens()
        is_speculative = self.core.is_speculative()
        vocab_size = self.core.get_vocab_size()

        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        if generation_config is None:
            generation_config = AraGenerationConfig.from_model_config(self.config)
            logging.info("Loading default Ara generation config object")

        # updates generation variables according to the generation config sent or kwargs
        generation_config, _ = self._prepare_generation_config(
            generation_config, **kwargs
        )

        numpy_inputs = inputs
        if numpy_inputs is None:
            raise ValueError("inputs cannot be None")
        if generation_config is None:
            raise ValueError("generation_config cannot be None")
        host_input_ids: List[int] = numpy_inputs.tolist()
        eos_token_ls = (
            generation_config.eos_token_id
            if isinstance(generation_config.eos_token_id, list)
            else [generation_config.eos_token_id]
        )
        eos_token_ls.append(eos_token_id)

        prepared_inputs = self.prepare_inputs_for_generation(
            cast(torch.LongTensor, numpy_inputs), generation_config=generation_config
        )
        input_array = prepared_inputs["input_array"]
        output_array = prepared_inputs["output_array"]

        num_tokens = numpy_inputs.size
        generated_list = []
        active_tokens = num_tokens
        num_valid_tokens = num_tokens
        self._token_generation_time = 0
        self._generated_tokens_count = 0
        self._ttft = 0

        ttft_start = time.time()
        self._generate_first_token(
            input_array,
            output_array,
            active_tokens,
            num_valid_tokens,
            generation_config,
        )
        ttft = time.time() - ttft_start

        # Store TTFT for later access
        self._ttft = ttft

        if generation_config.ara.target_prompt_post_mcp:
            logging.info(
                "This model.dvm requires post processing on Host, ignoring target_prompt_post_mcp coming from generation config"
            )

        if custom_logits_processor:
            selected_tokens, candidate_tokens = custom_logits_processor(
                cast(
                    torch.LongTensor,
                    torch.as_tensor([host_input_ids], dtype=torch.long),
                ),
                output_array.reshape(4, vocab_size)[:1],
            )
        else:
            candidate_tokens = np.array([])
            selected_tokens = np.argmax(
                output_array.reshape(4, vocab_size)[:1], axis=-1
            )

        generated_list += selected_tokens.flatten().tolist()
        host_input_ids.extend(cast(List[int], selected_tokens.flatten().tolist()))
        if streamer:
            streamer.put(selected_tokens)

        num_valid_tokens = candidate_tokens.size + 1
        speculative_token_padding = 4 if is_speculative else 0

        if generation_config.max_length > 0:
            user_max_length = (
                generation_config.max_length
                + numpy_inputs.size
                + speculative_token_padding
            )
            if user_max_length > num_max_tokens:
                user_max_length = num_max_tokens
        else:
            user_max_length = num_max_tokens

        while True:
            if len(
                generated_list
            ) + numpy_inputs.size + speculative_token_padding >= user_max_length or any(
                token in eos_token_ls for token in selected_tokens
            ):
                break

            input_array = np.full(4, pad_token_id)
            input_array[0] = selected_tokens[-1]
            for i in range(0, min(candidate_tokens.size, input_array.size - 1)):
                input_array[i + 1] = candidate_tokens[i]

            num_active_tokens = (numpy_inputs.size + len(generated_list)) - 1
            num_valid_tokens = 1 + candidate_tokens.size

            input_array = input_array.astype(np.int32)

            _, _, infer_time = self._generate_next_token(
                input_array=input_array,
                output_array=output_array,
                num_valid_tokens=num_valid_tokens,
                num_active_tokens=num_active_tokens,
                generation_config=generation_config,
            )
            self._token_generation_time += infer_time

            if custom_logits_processor:
                selected_tokens, candidate_tokens = custom_logits_processor(
                    cast(
                        torch.LongTensor,
                        torch.as_tensor([host_input_ids], dtype=torch.long),
                    ),
                    output_array.reshape(4, vocab_size),
                )
            else:
                candidate_tokens = np.array([])
                selected_tokens = np.argmax(
                    output_array.reshape(4, vocab_size)[:1], axis=-1
                )

            if streamer:
                streamer.put(selected_tokens)

            for token in selected_tokens:
                if token in eos_token_ls:
                    break
                generated_list.append(token)
                host_input_ids.extend(cast(List[int], token.flatten().tolist()))

        if streamer:
            streamer.end()  # plus one for EOS token
        self._generated_tokens_count = len(generated_list) + 1

        generated_list = torch.LongTensor([generated_list])

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=generated_list)
        return generated_list

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
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor, Any]:
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
            **kwargs: Additional generation parameters (e.g., input_ids, custom_logits_processor).

        Returns:
            Union[GenerateDecoderOnlyOutput, torch.LongTensor]: Generated token IDs.
        """
        self._ttft = 0
        self._token_generation_time = 0
        self._generated_tokens_count = 0

        input_ids = kwargs.pop("input_ids", None)
        custom_logits_processor = kwargs.pop("custom_logits_processor", None)

        if input_ids is not None:
            inputs = cast(torch.Tensor, input_ids)

        if isinstance(inputs, torch.Tensor):
            numpy_inputs = inputs.detach().numpy()
        elif isinstance(inputs, list):
            numpy_inputs = np.array(inputs)
        else:
            numpy_inputs = inputs

        self._num_input_tokens = (
            torch.from_numpy(numpy_inputs)
            if isinstance(numpy_inputs, np.ndarray)
            else numpy_inputs
        )

        if numpy_inputs is None or numpy_inputs.size == 0:
            logging.error("input is empty")
            return torch.LongTensor([[]])

        # updates generation variables according to the generation config sent or kwargs
        generation_config, _ = self._prepare_generation_config(
            generation_config, **kwargs
        )

        self.logits_processor = (
            logits_processor
            if logits_processor is not None
            else get_logits_processor(generation_config)
        )

        self.core.update_llm_params(generation_config)

        if self.embedding is None and self.embedding_scales is None:
            ara_cfg = generation_config
            if (
                not ara_cfg.ara.target_prompt_pre_mcp
                and not ara_cfg.ara.target_token_pre_mcp
            ):
                self._load_embedding_table(self.dvm_path)

        # print(self.core.dv_internal.model._model.llm_params.contents)
        pad_token_id = self.core.get_pad_token_id()
        eos_token_id = self.core.get_eos_token_id()
        is_speculative = self.core.is_speculative()
        num_max_tokens = self.core.get_max_num_tokens()
        vocab_size = self.core.get_vocab_size()

        # Backward Compatible with releases before r1.3.1
        is_host_specd = self.core.is_host_specd()
        # if hasattr(self.core.dv_internal.model._model.llm_params.contents, 'is_host_specd'):
        #     is_host_specd = self.core.dv_internal.model._model.llm_params.contents.is_host_specd
        # else:
        #     is_host_specd = False  # or some default value
        #     logging.warning("'is_host_specd' attribute not found in dv_model_llm_params")

        if numpy_inputs.size >= num_max_tokens:
            logging.error(
                f"input length is greater than or equal to max token length {num_max_tokens}"
            )
            return torch.LongTensor([[]])

        if is_host_specd:
            logging.info("'is host specd' is True")
            result = self._handle_host_specd(
                inputs=numpy_inputs,
                generation_config=generation_config,
                streamer=streamer,
                custom_logits_processor=custom_logits_processor,
                logits_processor=logits_processor,
                kwargs=kwargs,
            )
            return result

        eos_token_ls = (
            generation_config.eos_token_id
            if isinstance(generation_config.eos_token_id, list)
            else [generation_config.eos_token_id]
        )
        eos_token_ls.append(eos_token_id)

        prepared_inputs = self.prepare_inputs_for_generation(
            numpy_inputs, generation_config=generation_config
        )

        num_tokens = numpy_inputs.size
        next_token_index = num_tokens

        input_array = prepared_inputs["input_array"]
        output_array = prepared_inputs["output_array"]

        generated_list = []
        active_tokens = num_tokens
        num_valid_tokens = num_tokens

        # Measure Time to First Token (TTFT)
        ttft_start = time.time()
        self._generate_first_token(
            input_array,
            output_array,
            active_tokens,
            num_valid_tokens,
            generation_config,
        )
        ttft = time.time() - ttft_start

        # Store TTFT for later access
        self._ttft = ttft
        self._token_generation_time = 0

        if generation_config.ara.target_prompt_post_mcp:
            valid_tokens = output_array[:1].copy().astype("int32")
        else:
            if generation_config.do_sample:
                # Apply MCP on the output logits
                logits_torch = (
                    torch.from_numpy(output_array[:vocab_size]).float().unsqueeze(0)
                )
                input_ids_torch = torch.tensor(self._num_input_tokens).long()
                if input_ids_torch.shape[0] != logits_torch.shape[0]:
                    input_ids_torch = input_ids_torch.unsqueeze(0)
                # Apply processors
                self.logits_processor.insert(0, LogitsDequantizer())
                self.logits_processor.append(SampleLogitsProcessor())
                processed_logits = self.logits_processor(
                    cast(torch.LongTensor, input_ids_torch),
                    cast(torch.FloatTensor, logits_torch),
                )
                valid_tokens = processed_logits.detach().cpu().numpy().flatten()
            else:
                valid_tokens = np.argmax(output_array, axis=-1, keepdims=True).astype(
                    "int32"
                )
        generated_list += valid_tokens[:1].flatten().tolist()

        next_token_index += 1
        # First Inference always generate only 1 valid token
        num_valid_tokens = 1

        if streamer:
            streamer.put(valid_tokens)

        speculative_token_padding = 4 if is_speculative else 0

        if generation_config.max_length > 0:
            user_max_length = (
                generation_config.max_length
                + numpy_inputs.size
                + speculative_token_padding
            )
            if user_max_length > num_max_tokens:
                user_max_length = num_max_tokens
        else:
            user_max_length = num_max_tokens

        while True:
            if len(
                generated_list
            ) + numpy_inputs.size + speculative_token_padding >= user_max_length or any(
                token in eos_token_ls for token in valid_tokens
            ):
                break

            if is_speculative:
                input_array = np.pad(
                    valid_tokens,
                    (0, 4 - len(valid_tokens)),
                    "constant",
                    constant_values=pad_token_id,
                )
            else:
                input_array = valid_tokens

            status, inf_req, infer_time = self._generate_next_token(
                input_array=input_array,
                output_array=output_array,
                num_valid_tokens=num_valid_tokens,
                num_active_tokens=next_token_index - num_valid_tokens,
                generation_config=generation_config,
            )
            self._token_generation_time += infer_time

            # assert inf_req is not None

            num_valid_tokens = (
                inf_req.contents.llm_infer_info.contents.llm_infer_resp_num_valid_tokens
            )

            # Check if we got any valid tokens
            if num_valid_tokens == 0:
                # No more tokens generated, break the loop
                logging.error("No more valid token generated!")
                break

            if generation_config.ara.target_token_post_mcp:
                valid_tokens = output_array[:num_valid_tokens]
            else:
                if generation_config.do_sample:
                    # Apply MCP on the output logits
                    logits_torch = (
                        torch.from_numpy(output_array[:vocab_size]).float().unsqueeze(0)
                    )
                    input_ids_torch = torch.tensor(self._num_input_tokens).long()
                    if input_ids_torch.shape[0] != logits_torch.shape[0]:
                        input_ids_torch = input_ids_torch.unsqueeze(0)
                    # Apply processors
                    processed_logits = self.logits_processor(
                        cast(torch.LongTensor, input_ids_torch),
                        cast(torch.FloatTensor, logits_torch),
                    )
                    valid_tokens = processed_logits.detach().cpu().numpy().flatten()
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
        # plus one for EOS token
        self._generated_tokens_count = len(generated_list) + 1

        # currently, since only 1 batch size is supported
        # so this is resized to give a shape of (batch_size, num_samples)
        generated_list_torch = torch.LongTensor(generated_list)
        torch_inputs = torch.from_numpy(numpy_inputs)
        torch_inputs = torch_inputs.squeeze(0)
        output_torch = torch.cat([torch_inputs, generated_list_torch], dim=0).unsqueeze(
            0
        )
        output_torch = cast(torch.LongTensor, output_torch)
        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=output_torch)
        return output_torch

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
    def from_pretrained(  # pyrefly: ignore=bad-override
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        config: Optional[AraPretrainedConfig] = None,
        *args,
        **kwargs,
    ) -> "AraModelForCausalLM":
        """
        Loads a pretrained AraModelForCausalLM from disk.

        Args:
            pretrained_model_name_or_path (Union[str, Path, os.PathLike]): Model identifier or path containig config.json and generation_config.json.
            config (Optional[Union[PretrainedConfig, AraPretrainedConfig]]): Model configuration.

            *args: forwarded to parent class.
            **kwargs: Additional arguments (e.g., use_cache, file_name).

        Returns:
            AraModelForCausalLM: Loaded model instance.
        """

        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        config = cls._get_config(  # pyrefly: ignore=bad-assignment
            pretrained_model_name_or_path, config
        )
        assert isinstance(config, AraPretrainedConfig), "Invalid configuration type"
        gen_config = cls._get_generation_config(
            pretrained_model_name_or_path, config, kwargs.pop("generation_config", None)
        )

        file_name = kwargs.pop("file_name", None)
        use_cache = kwargs.pop("use_cache", True)
        # Get LLM DVM path using parent class method
        llm_path = cls._get_llm_dvm_path(
            pretrained_model_name_or_path,
            config,
            file_name,
        )

        # Create LLM instance
        init_cls = cls(
            model_path=llm_path,
            config=config,
            use_cache=use_cache,
            generation_config=gen_config,
            **kwargs,
        )

        # Load LLM models
        init_cls.load_model(llm_path, device_map, max_memory)
        ara_cfg = init_cls.generation_config.ara
        if not ara_cfg.target_prompt_pre_mcp and not ara_cfg.target_token_pre_mcp:
            init_cls._load_embedding_table(init_cls.dvm_path)

        return init_cls

    @classmethod
    def from_config(cls, config: AraPretrainedConfig, **kwargs):
        """
        Instantiates the model from a model configuration object.

        Args:
            config (AraPretrainedConfig): Model configuration.
            **kwargs: Additional arguments.

        Returns:
            AraModelForCausalLM: Loaded model instance.
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
