# Copyright (c) 2025-2026, Kinara, Inc. All rights reserved.

from ...modeling_seq2seq import AraModelForSpeechSeq2Seq
from .configuration_whisper import AraWhisperConfig
import torch
from typing import Optional, Tuple, Union, Any, List
import torch.nn as nn
import numpy as np
import soundfile as sf

from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateEncoderDecoderOutput
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from optimum.ara.generation.configuration_utils import AraGenerationConfig

class AraModelForWhisper(AraModelForSpeechSeq2Seq, WhisperGenerationMixin):
    """ARA-specific Whisper model wrapper.

    Minimal class-level documentation; detailed docs are provided on each
    method.
    """

    config_class = AraWhisperConfig

    def __init__(self, *args, **kwargs):
        """Initialize the Ara Whisper model wrapper.

        All positional and keyword args are forwarded to
        :class:`AraModelForSpeechSeq2Seq` initializer which performs model and
        session initialization.

        Args:
            *args: forwarded to parent class.
            **kwargs: forwarded to parent class.
        """
        super().__init__(*args, **kwargs)
        self.fixed_config = [50258, 50259, 50359, 50363]  # start, eng, transcribe, no timestamps

    def generate(  # pyrefly: ignore
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[Union[GenerationConfig, AraGenerationConfig]] = None,
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
        return super().generate(
            inputs=input_features,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            custom_generate=custom_generate,
            **kwargs,
        )

    def quantize(self, input_features):
        """Quantize a floating-point tensor to int8 representation with scales.

        The quantization used here is a custom per-block scheme that groups
        values into 64-element blocks and computes a small integer scale per
        block plus an int8 quantized data array.

        Args:
            tensor (torch.Tensor): The floating point input tensor. Can be on
                any device; it will be moved to CPU for numpy conversion.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): A pair (scales, quantized)
            where `scales` is an int8 array of per-block scale metadata and
            `quantized` is an int8 array with the same shape as `tensor`.

        Notes:
            - This function intentionally returns numpy arrays (not tensors)
                because downstream code concatenates and sends raw bytes to
                the Ara device.
        """

        actual_input_features = input_features.clone().cpu().detach().numpy()
        actual_input_features = np.round(actual_input_features * 74.4745).astype(np.int8)
        actual_input_features_scales = np.ones(3000).astype(np.int8) #dummy

        return (actual_input_features, actual_input_features_scales)

    def pad_tokens(self, input_array: np.ndarray) -> np.ndarray:
        # llm_params_obj = self.core.dv_internal.model._model.llm_params.contents
        num_max_tokens = self.core.get_max_num_tokens() #1536
        pad_token = self.core.get_eos_token_id()

        # Ensure a 1-D copy of the original input token sequence is kept for MCP post-processing
        orig_tokens = input_array.ravel().astype(np.int32).copy()
        # self._num_input_tokens = orig_tokens

        # Pad/truncate to model's max token length
        if orig_tokens.size < num_max_tokens:
            pad_count = int(num_max_tokens - orig_tokens.size)
            pad_arr = np.full(pad_count, pad_token, dtype=np.int32)
            # rest_bytes = np.full(rest_of_bytes, 0, dtype=np.int8)
            input_array = np.concatenate(
                (orig_tokens, pad_arr), axis=0
            )

        else:
            input_array = orig_tokens[: int(num_max_tokens)]
        return input_array

    def _encode(
        self,
        input_array: np.ndarray,
        input_features: torch.Tensor,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> float:
        """
        Generates the first token using prompt processing.

        Args:
            input_array (np.ndarray): Input array.
            output_array (np.ndarray): Output array.
            num_tokens (int): Number of tokens in the prompt.

        Returns:
            dv_status_code: Status code of the inference.
        """

        """Prepare and run Ara prompt-processing for the first token.

        This method performs any host-side pre-processing required when the
        generation configuration specifies that prompt processing should be
        performed on the host (i.e. `target_prompt_pre_mcp == 0`). It then
        calls ``self.forward`` with the appropriate inference type to invoke
        prompt processing on the Ara device. After prompt processing it sets a
        series of fixed configuration tokens required by the device/runtime.

        Args:
            input_array (np.ndarray): The host-side input prompt buffer. For
                Whisper this typically contains token ids or preparsed bytes
                required by the Ara kernel.
            input_features (torch.Tensor): Precomputed audio features or
                embedding tensor expected by the kernel.
            output_array (np.ndarray): Preallocated output buffer the device
                will write into (logits or other response blobs).
            num_valid_tokens (int): Number of valid tokens expected for the
                first inference call.
            num_active_tokens (int): The index of the next active token in
                the device buffer.
            generation_config (Optional[AraGenerationConfig]): Generation
                configuration controlling MCP/host processing flags. If None,
                callers should pass a valid config.
            **kwargs: Additional keyword arguments passed through (e.g.
                tokens_to_skip).

        Returns:
            dv_status_code: Status code returned from the Ara forward call.
        """

        skp_tokns = kwargs.pop("tokens_to_skip", 0)
        context_size = self.core.get_max_num_tokens()
        embedding_size = self.core.get_embedding_size()
        # print(f'hidden_sizes: {hidden_sizes}, context_size: {context_size}')
        if generation_config is None:
            raise ValueError("generation_config cannot be None")
        
        if (
            not generation_config.ara.target_prompt_pre_mcp
        ):  # mcp = 0 means processing on host.
            input_array_block = input_array
            actual_input_features, actual_input_features_scales = self.quantize(input_features)

            actual_input_features = actual_input_features.flatten().astype(np.int8)
            padd_features = np.zeros(
                context_size * embedding_size - actual_input_features.size, dtype=np.int8
            )
            padd_scales = np.zeros(
                context_size * embedding_size//64 - actual_input_features_scales.size, dtype=np.int8
            )
            input_array = np.concatenate(
                (
                    input_array_block.view(np.int8),
                    actual_input_features.flatten().astype(np.int8),
                    padd_features,
                    actual_input_features_scales.flatten().astype(np.int8),
                    padd_scales,
                ),
                axis=0,
            )

        status, _, infer_time = self.core.forward(
            input_array,
            output_array,
            active_tokens=num_active_tokens,
            valid_tokens=num_valid_tokens,
            infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_PROMPT_PROCESSING,
            tokens_to_skip=skp_tokns,
        )

        for config in self.fixed_config:
            status, _, infer_time_fixed = self.core.forward(
                np.array([config]),
                output_array,
                infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION,
                active_tokens=num_active_tokens,
                valid_tokens=1,
                tokens_to_skip=skp_tokns,
            )
            infer_time += infer_time_fixed
        return infer_time  # float: seconds

    def _decode(
         self,
        input_array: np.ndarray,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> Tuple[Any, float]:
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
        """Generate next token(s) using the Ara token-generation kernel.

        This function prepares any host-side inputs required when
        `generation_config.ara.target_token_pre_mcp == 0` (i.e. embedding
        lookup and packing), calls the Ara token-generation inference and,
        if required, applies host-side post-processing of logits through the
        configured logits processor.

        Args:
            input_array (np.ndarray): Packed host/device input for token
                generation. Depending on configuration this may already
                contain embedding bytes or only token ids.
            output_array (np.ndarray): Preallocated output buffer that will
                contain logits or other per-token results from the device.
            num_valid_tokens (int): Number of valid tokens to generate in
                this call.
            num_active_tokens (int): Current active token index used by the
                device kernel.
            generation_config (Optional[AraGenerationConfig]): Generation
                configuration flags used to decide pre/post MCP steps.

        Returns:
            tuple: (status, inf_req_obj, infer_time) where `status` is a
            dv_status_code, `inf_req_obj` is the device-inference request
            structure (if any) and `infer_time` is inference duration in
            seconds (or device-reported time).
        """

        status, inf_req_obj, infer_time = self.core.forward(
            input_array,
            output_array,
            active_tokens=num_active_tokens,
            valid_tokens=num_valid_tokens,
            infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION,
            tokens_to_skip=0,
        )

        return inf_req_obj, infer_time
