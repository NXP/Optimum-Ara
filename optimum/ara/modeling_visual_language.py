# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import warnings
import os
import logging
import time
from pathlib import Path
from typing import Optional, Union, Any, cast, Dict, Tuple
from time import perf_counter

import numpy as np
import torch
from transformers import GenerationConfig, PretrainedConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils.generic import ModelOutput
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput

from .configuration_utils import AraConfig, AraPretrainedConfig, InterfaceType
from .generation.configuration_utils import AraGenerationConfig
from .generation import CustomLogitsProcessor
from .modeling_decoder import AraModelForCausalLM
from .utils.constants import DEFAULT_GENERATION_CONFIG_NAME
from .utils.utils import dequantize, quantize

# Configure logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("qwen_vl_utils").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torchvision").setLevel(logging.WARNING)


class AraModelForVisualCausalLM(AraModelForCausalLM):
    """
    Base class for visual language models (LLaVA, Qwen-VL, etc.).

    Provides common functionality for vision-language models including:
    - Image processing and feature extraction
    - Vision-language fusion
    - Quantization utilities
    """

    NUM_TKNS_PER_FRAME: int = 0
    vision_token_id: int = 0
    is_image_model: bool = False
    is_multimodal_model: bool = False

    def __init__(
        self,
        model_path: Union[Path, str],
        config,
        generation_config,
        **kwargs,
    ):
        """
        Initialize visual language model.

        Args:
            session: Session for model management (LLaVASession, etc.)
            model_path: Path to the model directory
            config: Model configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            model_path=model_path,
            config=config,
            generation_config=generation_config,
            **kwargs,
        )
        self.is_image_model = kwargs.get("is_image_model", False)
        self.is_multimodal_model = kwargs.get("is_multimodal_model", False)
        self._token_generation_time: float = 0.0
        self._ttft: float = 0.0
        self._generated_tokens_count: int = 0

        self._token_generation_time: float = 0.0
        self._ttft: float = 0.0
        self._generated_tokens_count: int = 0

    def load_components(self):
        # Load components
        self._load_quantization_constants()
        # Load embeddings
        self._load_embedding_table(self.dvm_path)
        self._load_vision_model_inputs()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        config: Optional[Union[PretrainedConfig, AraPretrainedConfig]] = None,
        *args,
        **kwargs,
    ) -> "AraModelForVisualCausalLM":
        """
        VLM-specific from_pretrained that loads both LLM and vision models.
        """
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        use_cache = kwargs.pop("use_cache", True)
        file_name = kwargs.pop("file_name", None)

        # Load config using parent class method
        config_result = cls._get_config(
            pretrained_model_name_or_path, cast(Optional[AraPretrainedConfig], config)
        )
        if not isinstance(config_result, AraPretrainedConfig):
            raise TypeError(f"Expected AraPretrainedConfig, got {type(config_result)}")
        config_ara = config_result

        # Load generation config use parent class method
        gen_config = cls._get_generation_config(
            pretrained_model_name_or_path,
            config_ara,
            kwargs.pop("generation_config", None),
        )

        # Get LLM DVM path using parent class method
        llm_path = cls._get_llm_dvm_path(
            pretrained_model_name_or_path, config_ara, file_name
        )

        if config_ara.ara is None:
            raise ValueError("config.ara cannot be None")
        # Get vision DVM path using new method
        vision_path = cls._get_vision_dvm_path(config_ara)

        # Create VLM instance
        init_cls = cls(
            model_path=llm_path,
            config=config_ara,
            use_cache=use_cache,
            generation_config=gen_config,
            **kwargs,
        )

        # VLM-specific: Load both LLM and vision models
        init_cls.load_model(str(llm_path), device_map, max_memory)
        init_cls.load_vision_model(str(vision_path), device_map, max_memory)
        init_cls.load_components()

        return init_cls

    @classmethod
    def _get_vision_dvm_path(cls, config: "AraPretrainedConfig"):
        if config is None:
            raise ValueError("config cannot be None")
        if config.ara is None:
            raise ValueError("config.ara cannot be None")
        vision_path = ""
        if hasattr(config.ara, "vision_model_path"):
            vision_model_path = config.ara.vision_model_path
            if (
                vision_model_path is not None
                and os.path.exists(str(vision_model_path))
                and str(vision_model_path).endswith(".dvm")
            ):
                vision_path = Path(vision_model_path)
            else:
                logging.warning(
                    f"incorrect vision model dvm file path provided in config {vision_model_path}"
                )
                logging.warning("vision_model_path should is a path to *.dvm file")
        else:
            logging.warning("'vision_model_path' is not present in config.ara")

        if vision_path == "":
            raise ValueError("Failed to find vision model dvm path")

        logging.info(f"Found vision dvm_file {vision_path}")
        return vision_path.resolve()

    def load_vision_model(self, model_path: str, device_map, max_memory):
        """Load vision model from file."""
        endpoint_id = self._handle_device_map(model_path, device_map, max_memory)
        ret = self.core.load_vision_model(str(model_path), endpoint_id)

        return ret

    def _load_quantization_constants(self):
        """Load quantization constants if available."""
        # This will be implemented by subclasses if needed
        pass

    def _load_vision_model_inputs(self):
        """Load specific vision model inputs."""
        # This will be implemented by subclasses if needed
        pass

    def _get_num_image_tokens(self, pixel_values):
        """Get the number of image tokens for the given pixel values."""
        # This will be implemented by subclasses
        raise NotImplementedError

    def process(self, image, conversation: list, new_image: bool = True, **kwargs):
        """
        Process image and conversation for inference.

        Args:
            image: PIL Image to process
            conversation: Conversation turns for chat template
            new_image: Whether this is a new image
            **kwargs: Additional processing arguments
        """
        # This will be implemented by subclasses
        raise NotImplementedError

    def encode(self):
        """Encode the processed inputs."""
        # This will be implemented by subclasses
        raise NotImplementedError

    def _get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using the vision model.

        Args:
            pixel_values: Image pixel values

        Returns:
            Image features tensor
        """
        # This will be implemented by subclasses
        raise NotImplementedError

    def _merge_input_image_embeds(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge image features with input embeddings.

        Args:
            image_features: Image features tensor
            inputs_embeds: Input embeddings tensor

        Returns:
            Merged embeddings tensor
        """
        # This will be implemented by subclasses
        raise NotImplementedError

    def merge_features(
        self,
        mask_unsqueezed: torch.Tensor,
        feature_embeds: Union[torch.Tensor, "np.ndarray"],
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = mask_unsqueezed.expand_as(text_embeds)
        if isinstance(feature_embeds, np.ndarray):
            feature_embeds = torch.from_numpy(feature_embeds).to(text_embeds.device)
        feature_embeds = feature_embeds.to(text_embeds.dtype)
        text_embeds = text_embeds.masked_scatter(mask_expanded, feature_embeds)
        return text_embeds

    def merge_multi_modal_features(
        self,
        input_ids: torch.Tensor,
        feature_embeds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        text_embeds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        vision_token_id,  # TODO: type?
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Merge multi-modal features."""
        n_video_tokens = (input_ids == vision_token_id).sum().item()
        n_video_features = feature_embeds[0].shape[0]

        if n_video_tokens != n_video_features:
            if n_video_tokens > n_video_features:
                # The processor generated more image tokens than the DVM produced features.
                # This happens with LLaVA-Next tiling: the processor tiles the image
                # into multiple sub-images, but the DVM only processes the first tile.
                # Fix: keep only n_video_features image tokens, replace the rest with pad.
                warnings.warn(
                    f"Trimming image tokens: processor generated {n_video_tokens} "
                    f"but DVM produced {n_video_features} features. "
                    f"Keeping first {n_video_features} image tokens."
                )
                pad_token_id = getattr(self.config, "pad_token_id", 0)

                # Find all positions of the vision token
                vision_positions = (input_ids == vision_token_id).nonzero(as_tuple=True)
                # Keep only the first n_video_features, replace the rest with pad
                if len(vision_positions) == 2:  # [batch, seq]
                    excess_positions = vision_positions[1][n_video_features:]
                    input_ids[
                        vision_positions[0][n_video_features:], excess_positions
                    ] = pad_token_id
                else:  # [seq]
                    excess_positions = vision_positions[0][n_video_features:]
                    input_ids[excess_positions] = pad_token_id

                # Also need to re-embed the modified input_ids
                if self.embedding is not None and self.embedding_scales is not None:
                    text_embeds = (
                        self.embedding(input_ids),
                        self.embedding_scales(input_ids),
                    )
                else:
                    raise ValueError("embedding or embedding_scales is None")

                # Verify the count now matches
                n_video_tokens = (input_ids == vision_token_id).sum().item()

            if n_video_tokens != n_video_features:
                unique_tokens = torch.unique(input_ids)
                possible_image_tokens = [
                    t for t in unique_tokens.tolist() if t > vision_token_id
                ]
                error_msg = (
                    f"Video features and video tokens do not match:\n"
                    f"  Expected token ID: {vision_token_id}\n"
                    f"  Found {n_video_tokens} occurrences in input_ids\n"
                    f"  But have {n_video_features} vision features\n"
                    f"  Possible image tokens (>32000): {possible_image_tokens}"
                )
                raise ValueError(error_msg)

        mask = input_ids == vision_token_id
        mask_unsqueezed = mask.unsqueeze(-1)

        # merging embeddings
        inputs_embeds = self.merge_features(
            mask_unsqueezed, feature_embeds[0], text_embeds[0]
        )

        # merging scales
        inputs_embeds_scales = self.merge_features(
            mask_unsqueezed, feature_embeds[1], text_embeds[1]
        )

        return inputs_embeds, inputs_embeds_scales

    def get_embeddings(
        self, ids: torch.LongTensor
    ) -> torch.Tensor:  # not needed, @naveen to confirm
        """Get embeddings for token IDs."""
        # This will be implemented by subclasses
        raise NotImplementedError

    def vision_inference(
        self, vision_inputs: torch.Tensor, num_frames: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform vision inference."""
        # This will be implemented by subclasses
        raise NotImplementedError

    # TODO: Not needed, naveen to confirm
    def forward(  # pyrefly: ignore[bad-override]
        self,
        input_ids: np.ndarray,
        output_ids: np.ndarray,
        active_tokens: int,
        valid_tokens: int,
        infer_type,
        tokens_to_skip: int = 0,
        **kwargs,
    ) -> tuple[Any, Optional[ModelOutput], float]:  # pyrefly: ignore
        """
        VLM-specific forward method that processes logits.
        Overrides the generic base class method to handle logits processing.
        """
        # Call the parent's forward method to get the raw inference
        status, inf_req, infer_time = self.core.forward(
            input_ids=input_ids,
            output_ids=output_ids,
            active_tokens=active_tokens,
            valid_tokens=valid_tokens,
            infer_type=infer_type,
            tokens_to_skip=tokens_to_skip,
            **kwargs,
        )
        status, inf_req = self.core.process_inference_output(
            status, output_ids, inf_req
        )
        return status, inf_req, infer_time

    # TODO: Not needed, naveen to confirm. Modeling decoders function must be used.
    def _generate_first_token(
        self,
        input_array: np.ndarray,
        output_array: np.ndarray,
        num_valid_tokens: int,
        num_active_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Generates the first token using prompt processing.
        VLM-specific implementation with tokens_to_skip calculation.

        Args:
            input_array (np.ndarray): Input array.
            output_array (np.ndarray): Output array.
            num_valid_tokens (int): Number of valid tokens.
            num_active_tokens (int): Number of active tokens.

        Returns:
            dv_status_code: Status code of the inference.
        """

        # VLM-specific tokens_to_skip calculation
        # This should be overridden by specific VLM implementations
        tokens_to_skip = kwargs.pop("tokens_to_skip", 0)

        status, inf_req, infer_time = self.forward(
            input_array,
            output_array,
            infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_PROMPT_PROCESSING,
            active_tokens=num_active_tokens,
            valid_tokens=num_valid_tokens,
            tokens_to_skip=tokens_to_skip,
        )

        return infer_time

    """
        Add padding to `input_ids` so that the first occurrence of a vision token id
            starts at an index that is a multiple of 128.
        This is done since the rope table only works when the vision token ids start
            at multiples of 128. We do this by padding the system prompt with space.
    """

    def _pad_until_vision_starts_at_multiple_of(
        self, input_ids, vision_token_id, multiple=128
    ):
        if not vision_token_id:
            return input_ids
        # Find first occurrence of tkn_id (e.g., vision token)
        first_idx = (input_ids == vision_token_id).nonzero(as_tuple=True)[1][0].item()

        # Calculate padding needed to align first_idx to multiple
        pad = multiple - (first_idx % multiple) if (first_idx % multiple) else 0

        # Find im_end_token index
        system_end_tkn_index = (
            (input_ids == self.config.eos_token_id).nonzero(as_tuple=True)[1][0].item()
        )
        logging.info("system token ends at %d", system_end_tkn_index)

        # Insert pad_token_id's before system_end_tkn_index
        padded = torch.cat(
            [
                input_ids[:, :system_end_tkn_index],
                torch.full(
                    (1, pad),
                    self.config.sys_pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                input_ids[:, system_end_tkn_index:],
            ],
            dim=-1,
        )

        return padded

    def generate(
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
        Generate text based on video and/or text input using custom generation loop.

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
            **kwargs: Additional generation parameters (e.g., input_ids, pixel_values_videos, pixel_values).

        Returns:
            torch.Tensor: Generated sequences including input tokens, shape (batch_size, sequence_length)
        """
        self._ttft = 0
        self._token_generation_time = 0
        self._generated_tokens_count = 0

        input_ids = kwargs.pop("input_ids", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        pixel_values = kwargs.pop("pixel_values", None)

        input_ids = inputs if inputs is not None else cast(torch.Tensor, input_ids)
        if input_ids is None:
            raise ValueError("Either inputs or input_ids must be provided")

        # Video models must not receive image-only inputs; raise before any other path
        if not self.is_image_model and pixel_values is not None:
            logging.warning(
                "This is a video model and won't work on image inputs, ignoring image inputs"
            )

        # Determine num_frames based on actual inputs used
        if self.is_image_model:
            # Default for Qwen image models (processes as 1 group of 2)
            # LLava only works with num_frames = 1
            if self.config.model_type == "ara_llava":
                num_frames = 1
            else:
                num_frames = 2
            vision_inputs = pixel_values
            vision_token_id = self.config.image_token_id
        elif self.is_multimodal_model:
            if (input_ids == self.config.image_token_id).sum() > 0:
                vision_token_id = self.config.image_token_id
                vision_inputs = pixel_values
                num_frames = 2
            elif (input_ids == self.config.video_token_id).sum() > 0:
                vision_token_id = self.config.video_token_id
                vision_inputs = pixel_values_videos
                num_frames = vision_inputs.shape[0] // (self.NUM_TKNS_PER_FRAME * 2)
            else:
                vision_token_id = None
                vision_inputs = None
                num_frames = None
            # multimodal model requires vision token id to start at multiple of 128
            if vision_token_id is not None:
                input_ids = self._pad_until_vision_starts_at_multiple_of(
                    input_ids, vision_token_id
                )
        else:
            # assuming video model
            vision_inputs = pixel_values_videos
            vision_token_id = self.config.video_token_id
            if vision_inputs is not None and vision_inputs.numel() > 0:
                # Video models usually group frames (e.g. by 2)
                num_frames = vision_inputs.shape[0] // (self.NUM_TKNS_PER_FRAME * 2)
            else:
                num_frames = 0

        mcp_dict = {
            "target_prompt_pre_mcp": 0,
            "target_prompt_post_mcp": 0,
            "target_token_pre_mcp": 0,
            "target_token_post_mcp": 0,
        }

        # setting prompt pre mcp to 0
        kwargs.update(
            {
                "target_prompt_pre_mcp": 0,
            }
        )
        logging.info(
            "Setting prompt_pre_mcp to run on host since VLMs only "
            "run with mcp on host."
        )
        # Update generation config with user parameters BEFORE setting up processors
        generation_config, _ = self._prepare_generation_config(
            generation_config, **kwargs
        )
        ara_cfg = generation_config.ara
        if (
            ara_cfg.target_prompt_post_mcp != mcp_dict["target_prompt_post_mcp"]
            or ara_cfg.target_prompt_pre_mcp != mcp_dict["target_token_pre_mcp"]
            or ara_cfg.target_token_post_mcp != mcp_dict["target_token_post_mcp"]
            or ara_cfg.target_token_pre_mcp != mcp_dict["target_token_pre_mcp"]
        ):
            logging.warning(f"Suggested mcp values for good results are {mcp_dict}")
        self.core.update_llm_params(generation_config)
        self._ensure_generation_operators(generation_config)

        # VLM-specific preprocessing
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long()

        numpy_inputs = input_ids.detach().cpu().numpy().astype(np.int64)
        self._num_input_tokens = torch.from_numpy(numpy_inputs)
        num_max_tokens = self.core.get_max_num_tokens()
        eos_token_ls = (
            generation_config.eos_token_id
            if isinstance(generation_config.eos_token_id, list)
            else [generation_config.eos_token_id]
        )
        if self.embedding is None or self.embedding_scales is None:
            raise ValueError("embedding or embedding_scales is None")

        input_embeds = (self.embedding(input_ids), self.embedding_scales(input_ids))
        if vision_inputs is not None:
            assert num_frames is not None
            video_embeds = self.vision_inference(vision_inputs, num_frames)
            input_embeds = self.merge_multi_modal_features(
                input_ids, video_embeds, input_embeds, vision_token_id
            )

        prepared_inputs = self.prepare_inputs_for_generation(
            input_ids=cast(torch.LongTensor, input_ids),
            inputs_embeds=input_embeds,
            generation_config=generation_config,
        )
        self.input_token_ids = (
            input_ids.detach().cpu().numpy().flatten().astype(np.int64).copy()
        )

        # Initialize generation variables
        input_array = prepared_inputs["input_array"]
        output_array = prepared_inputs["output_array"]

        generated_list = []
        num_tokens = numpy_inputs.size
        next_token_index = num_tokens
        active_tokens = num_tokens
        num_valid_tokens = num_tokens

        # Measure Time to First Token (TTFT)

        # First token generation
        self._ttft = self._generate_first_token(
            input_array,
            output_array,
            active_tokens,
            num_valid_tokens,
            generation_config,
        )
        self._subsequent_start = time.time()
        # Extract next tokens
        next_tokens = self._extract_next_tokens(
            output_array,
            self.input_token_ids,
            generation_config,
            num_valid_tokens=1,
            is_first_token=True,
        )

        # Apply unfinished sequences logic
        unfinished_sequences = np.ones(1, dtype=np.float32)
        next_tokens = (
            next_tokens * unfinished_sequences
            + generation_config.pad_token_id * (1 - unfinished_sequences)
        )
        # Update input token IDs for next iteration
        next_tokens = next_tokens.astype(np.int64)

        self.input_token_ids = np.concatenate(
            [
                self.input_token_ids.astype(np.int64),
                next_tokens.flatten().astype(np.int64),
            ]
        )
        generated_list.append(next_tokens[0])

        if streamer is not None:
            streamer.put(next_tokens.astype(np.int64))

        if generation_config.max_length > 0:
            user_max_length = generation_config.max_length + numpy_inputs.size
            if user_max_length > num_max_tokens:
                user_max_length = num_max_tokens
        else:
            user_max_length = num_max_tokens

        while True:
            if len(generated_list) + numpy_inputs.size >= num_max_tokens or any(
                token in eos_token_ls for token in next_tokens
            ):
                break
            next_tokens_tensor = torch.from_numpy(next_tokens[:, None]).long()
            if self.embedding is None:
                raise ValueError("embedding is None")
            embeds = self.embedding(next_tokens_tensor).flatten()
            if self.embedding_scales is None:
                raise ValueError("embedding_scales is None")
            embed_sclaes = self.embedding_scales(next_tokens_tensor).flatten()
            token_ids = next_tokens.astype(np.int32).view(np.int8)
            input_array = np.concatenate((token_ids, embeds, embed_sclaes))

            status, model_output, infer_time = self.core.forward(
                input_ids=input_array,
                output_ids=output_array,
                active_tokens=next_token_index,
                valid_tokens=1,
                infer_type=self.core.infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION,
                tokens_to_skip=0,
            )
            self._token_generation_time += infer_time
            next_token_index += 1

            # Extract next tokens using _extract_next_tokens
            next_tokens = self._extract_next_tokens(
                output_array,
                self.input_token_ids,
                generation_config,
                num_valid_tokens=1,
                is_first_token=False,
            )

            self.input_token_ids = np.concatenate(
                [self.input_token_ids, next_tokens.flatten()]
            )
            generated_list.append(next_tokens[0])

            if streamer is not None:
                streamer.put(next_tokens.astype(np.int64))

        if streamer is not None:
            streamer.end()
        generated_list = torch.LongTensor(generated_list)
        # Plus one for EOS token
        self._generated_tokens_count = len(generated_list) + 1

        output_torch = torch.cat(
            [torch.from_numpy(numpy_inputs), generated_list.unsqueeze(0)], dim=1
        )
        output_torch = cast(torch.LongTensor, output_torch)
        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=output_torch)
        return output_torch

    def prepare_inputs_for_generation(
        self,
        input_ids: Union[torch.LongTensor, np.ndarray],
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[
            Union[torch.Tensor, np.ndarray, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        cache_position: Optional[torch.LongTensor] = None,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Handle text-only generation call from super().generate()
        # where signature is (input_ids, generation_config)
        if inputs_embeds is None:
            # We are likely called from AraModelForCausalLM.generate with token IDs
            # Delegate to AraModelForCausalLM's implementation
            return super(AraModelForVisualCausalLM, self).prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                generation_config=generation_config,
                **kwargs,
            )

        """
        Prepare input/output arrays in the same format used by the generic VLM path
        (tokens + embeddings + embedding scales), matching the Qwen-VL implementation.
        """
        num_max_tokens = self.core.get_max_num_tokens()
        embedding_dim = self.core.get_embedding_size()

        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required")

        embeds, embeds_scales = inputs_embeds

        # Ensure input_ids is a tensor for size checks
        input_ids_tensor = (
            input_ids
            if isinstance(input_ids, torch.Tensor)
            else torch.from_numpy(np.array(input_ids))
        )
        valid_tokens = embeds.shape[1]
        padding_amt = num_max_tokens - valid_tokens
        embeds = torch.nn.functional.pad(
            embeds, (0, 0, 0, padding_amt), value=0
        ).flatten()
        assert embeds.shape[0] == (num_max_tokens * embedding_dim), (
            f"Expected ({num_max_tokens * embedding_dim}), got {embeds.shape}"
        )

        scales_padding_amt = ((num_max_tokens * embedding_dim) // 64) - (
            embeds_scales.shape[1] * embeds_scales.shape[2]
        )

        embeds_scales = torch.nn.functional.pad(
            embeds_scales.flatten(), (0, scales_padding_amt), value=0
        )
        assert embeds_scales.shape[0] == (num_max_tokens * embedding_dim // 64), (
            f"Expected ({num_max_tokens * embedding_dim // 64}), got {embeds_scales.shape}"
        )

        raw_pad_token_id = getattr(self.config, "pad_token_id", 0)
        pad_token_id = 0 if raw_pad_token_id is None else int(raw_pad_token_id)

        token_ids = (
            torch.nn.functional.pad(
                input_ids_tensor.squeeze(0),
                (0, int(num_max_tokens - input_ids_tensor.size(1))),
                value=pad_token_id,
            )
            .cpu()
            .numpy()
            .astype(np.int32)
            .view(np.int8)
        )

        input_array = np.concatenate(
            (
                token_ids,
                embeds.cpu().numpy().flatten(),
                embeds_scales.cpu().numpy().flatten(),
            )
        )
        output_array = np.zeros(
            (self.core.get_vocab_size() * 4), dtype=np.int32
        )  # 608256 elements
        return {
            "input_array": input_array,
            "output_array": output_array,
        }

    def __del__(self):
        super().__del__()
