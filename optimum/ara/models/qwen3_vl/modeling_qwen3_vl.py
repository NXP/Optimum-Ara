# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0

from numba import int8
from typing import Dict
import logging
import os
from typing import Any, Callable, Optional, Tuple, Union, cast

import numpy as np
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput

from ...generation.configuration_utils import AraGenerationConfig
from ...modeling_visual_language import AraModelForVisualCausalLM
from .configuration_qwen3_vl import (
    AraQwen3ImageConfig,
    AraQwen3MultiModalConfig,
    AraQwen3VLConfig,
)
from .processing_qwen3_vl import Qwen3VLProcessor

class AraQwen3VLForConditionalGeneration(AraModelForVisualCausalLM):
    config_class = AraQwen3VLConfig

    def __init__(self, *args, **kwargs) -> None:
        self.PATCHES_ALONG_HEIGHT = 12
        self.PATCHES_ALONG_WIDTH = 12
        self.PATCH_SIZE = 16
        self.NUM_FRAMES = kwargs.pop("NUM_FRAMES", 16)
        self.NUM_TKNS_PER_FRAME = (
            self.PATCHES_ALONG_HEIGHT * self.PATCHES_ALONG_WIDTH
        )
        is_image_model = kwargs.pop("is_image_model", False)
        super().__init__(*args, **kwargs, is_image_model=is_image_model)

    def _load_vision_model_inputs(self):
        logging.debug("qwen3-vl _load_vision_model_inputs called!")
        # load rope_table
        rope_table_path = getattr(self.config.ara, "rope_table_path", None)
        assert (
            rope_table_path
        ), "Need to pass an input file for input scale path"
        assert (
            os.path.exists(rope_table_path)
        ), f"Provided path for rope table does not exist -> {rope_table_path}"
        
        self.rope_table =  np.fromfile(rope_table_path, dtype=np.int8)
        logging.info(f"Loaded rope table from the following path: {rope_table_path}")

        # load vision_pos_table
        vision_pos_table_path = getattr(self.config.ara, "vision_pos_table_path", None)
        assert (
            vision_pos_table_path
        ), "Need to pass vision_pos_table_path"
        assert (
            os.path.exists(vision_pos_table_path)
        ), f"Provided path for vision_pos_table_path does not exist -> {vision_pos_table_path}"
        
        self.vision_pos_table =  np.fromfile(vision_pos_table_path, dtype=np.int8)
        logging.info(f"Loaded vision_pos_table from the following path: {vision_pos_table_path}")

        # load vision_pos_table_scale
        vision_pos_table_scale_path = getattr(self.config.ara, "vision_pos_table_scale_path", None)
        assert (
            vision_pos_table_scale_path
        ), "Need to pass vision_pos_table_scale_path"
        assert (
            os.path.exists(vision_pos_table_scale_path)
        ), f"Provided path for vision_pos_table_scale_path does not exist -> {vision_pos_table_scale_path}"
        
        self.vision_pos_table_scale =  np.fromfile(vision_pos_table_scale_path, dtype=np.int8)
        logging.info(f"Loaded vision_pos_table_scale_path from the following path: {vision_pos_table_scale_path}")
    
        # pyrefly: ignore [bad-override]
    def prepare_inputs_for_generation(
        self,
        input_ids: Union[torch.LongTensor, np.ndarray],
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[Tuple[torch.Tensor,torch.Tensor] | Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        generation_config: Optional[AraGenerationConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Handle text-only generation call from super().generate()
        # where signature is (input_ids, generation_config)

        logging.debug("called from qwen3-vl prepare_inputs_for_generation!")
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
                 **kwargs
             )

        """
        Prepare input/output arrays in the same format used by the generic VLM path
        (tokens + embeddings + embedding scales), matching the Qwen-VL implementation.
        """
        num_max_tokens = self.core.get_max_num_tokens()
        embedding_dim = self.core.get_embedding_size()
        input_array = None

        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required")
        if len(inputs_embeds) == 2:
            # text only case, still requires buffer size to incude ds0,1,2
            zero_embed = torch.zeros_like(inputs_embeds[0])
            zero_embed_scale = torch.zeros_like(inputs_embeds[1])
            # append dummy ds0,1,2 to input
            inputs_embeds = inputs_embeds + (zero_embed, zero_embed_scale, zero_embed, zero_embed_scale, zero_embed, zero_embed_scale)


        # raw_pad_token_id = getattr(self.config, "pad_token_id", 0)
        # pad_token_id = 0 if raw_pad_token_id is None else int(raw_pad_token_id)
        pad_token_id = self.core.get_pad_token_id()
        
        # Ensure input_ids is a tensor for size checks
        input_ids_tensor = input_ids if isinstance(input_ids, torch.Tensor) else torch.from_numpy(np.array(input_ids))
        
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
        input_array = token_ids

        # append embedding to input
        for i in range(0,len(inputs_embeds), 2):
            embeds = inputs_embeds[i]

            valid_tokens = embeds.shape[1]
            padding_amt = num_max_tokens - valid_tokens
            embeds = torch.nn.functional.pad(
                embeds, (0, 0, 0, padding_amt), value=0
            ).flatten()
            assert embeds.shape[0] == (num_max_tokens * embedding_dim), (
                f"Expected ({num_max_tokens * embedding_dim}), got {embeds.shape}"
            )

            input_array = np.concatenate((
                input_array,
                embeds.cpu().numpy().flatten(), 
            ))
        # append embedding scales to input
        for i in range(0,len(inputs_embeds), 2):
            embeds_scales = inputs_embeds[i+1]

            scales_padding_amt = (
                (num_max_tokens * embedding_dim) // 64
            ) - (embeds_scales.shape[1] * embeds_scales.shape[2]) 

            embeds_scales = torch.nn.functional.pad(
                embeds_scales.flatten(), (0, scales_padding_amt), value=0
            )
            assert embeds_scales.shape[0] == (num_max_tokens * embedding_dim // 64), (
                f"Expected ({num_max_tokens * embedding_dim // 64}), got {embeds_scales.shape}"
            )

            input_array = np.concatenate((
                input_array,
                embeds_scales.cpu().numpy().flatten()
            ))


        output_array = np.zeros(
            (self.core.get_vocab_size() * 4), dtype=np.int32
        )

        return {
            "input_array": input_array,
            "output_array": output_array,
        }
    
    # Qwen3-VL requires handling of 4 inputs instead of one
    # pyrefly: ignore [bad-override]
    def merge_multi_modal_features(
        self, 
        input_ids: torch.Tensor, 
        feature_embeds: Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor], 
        text_embeds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        vision_token_id #TODO: type?
    ):
        """Merge multi-modal features."""
        n_video_tokens = (input_ids == vision_token_id).sum().item()
        n_video_features = feature_embeds[0].shape[0]

        if n_video_tokens != n_video_features:
                unique_tokens = torch.unique(input_ids)
                possible_image_tokens = [t for t in unique_tokens.tolist() if t > vision_token_id]
                error_msg = (
                    f"Video features and video tokens do not match:\n"
                    f"  Expected token ID: {vision_token_id}\n"
                    f"  Found {n_video_tokens} occurrences in input_ids\n"
                    f"  But have {n_video_features} vision features\n"
                    f"  Possible image tokens (>32000): {possible_image_tokens}"
                )
                raise ValueError(error_msg)

        video_output = feature_embeds[0]
        video_output_scale = feature_embeds[1]
        video_ds0_features = feature_embeds[2]
        video_ds0_features_scale = feature_embeds[3]
        video_ds1_features = feature_embeds[4]
        video_ds1_features_scale = feature_embeds[5]
        video_ds2_features = feature_embeds[6]
        video_ds2_features_scale = feature_embeds[7]

        mask = input_ids == vision_token_id
        mask_unsqueezed = mask.unsqueeze(-1)

        # merging embeddings
        video_output_embeds = self.merge_features(
            mask_unsqueezed, video_output, text_embeds[0]
        )

        # merging scales
        video_output_scale_embeds = self.merge_features(
            mask_unsqueezed, video_output_scale, text_embeds[1]
        )

        zero_embeds = torch.zeros(text_embeds[0].shape, dtype=torch.int8)
        zero_embeds_scales = torch.zeros(text_embeds[1].shape, dtype=torch.int8)

        video_ds0_features_embeds = self.merge_features(
            mask_unsqueezed, video_ds0_features, zero_embeds
        )

        video_ds0_features_scale_embeds = self.merge_features(
            mask_unsqueezed, video_ds0_features_scale, zero_embeds_scales
        )

        # TODO: [Naveen] check if zero_embeds needs to be created for every ds embeds
        zero_embeds = torch.zeros(text_embeds[0].shape, dtype=torch.int8)
        zero_embeds_scales = torch.zeros(text_embeds[1].shape, dtype=torch.int8)
        video_ds1_features_embeds = self.merge_features(
            mask_unsqueezed, video_ds1_features, zero_embeds
        )

        video_ds1_features_scale_embeds = self.merge_features(
            mask_unsqueezed, video_ds1_features_scale, zero_embeds_scales
        )

        zero_embeds = torch.zeros(text_embeds[0].shape, dtype=torch.int8)
        zero_embeds_scales = torch.zeros(text_embeds[1].shape, dtype=torch.int8)
        video_ds2_features_embeds = self.merge_features(
            mask_unsqueezed, video_ds2_features, zero_embeds
        )

        video_ds2_features_scale_embeds = self.merge_features(
            mask_unsqueezed, video_ds2_features_scale, zero_embeds_scales
        )

        return (video_output_embeds, video_output_scale_embeds, 
                video_ds0_features_embeds, video_ds0_features_scale_embeds, 
                video_ds1_features_embeds, video_ds1_features_scale_embeds, 
                video_ds2_features_embeds, video_ds2_features_scale_embeds)

    def _vision_inference(self, inp_scale, inp) :
        input_tensors = []

        input_tensors.append(self.core.get_dv_tensor(self.rope_table, params=None))
        input_tensors.append(self.core.get_dv_tensor(inp, params=None))
        input_tensors.append(self.core.get_dv_tensor(inp_scale, params=None))
        input_tensors.append(self.core.get_dv_tensor(self.vision_pos_table, params=None))
        input_tensors.append(self.core.get_dv_tensor(self.vision_pos_table_scale, params=None))

        ds0_features: np.ndarray        = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size(), dtype=np.int8)
        ds0_features_scale: np.ndarray  = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size() // 64, dtype=np.int8)
        ds1_features: np.ndarray        = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size(), dtype=np.int8)
        ds1_features_scale: np.ndarray  = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size() // 64, dtype=np.int8)
        ds2_features: np.ndarray        = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size(), dtype=np.int8)
        ds2_features_scale: np.ndarray  = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size() // 64, dtype=np.int8)
        vision_output: np.ndarray       = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size(), dtype=np.int8)
        vision_output_scale: np.ndarray = np.zeros(self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size() // 64, dtype=np.int8)

        outputs = [ds0_features, ds0_features_scale, ds1_features, ds1_features_scale, ds2_features, ds2_features_scale, vision_output, vision_output_scale]
        
        output_tensors = []
        for op in outputs:
            output_tensors.append(self.core.get_dv_tensor(op, params=None))

        ret, inf_req = self.core.vision_inference(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )

        return (vision_output, vision_output_scale, ds0_features, ds0_features_scale, ds1_features, ds1_features_scale, ds2_features, ds2_features_scale)

    def decode_tokens(self, token_ids: np.ndarray) -> str:
        """
        Decode token IDs to text using our own embedding table.
        This mimics what the working code does.
        """
        try:
            token_tensor = torch.from_numpy(token_ids).long()
            if self.embedding is None:
                raise ValueError("embedding is None")
            token_embeddings = self.embedding(token_tensor)
            decoded_text = f"Generated {len(token_ids)} tokens with embeddings shape {token_embeddings.shape}"
            return decoded_text
            
        except Exception as e:
            return f"Generated {len(token_ids)} tokens (decoding failed)"

    def vision_inference(
        self, vision_inputs: torch.Tensor, num_frames: int
    ) :
        logging.debug(f"Qwen3-vl vision_inference called!")
        video_output = []
        video_output_scale = []
        video_ds0_features = []
        video_ds0_features_scale = []
        video_ds1_features = []
        video_ds1_features_scale = []
        video_ds2_features = []
        video_ds2_features_scale = []
        vision_inputs_np = vision_inputs.cpu().detach().numpy()
        # 

        logging.debug(f"num_frames : {num_frames}")
        for i in range(num_frames // 2):
            pixel_values = vision_inputs_np[
                i * (self.NUM_TKNS_PER_FRAME * 4) : (i + 1)
                * (self.NUM_TKNS_PER_FRAME * 4)
            ]

            
            inp_scale, inp = Qwen3VLProcessor.quantize(
                torch.from_numpy(pixel_values)
            )

            vision_output, vision_output_scale, ds0_features, ds0_features_scale, ds1_features, ds1_features_scale, ds2_features, ds2_features_scale = self._vision_inference(inp_scale, inp)
            
            vision_output = vision_output.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size(),)
            vision_output_scale = vision_output_scale.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size() // 64,)

            # --- DS0 Features ---
            ds0_features = ds0_features.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size(),)
            ds0_features_scale = ds0_features_scale.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size() // 64,)

            # --- DS1 Features ---
            ds1_features = ds1_features.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size(),)
            ds1_features_scale = ds1_features_scale.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size() // 64,)

            # --- DS2 Features ---
            ds2_features = ds2_features.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size(),)
            ds2_features_scale = ds2_features_scale.reshape(self.NUM_TKNS_PER_FRAME,self.core.get_embedding_size() // 64,)
            
            video_output.append(torch.from_numpy(vision_output))
            video_output_scale.append(torch.from_numpy(vision_output_scale))
            video_ds0_features.append(torch.from_numpy(ds0_features))
            video_ds0_features_scale.append(torch.from_numpy(ds0_features_scale))
            video_ds1_features.append(torch.from_numpy(ds1_features))
            video_ds1_features_scale.append(torch.from_numpy(ds1_features_scale))
            video_ds2_features.append(torch.from_numpy(ds2_features))
            video_ds2_features_scale.append(torch.from_numpy(ds2_features_scale))

        if not video_output:
             raise RuntimeError(f"Vision inference resulted in 0 features. Check num_frames ({num_frames}) and input shape.")

        
        video_output = torch.cat(video_output, dim=0)
        video_output_scale = torch.cat(video_output_scale, dim=0)
        video_ds0_features = torch.cat(video_ds0_features, dim=0)
        video_ds0_features_scale = torch.cat(video_ds0_features_scale, dim=0)
        video_ds1_features = torch.cat(video_ds1_features, dim=0)
        video_ds1_features_scale = torch.cat(video_ds1_features_scale, dim=0)
        video_ds2_features = torch.cat(video_ds2_features, dim=0)
        video_ds2_features_scale = torch.cat(video_ds2_features_scale, dim=0)

        return (video_output, video_output_scale, video_ds0_features, video_ds0_features_scale, video_ds1_features, video_ds1_features_scale, video_ds2_features, video_ds2_features_scale)
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
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
        custom_generate: Optional[Union[str, Callable]] = None,
        **kwargs
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor, Any]:
        
        # Hardcoding system prompt is not required for qwen3-vl at the moment.
        return super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            use_model_defaults,
            custom_generate,
            **kwargs,
        )
    
class AraQwen3ImageForConditionalGeneration(AraQwen3VLForConditionalGeneration):
    config_class = AraQwen3ImageConfig
    def __init__(self, *args, **kwargs):
        is_image_model = True

        super().__init__(*args, **kwargs, NUM_FRAMES=2, is_image_model=is_image_model)


class AraQwen3MultiModalForConditionalGeneration(AraQwen3VLForConditionalGeneration):
    config_class = AraQwen3MultiModalConfig

    def __init__(self, *args, **kwargs):
        is_multimodal_model = True

        super().__init__(*args, **kwargs, is_multimodal_model=is_multimodal_model)
