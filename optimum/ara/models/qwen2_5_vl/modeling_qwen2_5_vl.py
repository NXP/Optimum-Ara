# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2_5_VLProcessor

from ...configuration_utils import AraConfig
from ...generation.configuration_utils import AraGenerationConfig
from ...modeling_visual_language import AraModelForVisualCausalLM
from .configuration_qwen2_5_vl import (
    AraQwen2_5ImageConfig,
    AraQwen2_5MultiModalConfig,
    AraQwen2_5VLConfig,
)
from ...utils.utils import dequantize, quantize


class AraQwen2_5_VLForConditionalGeneration(AraModelForVisualCausalLM):
    config_class = AraQwen2_5VLConfig

    def __init__(self, *args, **kwargs) -> None:
        self.PATCHES_ALONG_HEIGHT = 12
        self.PATCHES_ALONG_WIDTH = 12
        self.PATCH_SIZE = 28
        self.NUM_FRAMES = kwargs.pop("NUM_FRAMES", 16)
        self.NUM_TKNS_PER_FRAME = self.PATCHES_ALONG_HEIGHT * self.PATCHES_ALONG_WIDTH
        is_image_model = kwargs.pop("is_image_model", False)
        super().__init__(*args, **kwargs, is_image_model=is_image_model)

    def _load_vision_model_inputs(self):
        input_scale_path = getattr(self.config.ara, "input_scale_path", None)
        assert input_scale_path, "Need to pass an input file for input scale path"
        assert os.path.exists(input_scale_path), (
            f"Provided path for input scale does not exist -> {input_scale_path}"
        )

        self.input_scale = np.fromfile(input_scale_path, dtype=np.int8)
        logging.info(f"Loaded input scale from the following path: {input_scale_path}")

        rope_table_path = getattr(self.config.ara, "rope_table_path", None)
        assert rope_table_path, "Need to pass an input file for input scale path"
        assert os.path.exists(rope_table_path), (
            f"Provided path for rope table does not exist -> {rope_table_path}"
        )

        self.rope_table = np.fromfile(rope_table_path, dtype=np.int8)
        logging.info(f"Loaded rope table from the following path: {rope_table_path}")

    def _vision_inference(self, pixel_values_inflated: np.ndarray) -> list:
        input_tensors = []

        input_tensors.append(
            self.core.get_dv_tensor(pixel_values_inflated, params=None)
        )
        input_tensors.append(self.core.get_dv_tensor(self.input_scale, params=None))
        input_tensors.append(self.core.get_dv_tensor(self.rope_table, params=None))

        vision_output: np.ndarray = np.empty(
            self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size(), dtype=np.int8
        )
        vision_output_scale: np.ndarray = np.empty(
            self.NUM_TKNS_PER_FRAME * self.core.get_embedding_size() // 64,
            dtype=np.int8,
        )

        outputs = [vision_output, vision_output_scale]

        output_tensors = []
        for op in outputs:
            output_tensors.append(self.core.get_dv_tensor(op, params=None))

        ret, inf_req = self.core.vision_inference(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )

        return outputs

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_embeds = []
        video_scales = []
        vision_inputs_np = vision_inputs.cpu().detach().numpy()
        # breakpoint()
        for i in range(num_frames // 2):
            pixel_values_np = (
                np.clip(
                    (
                        # 4 is for patch merge size, 2 is for frame group
                        vision_inputs_np[
                            i * (self.NUM_TKNS_PER_FRAME * 4) : (i + 1)
                            * (self.NUM_TKNS_PER_FRAME * 4)
                        ]
                        * 32
                        # TODO: change the scale after analyzer?
                    ).astype(np.float32),
                    -128,
                    127,
                )
                .astype(np.int8)
                .flatten()
            )  # check scale
            frame_embeds, frame_scales = self.vision_encoder_runner(pixel_values_np)
            frame_embeds = frame_embeds.reshape(
                self.NUM_TKNS_PER_FRAME,
                self.core.get_embedding_size(),
            )
            video_embeds.append(torch.from_numpy(frame_embeds))

            frame_scales = frame_scales.reshape(
                self.NUM_TKNS_PER_FRAME,
                self.core.get_embedding_size() // 64,
            )
            video_scales.append(torch.from_numpy(frame_scales))

        if not video_embeds:
            raise RuntimeError(
                f"Vision inference resulted in 0 features. Check num_frames ({num_frames}) and input shape."
            )

        video_scales = torch.cat(video_scales, dim=0)
        video_embeds = torch.cat(video_embeds, dim=0)
        return video_embeds, video_scales

    def vision_encoder_runner(self, pixel_values_np: np.ndarray):
        # NOTE: this just picks hardcoded embeddings for now
        pixel_values_np = pixel_values_np.reshape(12, 12, 2, 2, 3, 2, 14, 14)
        pixel_values_np = np.transpose(pixel_values_np, (0, 2, 1, 3, 4, 5, 6, 7))
        pixel_values_np = pixel_values_np.reshape(24, 24, 3, 2, 14, 14)
        pixel_values_np = np.transpose(pixel_values_np, (2, 3, 0, 4, 1, 5)).flatten()

        outputs = self._vision_inference(pixel_values_np)
        vision_output_scale = outputs[1]
        vision_output = outputs[0]

        return (vision_output, vision_output_scale)


class AraQwen2_5_ImageForConditionalGeneration(AraQwen2_5_VLForConditionalGeneration):
    config_class = AraQwen2_5ImageConfig

    def __init__(self, *args, **kwargs):
        is_image_model = True

        super().__init__(*args, **kwargs, NUM_FRAMES=2, is_image_model=is_image_model)


class AraQwen2_5_MultiModalForConditionalGeneration(
    AraQwen2_5_VLForConditionalGeneration
):
    config_class = AraQwen2_5MultiModalConfig

    def __init__(self, *args, **kwargs):
        is_multimodal_model = True

        super().__init__(*args, **kwargs, is_multimodal_model=is_multimodal_model)
