# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
import numpy as np
import torch
from typing import Union, Tuple

from ...modeling_visual_language import AraModelForVisualCausalLM
from .configuration_llava import AraLlavaConfig


class AraLlavaForConditionalGeneration(AraModelForVisualCausalLM):
    config_class = AraLlavaConfig

    def __init__(self, *args, **kwargs) -> None:
        # LLaVA is an image model, not a video model
        # Override any user-provided value to ensure it's always True
        is_image_model = kwargs.pop("is_image_model", True)
        # Number of image tokens - will be computed dynamically from vision output
        self._num_image_tokens = None
        # LLaVA 1.5 336x336 standard
        self.NUM_TKNS_PER_FRAME = 576
        super().__init__(*args, **kwargs, is_image_model=is_image_model)

    def _load_quantization_constants(self):
        """Load vision model quantization constants from config-based paths."""
        self.quantization_constants = []

        # Load vision model quantization constants in the order expected by the DVM
        qc_paths = [
            getattr(self.config.ara, "vision_input_scale_path", None),
            getattr(self.config.ara, "vision_add_input_path", None),
            getattr(self.config.ara, "vision_add_input_scale_path", None),
            getattr(self.config.ara, "vision_concat_input_path", None),
            getattr(self.config.ara, "vision_concat_input_scale_path", None),
        ]

        # Load constants from paths
        for i, path in enumerate(qc_paths):
            if path is None:
                continue
            const_path = Path(path)
            # if not const_path.is_absolute():
            #     const_path = (Path(self.core.model_path).parent / const_path).resolve()
            if const_path.exists():
                self.quantization_constants.append(
                    np.fromfile(const_path, dtype=np.int8)
                )
            else:
                # Warn but don't fail - some models may not need all constants
                logging.warning(
                    f"Vision quantization constant {i} file not found: {const_path}",
                    UserWarning,
                )

    def _vision_inference(self, pixel_values_inflated: np.ndarray) -> list:
        input_tensors = []
        input_tensors.append(self.core.get_dv_tensor(pixel_values_inflated, params=None))
        for const_array in self.quantization_constants:
            input_tensors.append(self.core.get_dv_tensor(const_array, params=None))

        # Use llm_params values instead of hardcoded constants
        embedding_size = self.core.get_embedding_size()
        
        # Get number of image tokens - use cached value if available, otherwise from config
        # This will be computed dynamically on first inference if not set
        if self._num_image_tokens is None:
            # For first inference, we need an estimate - use a reasonable default
            # This will be corrected after first inference based on actual output size
            # Standard LLaVA uses 576 (24x24 patches), but this varies by model size
            self._num_image_tokens = 576

            # Try to get from config
            if hasattr(self.config, "num_image_tokens") and self.config.num_image_tokens:
                self._num_image_tokens = self.config.num_image_tokens

        embedding_output = np.empty(
            self._num_image_tokens * embedding_size, dtype=np.int8
        )
        embedding_output_scale = np.empty(
            self._num_image_tokens * (embedding_size // 64), dtype=np.int8
        )
        outputs = [embedding_output, embedding_output_scale]

        output_tensors = []
        for op in outputs:
            output_tensors.append(self.core.get_dv_tensor(op, params=None))

        ret, inf_req = self.core.vision_inference(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )

        return outputs

    def vision_inference(
        self, vision_inputs: torch.Tensor, num_frames: int = 1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if num_frames > 1:
            raise RuntimeError(f"Ara Llava only support one image, provided num_frames = {num_frames}")

        # Handle both LLaVA (4D: batch, C, H, W) and LLaVA-Next (5D: batch, tiles, C, H, W)
        if vision_inputs.dim() == 5:
            # Take only the first tile - the global/thumbnail view
            vision_inputs = vision_inputs[:, 0, :, :, :]

        pixel_values_np = (
            np.clip((vision_inputs.numpy() * 32).astype(np.float16), -128, 127)
            .astype(np.int8)
            .flatten()
        )

        outputs = self._vision_inference(pixel_values_np)
        if not outputs:
            raise RuntimeError("Vision encoding failed")

        embedding_size = self.core.get_embedding_size()
        frame_embeds = outputs[0]
        frame_scales = outputs[1]

        # Compute number of image tokens dynamically from actual output size
        num_image_tokens = frame_embeds.size // embedding_size
        
        # Cache the computed value
        if self._num_image_tokens != num_image_tokens:
            self._num_image_tokens = num_image_tokens

        # Reshape from flattened to (num_patches, embedding_dim)
        frame_embeds = frame_embeds.reshape(
            num_image_tokens,
            embedding_size,
        )

        frame_scales = frame_scales.reshape(
            num_image_tokens,
            embedding_size // 64,
        )

        return frame_embeds, frame_scales