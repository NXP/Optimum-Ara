# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

"""
QwenVLProcessor for Demo Integration
A copy of the unified processor designed to work with the existing demo structure
"""

from typing import Dict, List, Optional, Any, cast

import torch
import PIL.Image
import torchvision.io as io
from qwen_vl_utils import process_vision_info
from transformers import BatchFeature, Qwen2_5_VLProcessor


class QwenVLProcessor:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the QwenVLProcessor.

        Args:
            model_id: HuggingFace model ID for the processor
        """
        self.model_id = model_id
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
        self.processed_inputs: Optional[BatchFeature] = None
        self.video_path: Optional[str] = None
        self.conversation: Optional[List[Dict[str, Any]]] = None
        self.text_only = False

        # Processing parameters from demo constants
        self.fps = 2
        self.resized_height = 12 * 28
        self.resized_width = 12 * 28
        self.num_frames = 16
        self.max_duration = 26

    @property
    def tokenizer(self):
        """Expose the underlying tokenizer."""
        return cast(Any, self.processor).tokenizer

    def decode(self, *args, **kwargs):
        """
        Decode the generated tokens to text.

        Args:
            *args: Arguments for the processor's decode method
            **kwargs: Keyword arguments for the processor's decode method

        Returns:
            str: Decoded text
        """
        return self.processor.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        """
        Apply chat template to the conversation.

        Args:
            *args: Arguments for the processor's apply_chat_template method
            **kwargs: Keyword arguments for the processor's apply_chat_template method

        Returns:
            str: Formatted text
        """
        return self.processor.apply_chat_template(*args, **kwargs)

    def __call__(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        conversation: Optional[List[Dict]] = None,
    ) -> BatchFeature:
        """
        Callable method for processing video and text.
        Main entry point for preprocessing.

        Args:
            video_path: Path to the video file
            prompt: Text prompt for the video
            conversation: Optional conversation format

        Returns:
            BatchFeature: Processed inputs ready for the model
        """
        if (
            video_path is not None
            and self.get_video_duration(video_path) > self.max_duration
        ):
            raise ValueError(
                (
                    "Video duration exceeds maximum allowed duration of "
                    f"{self.max_duration} seconds."
                )
            )
        return self.process_video_and_text(prompt, video_path, conversation)

    def get_video_duration(self, video_path):
        """
        Get the duration of the video in seconds.

        Args:
            video_path: Path to the video file

        Returns:
            float: Duration of the video in seconds
        """
        # pts = presentation timestamps
        pts, _ = io.read_video_timestamps(video_path, pts_unit="sec")

        # Duration is simply the last timestamp
        duration = pts[-1] if len(pts) > 0 else 0.0
        return int(duration)

    def process_video_and_text(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        conversation: Optional[List[Dict]] = None,
    ) -> BatchFeature:
        """
        Main preprocessing method.

        Args:
            video_path: Path to the video file
            prompt: Text prompt for the video
            conversation: Optional conversation format

        Returns:
            BatchFeature: Processed inputs ready for the model
        """
        self.text_only = True if not video_path else False

        if conversation is None:
            self.conversation = self._create_conversation(prompt, video_path)
        else:
            self.conversation = conversation

        self.processed_inputs = self._process_inputs()

        if self.processed_inputs is None:
            raise ValueError("Failed to process inputs")

        return self.processed_inputs

    def _create_conversation(
        self, prompt: str, video_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Create conversation format.

        Args:
            prompt: Text prompt
            video_path: Path to video file (optional, uses self.video_path if not provided)

        Returns:
            List[Dict]: Conversation
        """

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if video_path:
            video_obj = {
                "type": "video",
                "video": video_path,
                "fps": self.fps,
                "resized_height": self.resized_height,
                "resized_width": self.resized_width,
            }
            content = conversation[0]["content"]
            if isinstance(content, list):
                cast(List[Any], content).insert(0, video_obj)

        return conversation

    def _process_inputs(self) -> BatchFeature:
        """
        Process video and text inputs.

        Returns:
            BatchFeature: Processed inputs
        """
        if self.conversation is None:
            raise ValueError("conversation cannot be None")

        text = self.processor.apply_chat_template(
            self.conversation, tokenize=False, add_generation_prompt=True
        )

        if not self.text_only:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                self.conversation, return_video_kwargs=True
            )
            if video_kwargs is None:
                video_kwargs = {}

            inputs = self.processor(
                text=[text],
                images=cast(List[Any], image_inputs),
                videos=cast(List[Any], video_inputs),
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        return inputs

    def get_processed_inputs(self) -> Optional[BatchFeature]:
        """Get the processed inputs if available."""
        return self.processed_inputs

    def get_video_features(self) -> Optional[torch.Tensor]:
        """Get video features if available."""
        if self.processed_inputs is None:
            return None

        if "pixel_values_videos" in self.processed_inputs:
            return self.processed_inputs["pixel_values_videos"]
        return None

    def get_attention_mask(self) -> Optional[torch.Tensor]:
        """Get attention mask if available."""
        if self.processed_inputs is None:
            return None

        if "attention_mask" in self.processed_inputs:
            return self.processed_inputs["attention_mask"]
        return None

    def get_processing_info(self) -> Dict:
        """Get information about the processing parameters."""
        return {
            "model_id": self.model_id,
            "fps": self.fps,
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "num_frames": self.num_frames,
            "video_path": str(self.video_path) if self.video_path else None,
            "has_processed_inputs": self.processed_inputs is not None,
            "implementation": "demo_integration",
        }

    def reset(self):
        """Reset the processor state."""
        self.processed_inputs = None
        self.video_path = None
        self.conversation = None
