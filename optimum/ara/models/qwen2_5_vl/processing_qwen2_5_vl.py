# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import logging
import typing
from typing import Any, Optional, Union

import typing_extensions
from transformers import Qwen2_5_VLProcessor as _Qwen2_5_VLProcessor
from transformers.processing_utils import AllKwargsForChatTemplate

from typing_extensions import Unpack


class ExtendedChatTemplateKwargs(AllKwargsForChatTemplate, total=False):
    add_generation_prompt: Optional[bool]
    tokenize: Optional[bool]


class Qwen2_5_VLProcessor(_Qwen2_5_VLProcessor):
    """A thin subclass of the Transformers Qwen2_5_VLProcessor that enforces
    fixed image dimensions (336x336) and video FPS (2) via apply_chat_template.
    """

    def apply_chat_template(  # type: ignore[override]
        self,
        conversation: Union[list[dict[str, Any]], list[list[dict[str, Any]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[ExtendedChatTemplateKwargs],
    ) -> str:
        RESIZED_DIM = 336
        FPS = 2

        for msg in conversation:
            if not isinstance(msg, dict):
                continue

            if msg["role"] == "user":
                for content in msg["content"]:
                    # hardcoding resized heights/widths
                    if (
                        isinstance(content, dict)
                        and content["type"] == "image"
                        and content.get("resized_height", RESIZED_DIM) != RESIZED_DIM
                    ):
                        logging.info(
                            f"Resetting image height to {RESIZED_DIM}px. "
                            f"Got {content['resized_height']}"
                        )
                        content["resized_height"] = RESIZED_DIM

                    if (
                        isinstance(content, dict)
                        and content["type"] == "image"
                        and content.get("resized_width", RESIZED_DIM) != RESIZED_DIM
                    ):
                        logging.info(
                            f"Resetting image width to {RESIZED_DIM}px. "
                            f"Got {content['resized_width']}"
                        )
                        content["resized_width"] = RESIZED_DIM

                    # hardcoding fps
                    if (
                        isinstance(content, dict)
                        and content["type"] == "video"
                        and content.get("fps", FPS) != FPS
                    ):
                        logging.info(f"Resetting FPS to {FPS}px. Got {content['fps']}")
                        content["resized_width"] = FPS

        return super().apply_chat_template(conversation, chat_template, **kwargs)


# ---------------------------------------------------------------------------
# Legacy processor — kept for backwards compatibility
# ---------------------------------------------------------------------------

from typing import Dict, List, cast  # noqa: E402

import torch  # noqa: E402
import PIL.Image  # noqa: E402
import torchvision.io as io  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import BatchFeature  # noqa: E402


class QwenVLProcessor:
    """Stateful video/image processor wrapping Qwen2_5_VLProcessor.

    .. deprecated::
        Use :class:`Qwen2_5_VLProcessor` instead.
        ``QwenVLProcessor`` will be removed in a future release.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the QwenVLProcessor.

        Args:
            model_id: HuggingFace model ID for the processor
        """
        logging.warning(
            "QwenVLProcessor is deprecated and will be removed in a future release. "
            "Please use Qwen2_5_VLProcessor instead."
        )
        self.model_id = model_id
        self.processor = _Qwen2_5_VLProcessor.from_pretrained(model_id)
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
            self.conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not self.text_only:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                self.conversation,
                return_video_kwargs=True,
            )
            if video_kwargs is None:
                video_kwargs = {}

            inputs = self.processor(
                text=[text],
                images=cast(List[Any], image_inputs),
                videos=cast(List[Any], video_inputs),
                padding=True, # pyrefly: ignore [unexpected-keyword]
                return_tensors="pt", # pyrefly: ignore [unexpected-keyword]
                **video_kwargs,
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True, # pyrefly: ignore [unexpected-keyword]
                return_tensors="pt", # pyrefly: ignore [unexpected-keyword]
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
