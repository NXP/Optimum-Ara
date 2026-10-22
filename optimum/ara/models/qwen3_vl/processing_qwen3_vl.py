# Copyright 2026 NXP
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import logging
import typing
from typing import Any, Optional, Union

# dependencies for resizing the image
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import tempfile
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import os
import torch

import typing_extensions
# from transformers import Qwen3_VLProcessor as _Qwen3_VLProcessor
from transformers.processing_utils import AllKwargsForChatTemplate

from typing_extensions import Unpack


class ExtendedChatTemplateKwargs(AllKwargsForChatTemplate, total=False):
    add_generation_prompt: Optional[bool]
    tokenize: Optional[bool]

class Qwen3VLProcessor():
    """A thin subclass of the Transformers Qwen3_VLProcessor that enforces
    fixed image dimensions (384x384) and video FPS (2) via apply_chat_template.
    """
    @staticmethod
    def resize_image(input_path, target_width=384, target_height=384):
        """Reads an image, resizes it, and saves it to a temporary file."""
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV (cv2) is required to resize Qwen3-VL images")
        
        # Load the image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not open or find the image: {input_path}")
            
        # Resize the image
        # Like your video script, this will squash/stretch to fit the exact dimensions
        resized_img = cv2.resize(img, (target_width, target_height))
        
        # Create a temporary file to hold the resized image
        # We use .jpg here, but you can change the suffix to .png if needed
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        output_path = temp_file.name
        temp_file.close()

        # Save the image to the temporary path
        cv2.imwrite(output_path, resized_img)
        
        return output_path
    @staticmethod
    def resize_video(input_path, target_width=384, target_height=384):
        """Reads a video, resizes it, and saves it to a temporary file."""
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV (cv2) is required to resize Qwen3-VL videos")
        
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        # Get original video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create a temporary file to hold the resized video
        # delete=False ensures it stays on disk long enough for the model to read it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_file.name
        temp_file.close()

        # Define the codec and create VideoWriter object
        # 'mp4v' is a standard, widely compatible codec for mp4 in OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # pyrefly: ignore [missing-attribute]
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        # Process frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Reached the end of the video
                
            # Resize the frame to exact dimensions (will squash/stretch if aspect ratio differs)
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)

        # Release everything
        cap.release()
        out.release()
        
        return output_path

    @staticmethod
    def open_image(source):
        if source.startswith("http://") or source.startswith("https://"):
            return Image.open(BytesIO(requests.get(source).content))
        elif os.path.exists(source):
            return Image.open(source)
        else:
            raise ValueError("Invalid source: Not a valid URL or file path")


    # def quantize(inp_array):
    #     # arr_a = tensor.clone().cpu().detach().numpy()
    #     arr_a = inp_array
    #     arr_a = arr_a.reshape((int(arr_a.size / 64)), 64)
    #     arr_b = np.abs(arr_a)
    #     arr_c = np.max(arr_b, axis=1)
    #     arr_c = np.log2(arr_c)
    #     arr_c = np.floor(arr_c) + 1
    #     arr_c = 7 - arr_c
    #     arr_c = arr_c.astype(np.int8)
    #     arr_c = arr_c.reshape(-1, 1)
    #     q_a = np.floor(arr_a * (np.float_power(2, arr_c.astype(np.int32))) + 0.5)
    #     q_a = np.clip(q_a, -128, 127).astype(np.int8)
    #     # breakpoint()
    #     return (arr_c, q_a.reshape(inp_array.shape))

    @staticmethod
    def quantize(inp_tensor: torch.Tensor):
        # Ensure working copy
        arr_a = inp_tensor.view(-1, 64)  # reshape
        # Absolute values
        arr_b = torch.abs(arr_a)
        # Max along rows
        arr_c = torch.max(arr_b, dim=1).values
        # log2 + floor
        arr_c = torch.log2(arr_c)
        arr_c = torch.floor(arr_c) + 1
        # scaling factor
        arr_c = 7 - arr_c
        # convert to int8
        arr_c = arr_c.to(torch.int8)
        # reshape for broadcasting
        arr_c = arr_c.view(-1, 1)
        # compute quantized values
        scale = torch.pow(2.0, arr_c.to(torch.int32))
        q_a = torch.floor(arr_a * scale + 0.5)
        # clip to int8 range
        q_a = torch.clamp(q_a, -128, 127).to(torch.int8)

        return arr_c, q_a.view(inp_tensor.shape)

    @staticmethod
    def dequantize(quantized: np.ndarray, scales: np.ndarray, shape: tuple) -> np.ndarray:
        # divide by 2**scale in order to  dequantize
        q_new = quantized.reshape(int(quantized.size / 64), 64)
        out = scales.reshape(-1, 1)
        q_new = q_new / np.float_power(2, out.astype(np.float16))
        return q_new.reshape(shape)
