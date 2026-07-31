# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from typing import Optional, Union, cast

import torch
import torch.nn as nn
import numpy as np
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation import (
    ConfidenceCriteria,
    EosTokenCriteria,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    GenerationConfig,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.logits_process import HammingDiversityLogitsProcessor

logger = logging.getLogger(__name__)


class QEmbedding(nn.Module):
    """
    A embedding layer that keeps weights as int8
    and performs lookups without dequantization.
    """

    def __init__(self, weight_int8: torch.Tensor):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8.to(torch.int8))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = cast(torch.Tensor, self.weight_int8)
        return w.index_select(0, input_ids.view(-1)).view(*input_ids.shape, -1)

    @property
    def weight(self):
        # mimic nn.Embedding.weight for compatibility
        return self.weight_int8


def signal_handler(llm_app, signum, frame):
    if llm_app.llm_session:
        llm_app.llm_session.close_llmapp_session()
    logger.info("You pressed Ctrl+C!")
    sys.exit(1)


class exit_handler(logging.StreamHandler):
    # TODO add continue on error and stop after N errors
    def emit(self, record):
        super().emit(record)
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            print("calling sys.exit")
            logging.shutdown()
            sys.exit(1)


def setup_logger(log_level="info"):
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }
    if log_level.lower() in log_level_map.keys():
        logging.disable(logging.NOTSET)
        logging.basicConfig(
            handlers=[exit_handler()],
            format="[%(levelname).1s:%(asctime)s:%(msecs)03d] [%(name)s] %(message)s",
            level=log_level_map[log_level.lower()],
            datefmt="%y%m%d:%H%M%S",
            force=True,
        )
        logger = logging.getLogger("LLMAPP")
        return logger
    else:
        sys.exit("log level should be one of debug|info|warn|error")


def get_stopping_criteria(
    generation_config: GenerationConfig,
) -> StoppingCriteriaList:
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        criteria.append(
            MaxLengthCriteria(
                max_length=2048,  # todo :: pick it from config
                max_position_embeddings=2048,
            )
        )
    if generation_config.eos_token_id is not None:
        criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
    if (
        generation_config.is_assistant
        and generation_config.assistant_confidence_threshold is not None
        and generation_config.assistant_confidence_threshold > 0
    ):
        criteria.append(
            ConfidenceCriteria(
                assistant_confidence_threshold=generation_config.assistant_confidence_threshold
            )
        )
    return criteria


def quantize(
    tensor: torch.Tensor, name: str = "", dump: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to int8 format.
    Returns: (quantized_array, scales_array)
    """
    numpy_array = tensor.clone().cpu().detach().numpy()
    # calculate scale per each group of 64 elements as 7 - log2(max(abs(group)))
    grouped = numpy_array.reshape((int(numpy_array.size / 64)), 64)
    absolute_values = np.abs(grouped)
    max_values_group = np.max(absolute_values, axis=1)

    # Handle zeros to avoid log2(0) and runtime warnings
    max_values_group = np.maximum(max_values_group, 1e-8)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_max_group = np.log2(max_values_group)
        scales = np.floor(log_max_group) + 1
        scales = 7 - scales
        # Replace any invalid values (inf/nan) with a safe default (7)
        scales = np.where(np.isfinite(scales), scales, 7)
        scales = scales.astype(np.int8)
    if dump:
        scales.tofile(f"{name}_scale.bin")
    # scale each group by 2**scale
    scales = scales.reshape(-1, 1)
    scaled_array = np.floor(
        grouped * (np.float_power(2, scales.astype(np.int32))) + 0.5
    )
    scaled_array = np.clip(scaled_array, -128, 127).astype(np.int8)
    if dump:
        scaled_array.tofile(f"{name}.bin")
    return (scaled_array.flatten(), scales.flatten())


def dequantize(quantized: np.ndarray, scales: np.ndarray, shape: tuple) -> torch.Tensor:
    """Dequantize int8 array back to float tensor."""
    q_new = quantized.reshape(int(quantized.size / 64), 64)
    out = scales.reshape(-1, 1)
    q_new = q_new / np.float_power(2, out.astype(np.float16))
    return torch.from_numpy(q_new.reshape(shape))


def human_readable_size(num_bytes: int, suffix="B") -> str:
    """
    Convert a byte count into a human-readable string using KB, MB, GB, etc.
    """
    units = ["", "K", "M", "G", "T", "P"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}{suffix}"
        value /= 1024.0
    return ""
