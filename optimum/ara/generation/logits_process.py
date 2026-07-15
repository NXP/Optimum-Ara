# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

# These components are for internal use and are added appropriately internally.
# Do not add them manually if you do not know what you are doing.

from __future__ import annotations

import torch
from transformers import LogitsProcessor

import logging
from typing import Optional


class SampleLogitsProcessor(LogitsProcessor):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
        # 1. Convert logits to probabilities
        probs = torch.nn.functional.softmax(scores, dim=-1)

        # 2. Sample one token per row
        if self.seed is not None:
            gen = torch.Generator(device=scores.device).manual_seed(self.seed)
            sampled_tokens = torch.multinomial(probs, num_samples=1, generator=gen)
        else:
            sampled_tokens = torch.multinomial(probs, num_samples=1)
        logging.debug(f"sampled tokens: {sampled_tokens}")
        return sampled_tokens  # LongTensor of token ids


class LogitsDequantizer(LogitsProcessor):
    """
    A utility class to dequantize logits by dividing them by (base ** exponent).

    Attributes:
        base (float): The numeric base for scaling.
        exponent (int): The exponent for scaling.
    """

    def __init__(self, base: float = 2.0, exponent: int = 14):
        self.base = base
        self.exponent = exponent

    def __call__(
        self, logits: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
        """
        Dequantize logits by dividing by (base ** exponent).

        Args:
            logits: torch.Tensor or numpy.ndarray of logits (any shape).
                    If integer-typed, it will be cast to float for correct scaling.

        Returns:
            torch.Tensor or numpy.ndarray: scaled logits of the same type as input.
        """

        scale_factor = torch.pow(
            torch.full((), self.base, dtype=torch.float32, device=logits.device),
            self.exponent,
        )
        scores_processed = scores / scale_factor
        return scores_processed
