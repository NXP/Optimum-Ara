# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

# These components are for internal use and are added appropriately internally.
# Do not add them manually if you do not know what you are doing.

import torch
import logging
from typing import Optional, cast, Union, Any
from transformers import LogitsProcessor


class SpeculativeDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, pad_token_id: int, seed: Optional[int] = None, do_sample: bool = False):
        """
        Args:
            pad_token_id: Padding token ID from the model.
            seed: Optional seed for reproducibility.
            do_sample: Whether to use stochastic sampling or deterministic (greedy) verification.
        """
        self.draft_token_ids = None
        self.draft_token_probabilities = None
        self.seed = seed
        self.pad_token_id = pad_token_id
        self.do_sample = do_sample
        self.gen = None
        if self.seed is not None:
            self.gen = torch.Generator().manual_seed(self.seed)

    def update_draft(
        self,
        draft_token_ids: torch.LongTensor,
        draft_token_probabilities: torch.FloatTensor,
    ):
        """Updates the draft tokens and probabilities for the next step."""
        self.draft_token_ids = torch.cat(
            [draft_token_ids, torch.tensor([self.pad_token_id], device=draft_token_ids.device)]
        )
        self.draft_token_probabilities = torch.cat(
            [
                draft_token_probabilities,
                torch.tensor([0.0], device=draft_token_probabilities.device),
            ]
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: Input IDs (unused in this processor, but required by signature).
            scores: Logits tensor of shape (batch_size, vocab_size).
        """
        # Ensure inputs are on the same device
        # logging.debug(f"SpeculativeDecodingLogitsProcessor: processing scores shape={scores.shape}, draft_token_ids={self.draft_token_ids}")
        if self.draft_token_ids is None or self.draft_token_probabilities is None:
            return scores
        if self.draft_token_ids.device != scores.device:
            self.draft_token_ids = self.draft_token_ids.to(scores.device)
        if self.draft_token_probabilities.device != scores.device:
            self.draft_token_probabilities = self.draft_token_probabilities.to(
                scores.device
            )

        # 1. Calculate Target Probabilities (P_target)
        batch_size = scores.shape[0]

        accepted_idx = 1

        for i in range(batch_size):
            probs = torch.nn.functional.softmax(scores[i], dim=-1)
            # Extract P_target and P_draft for current candidate
            draft_id = self.draft_token_ids[i]
            
            if not self.do_sample:
                target_token_id = torch.argmax(scores[i])
                if target_token_id == draft_id:
                    # ACCEPTED: Force the draft token to be selected
                    scores[i, :] = -float("inf")
                    scores[i, draft_id] = 0.0
                    accepted_idx = i + 1
                else:
                    # REJECTED: Force target's argmax token and discard the rest
                    scores[i, :] = -float("inf")
                    scores[i, target_token_id] = 0.0
                    accepted_idx = i + 1
                    break
                continue

            p_target = probs[draft_id]
            p_draft = self.draft_token_probabilities[i]

            # Calculate Acceptance Probability alpha = min(P_target / P_draft, 1.0)
            alpha = torch.minimum(
                p_target / (p_draft + 1e-10), torch.tensor(1.0, device=scores.device)
            )

            # Generate random number
            u = (
                torch.rand(1, generator=self.gen, device=scores.device).item()
                if self.gen
                else torch.rand(1, device=scores.device).item()
            )

            if u > 1 - alpha:
                # ACCEPTED: Force the draft token to be selected
                scores[i, :] = -float("inf")
                scores[i, draft_id] = 0.0
                accepted_idx = i + 1
            else:
                # REJECTED: Keep original scores for this token and discard the rest
                break
        return cast(torch.FloatTensor, scores[:accepted_idx, :])


class SampleLogitsProcessor(LogitsProcessor):
    def __init__(self, seed: Optional[int] = None, do_sample: bool = False) -> None:
        self.seed = seed
        self.do_sample = do_sample
        self.gen = None
        if self.seed is not None:
            self.gen = torch.Generator().manual_seed(self.seed)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
        if not self.do_sample:
            selected_tokens = torch.argmax(scores, dim=-1, keepdim=True)
            # logging.debug(f"greedy tokens: {selected_tokens}")
            return selected_tokens
            
        # 1. Convert logits to probabilities
        probs = torch.nn.functional.softmax(scores, dim=-1)

        # 2. Sample one token per row
        if self.seed is not None:
            sampled_tokens = torch.multinomial(probs, num_samples=1, generator=self.gen)
        else:
            sampled_tokens = torch.multinomial(probs, num_samples=1)
        # logging.debug(f"sampled tokens: {sampled_tokens}")
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

    def __call__(self, logits: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
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
                self.exponent
        )
        scores_processed =  scores / scale_factor
        return scores_processed

