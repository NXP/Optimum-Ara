# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np


class CustomLogitsProcessor:
    def __call__(
        self, input_ids: torch.LongTensor, scores: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        first return  : selected token ids from scores (logits vectors)
        second return : draft/candidate token ids
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
