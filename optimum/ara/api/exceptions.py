# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0


class DvApiException(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# class DvApiModelLoadExcception(BaseException):
#     def __init__(self, message):
