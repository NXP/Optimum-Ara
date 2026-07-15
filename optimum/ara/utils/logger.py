# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys

# Include timestamp in format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s] %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ANSI color codes for levels
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
    "RESET": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Add color
        color = COLORS.get(record.levelname, "")
        reset = COLORS["RESET"]
        # Define timestamp format
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        # Build and apply format
        log_fmt = f"{color}{LOG_FORMAT}{reset}"
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)


def setup_logger():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(LOG_FORMAT))

    logging.root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logging.root.handlers = [handler]
