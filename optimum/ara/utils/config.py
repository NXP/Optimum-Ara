# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Any
from pathlib import Path
import yaml
import os


class PostProcessConfig(BaseModel):
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    repetition_penalty: float
    num_repeat_ngram_size: int
    ngram_panalty: int
    apply_repetition_penalty: bool
    generation_mode: Literal["sample", "greedy_search"]
    temperature: float
    top_k: int
    top_p: float
    num_max_iterations: int
    use_chat_template: bool
    num_total_tokens: int
    num_embeddings_per_token: int
    num_max_tokens: int

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Top_p must be between 0 and 1")
        return v


class AppConfig(BaseModel):
    inference_device: Literal["MOCK", "ARA2"]
    tokenizer_path: str
    lookup_table_path: str
    lookup_table_scale_path: str
    model_path: str
    proxy_socket: str
    precompiled_url: str

    @field_validator(
        "tokenizer_path", "lookup_table_path", "lookup_table_scale_path", "model_path"
    )
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if v:  # Only process non-empty paths
            absolute_path = os.path.abspath(os.path.expanduser(v))
            return absolute_path
        return v


class Config(BaseModel):
    postprocess: PostProcessConfig
    app: AppConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)


# Example usage:
if __name__ == "__main__":
    try:
        # Load configuration from YAML file
        config = Config.from_yaml("config.yaml")

        # Access configuration values
        print(f"Inference device: {config.app.inference_device}")
        print(f"Model path: {config.app.model_path}")  # Will be absolute path
        print(f"Temperature: {config.postprocess.temperature}")

    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
