# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator
from transformers import GenerationConfig

from ..utils.constants import DEFAULT_GENERATION_CONFIG_NAME


class AraHostConfig(BaseModel):
    """
    Ara specific configuration for generation.
    ara_cfg = AraHostConfig(
        target_token_post_mcp=1,
        target_token_pre_mcp=1,
        target_prompt_post_mcp=1,
        target_prompt_pre_mcp=1,
    )
    """

    target_token_post_mcp: int = Field(1, ge=0, le=1)
    target_token_pre_mcp: int = Field(1, ge=0, le=1)
    target_prompt_post_mcp: int = Field(1, ge=0, le=1)
    target_prompt_pre_mcp: int = Field(1, ge=0, le=1)


class AraGenerationConfig(GenerationConfig):
    """
    Generation configuration for Ara models.
    This class extends the GenerationConfig to include Ara specific parameters.
    It contains the AraHostConfig which holds the MCP values for token and prompt generation.
    It can be initialized with default values or loaded from a pretrained model directory.
    """

    def __init__(
        self,
        ara_cfg: AraHostConfig,
        **kwargs,
    ) -> None:
        """
        Initialize AraGenerationConfig.

        Args:
            ara_cfg (AraHostConfig): The Ara-specific configuration.
            **kwargs: Additional keyword arguments for GenerationConfig.
        """
        super().__init__(**kwargs)
        self.top_k = kwargs.pop("top_k", 1)
        self.top_p = kwargs.pop("top_p", 0.0)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.ara = ara_cfg

    @classmethod
    def make_ara_config(
        cls,
        target_token_post_mcp: int = 1,
        target_token_pre_mcp: int = 1,
        target_prompt_post_mcp: int = 1,
        target_prompt_pre_mcp: int = 1,
    ) -> AraHostConfig:
        """
        Create an AraHostConfig instance with the specified MCP values.

        Args:
            target_token_post_mcp (int): MCP value for post-processing of next_token generation.
            target_token_pre_mcp (int): MCP value for token pre-processing of next_token generation.
            target_prompt_post_mcp (int): MCP value for prompt post-processing of first_token generation.
            target_prompt_pre_mcp (int): MCP value for prompt pre-processing of first_token generation.
        Setting any of these values to 0 will perform post/pre-processing on the host machine i.e cpu/server.
        Setting any of these values to 1 will perform post/pre-processing on the Ara devices.

        Returns:
            AraHostConfig: The created configuration.
        """
        return AraHostConfig(
            target_token_post_mcp=target_token_post_mcp,
            target_token_pre_mcp=target_token_pre_mcp,
            target_prompt_post_mcp=target_prompt_post_mcp,
            target_prompt_pre_mcp=target_prompt_pre_mcp,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "AraGenerationConfig":
        """
        Load an AraGenerationConfig from a pretrained model directory or file.

        Args:
            pretrained_model_name (str or PathLike): Path to the model directory or file.
            **kwargs: Additional keyword arguments.

        Returns:
            AraGenerationConfig: The loaded configuration.
        """
        if os.path.exists(pretrained_model_name):
            json_file = (
                pretrained_model_name
                if os.path.isfile(pretrained_model_name)
                else os.path.join(
                    pretrained_model_name,
                    config_file_name
                    if config_file_name
                    else DEFAULT_GENERATION_CONFIG_NAME,
                )
            )
            return cls.from_json_file(json_file, **kwargs)
        else:
            temp = GenerationConfig.from_pretrained(
                pretrained_model_name,
                config_file_name=config_file_name,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs,
            )
            ara_cfg = cls.make_ara_config()
            return cls(ara_cfg=ara_cfg, **temp.to_dict())

    def to_json_string(self, use_diff: bool = False, ignore_metadata=False) -> str:
        """
        Serialize the configuration to a JSON-formatted string.

        Args:
            use_diff (bool, optional): Whether to use diff dict.
            ignore_metadata (bool, optional): Whether to ignore metadata.

        Returns:
            str: The JSON string representation.
        """
        combined_dict = self.to_diff_dict()
        combined_dict.update({"ara": self.ara.model_dump()})
        return json.dumps(combined_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json_file(
        cls, json_file: Union[str, os.PathLike], **kwargs
    ) -> "AraGenerationConfig":
        """
        Load an AraGenerationConfig from a JSON file.

        Args:
            json_file (str or PathLike): Path to the JSON file.

        Returns:
            AraGenerationConfig: The loaded configuration.
        """
        if os.path.isdir(json_file):
            json_file = os.path.join(json_file, DEFAULT_GENERATION_CONFIG_NAME)
        assert os.path.exists(json_file)
        config_dict = cls._dict_from_json_file(json_file)
        merged_kwargs = config_dict | kwargs
        ara_dict = config_dict.pop("ara", {})
        ara_cfg = cls.make_ara_config(**ara_dict)
        return cls(ara_cfg=ara_cfg, **merged_kwargs)

    @classmethod
    def from_model_config(cls, model_config, **kwargs):
        """
        Create an AraGenerationConfig from a model configuration.

        Args:
            model_config: The model configuration.

        Returns:
            AraGenerationConfig: The created configuration.
        """
        gen_cfg = GenerationConfig.from_model_config(model_config)

        ara_kwargs = {
            k: kwargs[k] for k in list(AraHostConfig.model_fields.keys()) if k in kwargs
        }
        ara_cfg = cls.make_ara_config(**ara_kwargs)
        return cls(ara_cfg=ara_cfg, **gen_cfg.to_dict())
