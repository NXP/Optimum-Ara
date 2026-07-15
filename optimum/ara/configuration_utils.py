# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Optional, Union, Any, TypeVar

from pydantic import BaseModel
from transformers import AutoConfig, PretrainedConfig
from transformers.configuration_utils import recursive_diff_dict

from .api.core_configurations import InterfaceType, DEFAULT_INTERFACE_TYPE
from .utils.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_IP_ADDRESS,
    DEFAULT_NAMED_PIPE,
    DEFAULT_PORT,
    DEFAULT_SOCKET_PATH,
)


class AraConfig(BaseModel):
    """
    Base class for handling Ara parameters required during loading the models.
    """

    inference_device: str = DEFAULT_INFERENCE_DEVICE
    dvm_path: str
    interface_ip_address: str = DEFAULT_IP_ADDRESS
    interface_port: int = DEFAULT_PORT
    interface_named_pipe: str = DEFAULT_NAMED_PIPE
    interface_socket_file: str = DEFAULT_SOCKET_PATH
    interface_type: InterfaceType = DEFAULT_INTERFACE_TYPE


# type hinting: specifying the type of config class that inherits from AraPretrainedConfig
SpecificAraPretrainedConfigType = TypeVar(
    "SpecificAraPretrainedConfigType", bound="AraPretrainedConfig"
)


class AraPretrainedConfig(PretrainedConfig):
    """
    Ara Pretrained Config
    """

    config_class: type = AutoConfig
    ara_class: type[AraConfig] = AraConfig

    def __init__(
        self,
        ara_cfg: Optional[ara_class] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ara = ara_cfg if ara_cfg is not None else self.make_ara_config()
        self.is_composition = True

    @classmethod
    def make_ara_config(
        cls,
        inference_device: str = DEFAULT_INFERENCE_DEVICE,
        dvm_path: str = "",
        interface_ip_address: str = DEFAULT_IP_ADDRESS,
        interface_port: int = DEFAULT_PORT,
        interface_named_pipe: str = DEFAULT_NAMED_PIPE,
        interface_socket_file: str = DEFAULT_SOCKET_PATH,
        interface_type: InterfaceType = DEFAULT_INTERFACE_TYPE,
        **kwargs,
    ) -> Any:
        logging.debug(f"{dvm_path=}")
        return cls.ara_class(
            inference_device=inference_device,
            dvm_path=dvm_path,
            interface_ip_address=interface_ip_address,
            interface_port=interface_port,
            interface_named_pipe=interface_named_pipe,
            interface_socket_file=interface_socket_file,
            interface_type=interface_type,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls: type[SpecificAraPretrainedConfigType],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> SpecificAraPretrainedConfigType:
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path)
        ara_dict = config_dict.get("ara", {})
        ara_dict = cls.update_ara_dict_params(ara_dict, **kwargs)
        # ara_cfg = cls.make_ara_config(**ara_dict)
        config_dict["ara"] = ara_dict

        return cls.from_dict(config_dict, **kwargs)  # pyrefly: ignore=bad-override

    @classmethod
    def update_ara_dict_params(cls, ara_dict, **kwargs):
        """
        Updates the ara specific parameters sent as kwargs in from_pretrained api
        """
        for key, value in kwargs.items():
            logging.debug(f"{key=}")
            ara_cls = cls.ara_class
            if key in ara_cls.model_fields.keys():
                ara_dict[key] = value
        return ara_dict

    def to_diff_dict(self):
        """
        This function is copied directly from transformers library.
        Reference: transformers/configuration_utils.py

        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = (
            PretrainedConfig().to_dict() if not self.is_composition else {}
        )

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(
                    value, class_config_dict[key], config_obj=getattr(self, key, None)
                )
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        if hasattr(self, "dict_torch_dtype_to_str"):
            self.dict_torch_dtype_to_str(serializable_config_dict)
        elif hasattr(self, "dict_dtype_to_str"):
            self.dict_dtype_to_str(serializable_config_dict)
        else:
            pass

        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]

        return serializable_config_dict

    def to_json_string(self, use_diff: bool = False) -> str:
        combined_dict = self.to_diff_dict()
        combined_dict.update({"ara": self.ara.model_dump()})
        return json.dumps(combined_dict, indent=2, sort_keys=True) + "\n"

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, DEFAULT_CONFIG_NAME)
        self.to_json_file(output_config_file, use_diff=True)

    @classmethod
    def from_json_file(
        cls, json_file: Union[str, os.PathLike]
    ) -> "AraPretrainedConfig":
        """
        Load AraPretrainedConfig directly from the json_file passed in the function.
        """
        if os.path.isdir(json_file):
            json_file = os.path.join(json_file, DEFAULT_CONFIG_NAME)
        assert os.path.exists(json_file)
        config_dict = cls._dict_from_json_file(json_file)

        ara_dict = config_dict.pop("ara", {})
        ara_cfg = cls.make_ara_config(**ara_dict)
        return cls(ara_cfg=ara_cfg, **config_dict)

    @classmethod
    def from_dict(
        cls: type[SpecificAraPretrainedConfigType], config_dict, **kwargs
    ) -> SpecificAraPretrainedConfigType:
        """
        Load AraPretrainedConfig directly from the dictionary passed in the function.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        ara_dict = config_dict.pop("ara", {})
        ara_cfg = cls.make_ara_config(**ara_dict)

        config = cls(ara_cfg=ara_cfg, **config_dict)

        if return_unused_kwargs:
            return config, kwargs  # pyrefly: ignore=bad-override
        else:
            return config
