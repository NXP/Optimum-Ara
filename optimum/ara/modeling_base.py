# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

from transformers.utils import ModelOutput
from copy import deepcopy
import logging
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, cast
import time
import os

import numpy as np
import torch
import torch as t
import ctypes as c

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel as HFPreTrainedModel,
)
from transformers.generation import GenerationMixin
from .api.core import Core

from .configuration_utils import AraConfig, AraPretrainedConfig, InterfaceType
from .generation.configuration_utils import AraGenerationConfig, AraHostConfig
from .utils.constants import (
    DVM_DDR_BASE_ADDR,
    ADDITION_SPACE_REQUIRED_TO_LOAD_MODEL,
    DEFAULT_CONFIG_NAME,
    DEFAULT_GENERATION_CONFIG_NAME,
)
from .utils.file_utils import find_files_matching_pattern
from .utils.utils import QEmbedding, human_readable_size


np.set_printoptions(threshold=5000)


class PreTrainedModel(ABC):
    """
    Abstract base class for all pre-trained models in the Ara framework.
    Used for compatibility with HuggingFace pipelines.
    """

    pass


class AraModel(PreTrainedModel):
    """
    Base class for all Ara models.

    This class provides the core logic for session and endpoint management,
    and utility functions for inference and generation. It is designed to be extended
    by specific model implementations (e.g., AraModelForCausalLM).
    """

    model_type = "ara_model"
    auto_model_class = AutoModel
    config_class: type[AraPretrainedConfig] = AraPretrainedConfig
    use_merged: bool = False
    use_fp16: bool = False
    _token_generation_time: float = 0.0
    _num_input_tokens: Optional[torch.Tensor] = None
    _ttft: float = 0.0
    _generated_tokens_count: int = 0

    def __init__(
        self,
        model_path: Union[Path, str],
        config: "AraPretrainedConfig",
        generation_config: "AraGenerationConfig",
        preprocessors: Optional[List] = None,
    ):
        """
        Initialize the AraModel.

        Args:
            model_path (Union[Path, str]): Path to the model file.
            config (AraPretrainedConfig): Model configuration.
            preprocessors (Optional[List]): List of preprocessors to apply.
        """
        super().__init__()
        self._token_generation_time = 0
        self._num_input_tokens: torch.Tensor = torch.tensor(0)
        self._ttft = 0
        self._generated_tokens_count = 0
        self.core = Core(model_path, config.ara)
        self.config = config
        self.generation_config = generation_config
        self.preprocessors = preprocessors
        self.device = "cpu"
        self.embedding: Optional[QEmbedding] = None
        self.embedding_scales: Optional[QEmbedding] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: Optional[np.ndarray] = None,
        input_embeds: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ModelOutput:
        """Subclasses must override. Base implementation is not callable."""
        raise NotImplementedError("AraModel.forward must be overridden by subclass")

    def __del__(self):
        core = getattr(self, "core", None)
        if core is not None:
            core.__del__()

    @classmethod
    def _get_llm_dvm_path(
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        config: Optional["AraPretrainedConfig"] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
    ):
        dvm_file = ""
        if file_name:
            if os.path.exists(file_name) and file_name.endswith(".dvm"):
                dvm_file = Path(file_name)
            else:
                logging.warning("file_name provided should be path to *.dvm file")

        if dvm_file == "" and config:
            if "ara" in config:
                if os.path.exists(config.ara.dvm_path) and str(
                    config.ara.dvm_path
                ).endswith(".dvm"):
                    dvm_file = Path(config.ara.dvm_path)
                else:
                    logging.warning(
                        f"incorrect dvm file path in config : {config.ara.dvm_path}"
                    )
                    logging.warning("dvm_path should is a path to *.dvm file")
            else:
                logging.warning("'ara' is not present in config object")

        if dvm_file == "" and os.path.exists(pretrained_model_name_or_path):
            if str(pretrained_model_name_or_path).endswith(".dvm"):
                dvm_file = Path(pretrained_model_name_or_path)
            else:
                # If it's a directory, search for .dvm files (legacy behavior)
                logging.info(
                    f"Searching for DVM file in Dir : {pretrained_model_name_or_path}"
                )
                dvm_files = find_files_matching_pattern(
                    str(pretrained_model_name_or_path),
                    r".*\.dvm",
                    glob_pattern="**/*.dvm",
                    subfolder=subfolder,
                )
                if len(dvm_files) > 1:
                    logging.warning(
                        f"Found more than one DVM file in Dir : {pretrained_model_name_or_path}"
                    )
                if len(dvm_files) < 1:
                    logging.warning(
                        f"Warn: Didn't found any DVM file in Dir : {pretrained_model_name_or_path}"
                    )
                if len(dvm_files) == 1:
                    dvm_file = dvm_files[0]

        if dvm_file == "":
            raise FileNotFoundError(
                f"Could not find any DVM model file here {pretrained_model_name_or_path}"
            )
        else:
            logging.info(f"Found dvm_file, {dvm_file}")

        dvm_file = dvm_file.resolve()
        return dvm_file

    @classmethod
    def _get_config(
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        config: Optional["AraPretrainedConfig"] = None,
    ):
        if config:
            return config

        model_path = Path(pretrained_model_name_or_path)

        if model_path.is_file():
            return AutoConfig.from_pretrained(model_path)

        if model_path.is_dir():
            config_path = model_path / DEFAULT_CONFIG_NAME
            if config_path.exists():
                return AutoConfig.from_pretrained(config_path)
            logging.warning(
                f"Failed to find {DEFAULT_CONFIG_NAME} file in provided directory path, {pretrained_model_name_or_path}"
            )
        else:
            logging.warning(
                f"Failed to load config, provided str is not a directory path, {pretrained_model_name_or_path}"
            )

        # Note: Following code can return a config object which is not of type AraPretrainedConfig which can cause other issues.
        # thats why disabling following code for now.
        # try:
        #     if not model_path.exists():
        #         logging.info(
        #             f"Try downloading {DEFAULT_CONFIG_NAME} from hugging_face hub for model_id, {pretrained_model_name_or_path}"
        #         )
        #         return AutoConfig.from_pretrained(pretrained_model_name_or_path)  # pyrefly: ignore=bad-override
        # except OSError as e:
        #     logging.warning(
        #         f"Failed to load config due to OS error: {e}. Creating default config."
        #     )

        logging.info("create default config object")
        cfg_cls = cls.config_class
        return cfg_cls(cfg_cls.make_ara_config())

    @classmethod
    def _get_generation_config(
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        model_config: "AraPretrainedConfig",
        config: Optional["AraGenerationConfig"] = None,
    ):
        if config:
            return config

        model_path = Path(pretrained_model_name_or_path)
        logging.debug(f"Reading generation config from following path {model_path=}")

        if model_path.is_file() and model_path.suffix == ".json":
            return AraGenerationConfig.from_pretrained(model_path)

        if model_path.is_dir():
            config_path = model_path / DEFAULT_GENERATION_CONFIG_NAME
            if config_path.exists():
                return AraGenerationConfig.from_pretrained(config_path)
            logging.warning(
                f"Failed to find {DEFAULT_GENERATION_CONFIG_NAME} file in provided "
                f"directory path, {pretrained_model_name_or_path}"
            )
        else:
            logging.warning(
                f"Failed to load config, provided str is not a directory path, "
                f"{pretrained_model_name_or_path}"
            )

        try:
            if not model_path.exists():
                logging.info(
                    f"Try downloading {DEFAULT_GENERATION_CONFIG_NAME} "
                    f"from hugging_face hub for model_id, "
                    f"{pretrained_model_name_or_path}"
                )
                return AraGenerationConfig.from_pretrained(
                    pretrained_model_name_or_path
                )
        except OSError as e:
            logging.warning(
                f"Failed to load config due to OS error: {e}. Creating default config."
            )

        logging.info("Creating default generation config object from model config")
        return AraGenerationConfig.from_model_config(model_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, os.PathLike],
        config: Optional[Union["PretrainedConfig", AraPretrainedConfig]] = None,
        *args,
        **kwargs,
    ) -> "AraModel":
        raise NotImplementedError

    def _prepare_output_buffer(self, model, output_shape: Tuple[int], output_name: str):
        raise NotImplementedError

    def _output_shape_inference(
        self, axis_name: Union[str, int], dimensions: Dict[str, int]
    ) -> Union[str, int]:
        raise NotImplementedError

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        return isinstance(self, GenerationMixin)

    def display_perf_statistics(self):
        """
        print performance statistics

        Returns:
            None
        """
        token_rate = 0
        if self._num_input_tokens is None:
            num_tokens = 0
        elif isinstance(self._num_input_tokens, torch.Tensor):
            num_tokens = self._num_input_tokens.numel()  # total elements
        else:
            num_tokens = self._num_input_tokens.size  # numpy array

        if self._token_generation_time > 0 and self._generated_tokens_count > 0:
            token_rate = (
                self._generated_tokens_count - 1
            ) / self._token_generation_time

        print("############### Performance Statistics #################")
        print("Time to first token (TTFT)     :", str(round(self._ttft, 2)) + " sec")
        print(
            "Generation time (without TTFT) :",
            str(round(self._token_generation_time, 2)) + " sec",
        )
        print(
            "Number of input tokens         :",
            num_tokens,
            " tokens",
        )
        print(
            "Number of tokens generated     :",
            str(round(self._generated_tokens_count, 2)) + " tokens",
        )
        print(
            "token rate                     :",
            str(round(token_rate, 2)) + " tokens/sec",
        )
        print(
            "temperature                    :",
            str(round(self.core.get_device_temperature(), 2)) + " C",
        )
        print("\n")

    def _prepare_generation_config(  # type: ignore
        self,
        generation_config: Optional[
            Union[GenerationConfig, AraGenerationConfig]
        ] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[AraGenerationConfig, Dict[str, Any]]:
        """
        Updates the generation configuration according to generation config or kwargs sent by the user

        Args:
            generation_config (Optional[AraGenerationConfig]): Generation configuration.
            use_model_defaults (Optional[bool]): Whether to use model defaults.
            **kwargs: Additional generation parameters.

        Returns:
            AraGenerationConfig: Updated generation configuration.
        """
        # Ensure we always have a concrete AraGenerationConfig instance to return.
        if generation_config is None:
            gen_config = self.generation_config
        else:
            if isinstance(generation_config, AraGenerationConfig):
                gen_config = deepcopy(generation_config)
            else:
                # If it's a base GenerationConfig, wrap it into AraGenerationConfig
                gen_config = AraGenerationConfig.from_model_config(self.config)
                for key, value in generation_config.to_dict().items():
                    if hasattr(gen_config, key):
                        setattr(gen_config, key, value)

        for key, value in kwargs.items():
            if key in AraHostConfig.model_fields.keys():
                setattr(gen_config.ara, key, value)
            elif hasattr(gen_config, key):
                setattr(gen_config, key, value)
        return gen_config, kwargs

    def prepare_past_key_values(
        self,
        input_ids: np.ndarray,
        past_key_values: Union[None, tuple[tuple[np.ndarray]]],
        use_torch: bool = False,
    ):
        """Prepare (or synthesize) past key/value cache structures for generation.

        Many Ara backends embed KV-cache handling inside device kernels or do
        not expose the full set of input/output names. This helper returns a
        tuple describing whether the model should follow the cached-path and a
        suitable past_key_values structure for use by decoding routines.

        Args:
            input_ids (np.ndarray): Input token ids shaped (batch, seq_len).
            past_key_values (Optional[tuple[tuple[np.ndarray]]]): Existing
                past key/value tensors if any. If None this function may
                synthesize zero-length past arrays compatible with the model.
            use_torch (bool): If True prefer torch tensors for returned
                structures; otherwise NumPy arrays are used.

        Returns:
            tuple: (use_cache_branch, past_key_values, pkv_output_shape)
                - use_cache_branch: a small indicator/flag used by callers to
                  choose a merged-decoder branch when supported.
                - past_key_values: the possibly-modified past key/value cache
                  structure compatible with model decoding.
                - pkv_output_shape: a mapping from output-name to the
                  expected output shape when the model will append new
                  tokens to the cache.
        """

        pass

    def _handle_device_map(
        self,
        model_path: Union[str, Path],
        device_map: Optional[
            Union[str, dict[str, Union[int, str, torch.device]], int, torch.device]
        ],
        max_memory: Optional[dict],
    ) -> int:
        """
        handle device_map and decides which endpoint to load model on,
        based on device map.
        Args:
            model_path Union[str, Path]: model_path, used to check model side.
            device_map :
                None: use "auto"
                "auto": check all endpoints one by one for available dram
                <int> | "ara:<int>": load model on provided endpoint, space will not be checked in this case
            max_memory: not support yet # TODO

        Returns:
            int: endpoint id
        """
        logging.debug(f"_handle_device_map called with device_map : {device_map}")
        endpoint = None
        if device_map is None:
            device_map = "auto"

        # device_map dict, torch.device for layer level control not supported
        if type(device_map) in [None, dict, torch.device]:
            logging.warning(f"device map : {device_map} , not supported")
            device_map = "auto"
            logging.debug(f"Update device map is : {device_map}")

        # Handle device_map = "", "0", "cuda" etc
        if type(device_map) is str and device_map != "auto":
            if "ara:" not in device_map:
                logging.warning(
                    f"device map : {device_map} , not supported. Using auto"
                )
                device_map = "auto"
                logging.debug(f"Update device map is : {device_map}")
            elif "ara:" == device_map:
                device_map = "auto"

        # Convert "ara:<int>" to <int>
        if type(device_map) is str:
            if "ara:" in device_map:
                device_map = int(device_map.split(":")[1])
                logging.debug(
                    f"Update device map is : {device_map}, type : {type(device_map)}"
                )

        # Handle device map = "auto"
        if device_map == "auto":
            dram_stats = self.core.get_endpoint_dram_stats()
            if dram_stats is None:
                logging.warning(
                    "Failed to get DRAM stats for Endpoint, Using default Endpoint 0"
                )
                endpoint = 0
            else:
                model_size_in_bytes = os.path.getsize(model_path)
                total_memory_required = (
                    model_size_in_bytes + ADDITION_SPACE_REQUIRED_TO_LOAD_MODEL
                )
                logging.info(
                    f"Model size: {human_readable_size(model_size_in_bytes)}, Additional required memory: {human_readable_size(ADDITION_SPACE_REQUIRED_TO_LOAD_MODEL)}"
                )
                logging.info(
                    f"Total size required for Model: {human_readable_size(total_memory_required)}"
                )
                for index in range(len(dram_stats)):
                    logging.info(
                        f"Endpoint : {index} has free space : {human_readable_size(dram_stats[index].ep_total_free_size)}"
                    )
                    if dram_stats[index].ep_total_free_size > total_memory_required:
                        endpoint = index
                        break
                if endpoint is None:
                    logging.warning("No Available Endpoint has enough free space.")

        # Handle device map of type int
        if type(device_map) is int:
            if device_map < 0:
                logging.warning(f"device specified : {device_map} , not available")
                device_map = 0
            dram_stats = self.core.get_endpoint_dram_stats()
            if dram_stats is None:
                logging.warning("Failed to get DRAM stats for Endpoint")
                ep_list = self.core.get_endpoint_list()
                endpoint = device_map
                if device_map >= len(ep_list):
                    logging.info(
                        f"provided device_map {device_map} is not supported, max device_map supported is upto {len(ep_list) - 1}"
                    )
                    logging.info("setting device_map to default 0")
                    endpoint = 0
            else:
                model_size_in_bytes = os.path.getsize(model_path)
                total_memory_required = (
                    model_size_in_bytes + ADDITION_SPACE_REQUIRED_TO_LOAD_MODEL
                )
                logging.info(
                    f"Model size: {human_readable_size(model_size_in_bytes)}, Additional required memory: {human_readable_size(ADDITION_SPACE_REQUIRED_TO_LOAD_MODEL)}"
                )
                logging.info(
                    f"Total size required for Model: {human_readable_size(total_memory_required)}"
                )
                if device_map < len(dram_stats):
                    logging.info(
                        f"Endpoint : {device_map} has free space : {human_readable_size(dram_stats[device_map].ep_total_free_size)}"
                    )
                    if (
                        dram_stats[device_map].ep_total_free_size
                        > total_memory_required
                    ):
                        endpoint = device_map
                    else:
                        logging.warning(
                            f"Provided Endpoint:{device_map} Don't have enough free space."
                        )

        assert endpoint is not None

        return endpoint

    def load_model(
        self,
        path: Union[str, Path],
        device_map: Optional[
            Union[str, dict[str, Union[int, str, torch.device]], int, torch.device]
        ],
        max_memory: Optional[dict],
    ):
        """
        Instantiates an Ara inference session and loads a model using a .dvm file.

        Args:
            path (Union[str, Path]): Path to the model file.
            config (AraConfig): Model configuration.
            device_map: In case of multiple endpoint, specify which endpoint to load the model on.
            max_memory: <comming soon>, specify memory limit on individual endpoints
        Returns:
            tuple[DVSession, list[DVEndpoint], DVModel]: The session, endpoint, and loaded model.

        Raises:
            ValueError: If the interface type is unsupported.
            DvApiException: If session or endpoint creation fails.
        """

        endpoint_id = self._handle_device_map(path, device_map, max_memory)
        self.core.load_model(path, endpoint_id)

        self.core.get_endpoint_dram_stats()

    def to(self, device):
        """
        Workaround to bypass setting model device in pipeline
        """
        return self

    def _load_embedding_table(self, dvm_path: Union[str, Path]):
        """Load embedding table from DVM file using file-based approach (counterpart to C++ code)"""
        # Get the DVM file path from config
        if not dvm_path:
            raise RuntimeError("DVM file path not found in config")

        # Create embedding table
        # cls.embedding = t.nn.Embedding(
        #     llm_params['vocab_size'], llm_params['embedding_size']
        # )

        # Load embedding data from DVM file (counterpart to C++ load_embedding_lookup_from_model_dvm)
        self._load_embedding_lookup_from_model_dvm(dvm_path)

    def _load_embedding_lookup_from_model_dvm(self, dvm_path):
        """Load embedding lookup from model DVM file (counterpart to C++ function)"""
        # Convert device memory address to file offset by subtracting DVM_DDR_BASE_ADDR
        # This matches the C++ code: embedding_lookup_addr = llm_params->embedding_lookup_addr - AraCore::DVM_DDR_BASE_ADDR
        embedding_lookup_addr = (
            self.core.get_embedding_lookup_addr() - DVM_DDR_BASE_ADDR
        )
        scale_lookup_addr = self.core.get_scale_lookup_addr() - DVM_DDR_BASE_ADDR
        vocab_size = self.core.get_vocab_size()
        embedding_size = self.core.get_embedding_size()
        # Calculate table size: vocab_size * embedding_size * bytes_per_element
        # input_precision is in bits, so divide by 8 to get bytes per element
        input_precision = self.core.get_input_precision()
        bytes_per_element = input_precision // 8
        table_size_bytes = vocab_size * embedding_size * bytes_per_element

        # Calculate scales size: (vocab_size * embedding_size) / 64
        # Scales are one int8 value per group of 64 elements
        scale_size_bytes = vocab_size * embedding_size // 64

        logging.debug(
            f"Loading embedding table: addr={embedding_lookup_addr}, size={table_size_bytes}"
        )
        logging.debug(
            f"Loading embedding scales: addr={scale_lookup_addr}, size={scale_size_bytes}"
        )
        logging.debug(
            f"LLM params: vocab_size={vocab_size}, embedding_size={embedding_size}, input_precision={self.core.get_input_precision()}"
        )
        logging.debug(
            f"Raw LLM params: embedding_lookup_addr={self.core.get_embedding_lookup_addr()}, embedding_lookup_scale_addr={self.core.get_scale_lookup_addr()}"
        )

        # Check file size
        file_size = os.path.getsize(dvm_path)
        logging.debug(f"DVM file size: {file_size} bytes")
        logging.debug(
            f"Requested table offset: {embedding_lookup_addr}, size: {table_size_bytes}"
        )
        logging.debug(
            f"Requested scales offset: {scale_lookup_addr}, size: {scale_size_bytes}"
        )

        # Load embedding table from file
        table_raw = self._load_from_file_with_offset_size(
            dvm_path, embedding_lookup_addr, table_size_bytes
        )
        if table_raw is None:
            raise RuntimeError("Failed to read embedding data from DVM file")

        # Load embedding scales from file
        scales_raw = self._load_from_file_with_offset_size(
            dvm_path, scale_lookup_addr, scale_size_bytes
        )
        if scales_raw is None:
            raise RuntimeError("Failed to read embedding scales from DVM file")

        # Keep raw quantized table and scales for get_embeddings
        embedding_lookup_quantized = t.frombuffer(table_raw.copy(), dtype=t.int8)
        embedding_lookup_scales = t.frombuffer(scales_raw.copy(), dtype=t.int8)

        # Convert quantized embedding lookup to torch tensor and reshape
        embedding_data = embedding_lookup_quantized.reshape(vocab_size, embedding_size)
        embedding_scales_data = embedding_lookup_scales.reshape(
            vocab_size, embedding_size // 64
        )

        self.embedding = QEmbedding(embedding_data)

        self.embedding_scales = QEmbedding(embedding_scales_data)

    def _load_from_file_with_offset_size(self, file_path, offset, read_size):
        """Load from file with offset and size (counterpart to C++ function)"""

        # Open file in binary mode
        try:
            with open(file_path, "rb") as in_file:
                # Get file size
                in_file.seek(0, os.SEEK_END)
                file_size = in_file.tell()

                # Handle read_size = -1 (read entire file)
                if read_size == -1:
                    read_size = file_size

                # Check bounds
                if file_size < offset + read_size:
                    logging.error(
                        f"Provided file {file_path} doesn't have embedding lookup of size {read_size} from position {offset}"
                    )
                    return None

                # Seek to offset
                in_file.seek(offset, os.SEEK_SET)

                # Read data
                buffer = bytearray(in_file.read(read_size))

                return buffer

        except Exception as e:
            logging.error(f"Reading file {file_path}: {e}")
            return None
