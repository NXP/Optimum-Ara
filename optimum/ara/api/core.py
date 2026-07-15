# Copyright (c) 2025, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# SPDX-License-Identifier: Apache-2.0

# dvapi is loaded dynamically in optimum.ara.api.__init__ (see pyproject [tool.pyrefly] ignore-missing-imports)

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

import numpy as np
import torch
import ctypes as c
import dvapi
from dvapi import DVEndpoint, DVModel, DVSession, dv_status_code
from ..configuration_utils import AraConfig, AraPretrainedConfig, InterfaceType
from .exceptions import DvApiException
from .core_configurations import InterfaceType, CoreConfig
from transformers.utils.generic import ModelOutput

np.set_printoptions(threshold=5000)


class Core:
    """
    Base class for all Ara models.

    This class provides the core logic for session and endpoint management,
    and utility functions for inference and generation. It is designed to be extended
    by specific model implementations (e.g., CoreForCausalLM).
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        config: AraConfig,
    ):
        """
        Initialize the Core.

        Args:
            model_path (Union[Path, str]): Path to the model file.
            config (AraConfig): Model configuration.
        """

        self.model_path = (
            Path(model_path) if isinstance(model_path, str) else model_path
        )
        self.model_name = self.model_path.name
        self.__session: DVSession = None
        self.__model: DVModel = None
        self.__endpoint: Optional[List[DVEndpoint]] = None
        self.__vision_model: DVModel = None
        self.infer_type = dvapi.dv_infer_type
        self.llm_endpoint_id: int = 0
        self.vision_endpoint_id: int = 0

        self.create_session(config)
        self.get_endpoint_list()

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def __del__(self):
        if self.__model is not None:
            self.__model.__del__()
        if self.__vision_model is not None:
            self.__vision_model.__del__()
        if self.__session is not None:
            self.__session.__del__()

    def create_session(self, config: AraConfig) -> dv_status_code:
        """
        Create session with endpoint
        Args:
            Config (AraConig): Configuration to create session
        """
        # ToDO: Add code for creating session with named_pipe here.
        if config.interface_type == InterfaceType.SOCKET:
            ret, self.__session = DVSession.create_via_unix_socket(
                config.interface_socket_file
            )
        elif config.interface_type == InterfaceType.IPV4:
            ip = config.interface_ip_address
            port = config.interface_port
            ret, self.__session = DVSession.create_via_tcp_ipv4_socket(ip, port)
        else:
            raise ValueError(f"Unsupported interface type: {config.interface_type}")

        if ret != dvapi.dv_status_code.DV_SUCCESS:
            logging.error("could not establish session")
            raise DvApiException(
                f"Could not create session: {dvapi.dv_stringify_status_code(ret)}"
            )
        return ret

    def get_endpoint_list(self):
        """
        Retrieves the list of endpoints available in the session.

        Returns:
            List[dvapi.dv_endpoint]: List of endpoints.

        Raises:
            DvApiException: If retrieving the endpoint list fails.
        """

        assert self.__session is not None, "session is None, Exit!"

        if self.__endpoint:
            return self.__endpoint

        ret, eplist = self.__session.get_endpoint_list()

        if ret != dv_status_code.DV_SUCCESS:
            logging.error("could not get endpoint list")
            raise DvApiException(
                f"Could not load endpoints: {dvapi.dv_stringify_status_code(ret)}"
            )

        assert eplist is not None
        self.__endpoint = eplist[:]
        return self.__endpoint

    def get_endpoint_dram_stats(self) -> Any:
        if self.__session is None:
            return None
        if not hasattr(self.__session, "get_endpoint_dram_statistics"):
            logging.debug(
                "DVSession.get_endpoint_dram_statistics() is not present! Return None!"
            )
            return None
        status, dram_stats = self.__session.get_endpoint_dram_statistics()
        if status != dv_status_code.DV_SUCCESS:
            logging.warning("Failed to get DRAM stats for Endpoints, returning None")
            return None
        return dram_stats

    def get_llm_endpoint(self) -> DVEndpoint:
        """
        Return endpoint decided for vision model.
        default: 0
        """
        assert (
            self.__endpoint is not None and len(self.__endpoint) > self.llm_endpoint_id
        )
        return self.__endpoint[self.llm_endpoint_id]

    def get_vision_endpoint(self) -> DVEndpoint:
        """
        Return endpoint decided for vision model.
        default: 0
        """
        assert (
            self.__endpoint is not None
            and len(self.__endpoint) > self.vision_endpoint_id
        )
        return self.__endpoint[self.vision_endpoint_id]

    def _set_llm_endpoint_id(self, endpoint_id: int) -> None:
        """
        User will set endpoint thet want to load llm model on.
        default: 0
        """
        logging.debug(f"set llm_endpoint_id to {endpoint_id}")
        assert self.__endpoint is not None and len(self.__endpoint) > endpoint_id
        self.llm_endpoint_id = endpoint_id

    def _set_vision_endpoint_id(self, endpoint_id: int) -> None:
        """
        User will set endpoint thet want to load vision model on.
        default: 0
        """
        logging.debug(f"set vision_endpoint_id to {endpoint_id}")
        assert self.__endpoint is not None and len(self.__endpoint) > endpoint_id
        self.vision_endpoint_id = endpoint_id

    def forward(
        self,
        input_ids: np.ndarray,
        output_ids: np.ndarray,
        active_tokens: int,
        valid_tokens: int,
        infer_type,
        **kwargs,
    ) -> tuple[dv_status_code, c.POINTER(dvapi.dv_infer_request), float]:
        if self.__model is None:
            logging.error("model is not loaded!")
            return dv_status_code.DV_ERROR, None, 0.0

        tokens_to_skip = kwargs.pop("tokens_to_skip", 0)
        # endpoint = self._endpoint()

        input_tensors = [dvapi.DVTensor(input_ids, params=None)]
        output_tensors = [dvapi.DVTensor(output_ids, params=None)]

        infer_options: dvapi.DVInferOptions = dvapi.DVInferOptions.dv_infer_options(
            enable_stats=True,
            infer_type=infer_type,
            active_tokens=active_tokens,
            valid_tokens=valid_tokens,
            tokens_to_skip=tokens_to_skip,
        )

        start_time = time.time()
        status, inf_req = self.__model.infer_sync_with_options(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            endpoint=self.get_llm_endpoint()._endpoint,
            timeout_ms=CoreConfig.TIMEOUT_MS,
            infer_options=infer_options,
        )
        infer_time = time.time() - start_time

        if status != dv_status_code.DV_SUCCESS:
            raise DvApiException(
                "Token generation failed", f"{dvapi.dv_stringify_status_code(status)}"
            )

        return status, inf_req, infer_time

    def process_inference_output(
        self,
        status: dv_status_code,
        output_ids: np.ndarray,
        inf_req: Any,
    ) -> Tuple[dv_status_code, Any]:
        """
        Process inference output and return logits wrapped in ModelOutput on success.
        Returns:
            (status, output_or_inf_req, infer_time)
        """
        if status != dv_status_code.DV_SUCCESS:
            return status, inf_req
        # VLM-specific: extract logits
        vocab_size = self.get_vocab_size()
        logits = output_ids[:vocab_size]
        logits_reshaped = logits.reshape(1, 1, -1).astype(np.float32)
        return (status, ModelOutput(logits=torch.from_numpy(logits_reshaped)))

    @classmethod
    def _llm_options(cls) -> dvapi.DVModelLoadOptions:
        """
        Returns the model load options for LLM models.

        Returns:
            dvapi.DVModelLoadOptions: The model load options.
        """
        return dvapi.DVModelLoadOptions.model_load_option(
            model_name="llm_model",
            priority=dvapi.dv_model_priority_level.DV_MODEL_PRIORITY_LEVEL_DEFAULT,
            cache=False,
            model_async=False,
            model_type=dvapi.dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2,
        )

    def load_model(
        self, model_path: Union[str, Path], endpoint_id: int = 0
    ) -> dv_status_code:
        """
        Loads a model from file using the provided session and endpoint.

        Args:
            model_path (Union[str, Path]): Path to the model file.
        Returns:
            dv_status_code: dv status code

        Raises:
            DvApiException: If model loading fails.
        """

        if self.__vision_model is not None:
            logging.warning("llm model already loaded!")
            logging.warning("return success without loading again!")
            return dvapi.dv_status_code.DV_SUCCESS

        # Check if model_path is not None
        assert model_path is not None, "model_path must not be None"
        if isinstance(model_path, Path):
            model_path = str(model_path)

        self._set_llm_endpoint_id(endpoint_id)

        model_load_options = self._llm_options()
        ret, self.__model = self.__session.load_model_from_file_with_options(
            self.get_llm_endpoint(), model_path, options=model_load_options
        )
        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load model: {dvapi.dv_stringify_status_code(ret)}"
            )
        assert self.__model is not None

        return ret

    def update_llm_params(self, generation_config) -> None:
        """
        Updates the DV model's parameters according to the generation config.

        Args:
            generation_config (AraGenerationConfig): Generation configuration.
        """

        if self.__model is None:
            raise RuntimeError("model is not loaded.")

        llm_cfg_update: dvapi.DVLLMModelCfgUpdateParams = (
            dvapi.DVLLMModelCfgUpdateParams.llm_model_cfg_options(
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                temperature=generation_config.temperature,
                repetition_penalty=generation_config.repetition_penalty,
                target_token_post_mcp=generation_config.ara.target_token_post_mcp,
                target_token_pre_mcp=generation_config.ara.target_token_pre_mcp,
                target_prompt_post_mcp=generation_config.ara.target_prompt_post_mcp,
                target_prompt_pre_mcp=generation_config.ara.target_prompt_pre_mcp,
            )
        )  # the option values to be passed are same as default values of ip args
        ret = dvapi.dv_model_set_llm_cfg_params(
            session=self.__session._session,
            endpoint=self.get_llm_endpoint()._endpoint,
            model=self.__model._model,
            llm_cfg_update=llm_cfg_update._llm_config_options,
        )
        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load LLM params to DV model: {dvapi.dv_stringify_status_code(ret)}"
            )

    def get_device_temperature(self) -> float:
        """
        Get the temperature from endpoint connected
        """
        ep_stats: Any = None
        ep_count: Any = None
        try:
            status, ep_stats, ep_count = dvapi.dv_endpoint_get_statistics(
                self.__session._session, self.get_llm_endpoint()._endpoint
            )

            if status != dvapi.dv_status_code.DV_SUCCESS:
                print(
                    f"Warning: Failed to get device temperature: {dvapi.dv_stringify_status_code(status)}"
                )
                return 0.0

            if ep_count.value > 0:
                temperature = ep_stats[0].ep_temp
                return temperature
            else:
                print("Warning: No endpoints found for temperature reading")
                return 0.0

        except Exception as e:
            print(f"Warning: Error getting device temperature: {e}")
            return 0.0
        finally:
            if ep_stats is not None and ep_count is not None and ep_count.value > 0:
                try:
                    dvapi.dv_endpoint_free_statistics(ep_stats, ep_count.value)
                except Exception as e:
                    print(f"Warning: Error getting device temperature: {e}")
                    pass

    def get_dv_tensor(self, values: np.ndarray, params: Any) -> dvapi.DVTensor:
        """Return dv tensor from ndarray"""

        return dvapi.DVTensor(values, params=params)

    def load_vision_model(
        self, model_path: str, endpoint_id: int = 0
    ) -> dv_status_code:
        """Load vision model from file."""

        if self.__vision_model is not None:
            logging.warning("vision model already loaded!")
            logging.warning("return success without loading again!")
            return dvapi.dv_status_code.DV_SUCCESS

        assert model_path is not None, "model_path must not be None"
        if isinstance(model_path, Path):
            model_path = str(model_path)

        self._set_vision_endpoint_id(endpoint_id)
        ret, self.__vision_model = self.__session.load_model_from_file(
            self.get_vision_endpoint(), model_path
        )

        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load_vision_model: {dvapi.dv_stringify_status_code(ret)}"
            )

        return ret

    def vision_inference(
        self,
        input_tensors: list,
        output_tensors: list,
    ) -> tuple[dv_status_code, Optional[c.POINTER(dvapi.dv_infer_request)]]:
        """
        Perform vision inference using the DV API.

        Args:
            input_tensors (list): A list of input DVTensor objects containing the data to be processed.
            output_tensors (list): A list of output DVTensor objects where the inference results will be stored.

        Raises:
            RuntimeError: If the vision model is not initialized.
            Exception: If inference fails or returns an error.
        """

        ret, inf_req = self.__vision_model.infer_sync(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            endpoint=self.get_vision_endpoint(),
            timeout=CoreConfig.TIMEOUT_MS,
        )

        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load model: {dvapi.dv_stringify_status_code(ret)}"
            )

        return ret, inf_req

    def get_max_num_tokens(self) -> int:
        """
        Returns the max_num_tokens from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Check if the attribute exists
        if not hasattr(self.__model._model.llm_params.contents, "max_num_tokens"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.max_num_tokens"
            )

        try:
            return self.__model._model.llm_params.contents.max_num_tokens
        except Exception as e:
            # Unexpected error while reading value
            raise AttributeError(f"Failed to retrieve max_num_tokens: {e}")

    def get_vocab_size(self) -> int:
        """
        Returns the vocab_size from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if not hasattr(self.__model._model.llm_params.contents, "vocab_size"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.vocab_size"
            )
        try:
            return self.__model._model.llm_params.contents.vocab_size
        except Exception as e:
            # If something unexpected goes wrong while reading it, re-raise clearly
            raise AttributeError(f"Failed to retrieve vocab_size: {e}")

    def get_pad_token_id(self) -> int:
        """
        Returns the pad_token_id from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Explicit attribute check
        if not hasattr(self.__model._model.llm_params.contents, "pad_token_id"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.pad_token_id"
            )

        try:
            return self.__model._model.llm_params.contents.pad_token_id
        except Exception as e:
            # Unexpected read error
            raise AttributeError(f"Failed to retrieve pad_token_id: {e}")

    def get_eos_token_id(self) -> int:
        """
        Returns the eos_token _id from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Explicitly check if the attribute exists
        if not hasattr(self.__model._model.llm_params.contents, "eos_token_id"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.eos_token_id"
            )

        try:
            return self.__model._model.llm_params.contents.eos_token_id
        except Exception as e:
            # Unexpected error while reading the value
            raise AttributeError(f"Failed to retrieve eos_token_id: {e}")

    def is_speculative(self) -> bool:
        """
        Returns the is_speculative flag from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Check for attribute existence
        if not hasattr(self.__model._model.llm_params.contents, "is_speculative"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.is_speculative"
            )

        try:
            return self.__model._model.llm_params.contents.is_speculative
        except Exception as e:
            # Unexpected read error
            raise AttributeError(f"Failed to retrieve is_speculative: {e}")

    def is_host_specd(self) -> bool:
        """
        Returns the is_host_specd flag from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Check attribute existence
        if not hasattr(self.__model._model.llm_params.contents, "is_host_specd"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.is_host_specd"
            )

        try:
            return self.__model._model.llm_params.contents.is_host_specd
        except Exception as e:
            # Unexpected error while accessing the value
            raise AttributeError(f"Failed to retrieve is_host_specd: {e}")

    def get_embedding_size(self) -> int:
        """
        Returns the embedding_size from the model.
        Raises AttributeError if the attribute does not exist.
        """
        # Check attribute existence explicitly
        if not hasattr(self.__model._model.llm_params.contents, "embedding_size"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.embedding_size"
            )

        try:
            return self.__model._model.llm_params.contents.embedding_size
        except Exception as e:
            # If something unexpected goes wrong while reading it, re-raise clearly
            raise AttributeError(f"Failed to retrieve embedding_size: {e}")

    def get_input_precision(self) -> int:
        # Ensure the target attribute exists
        if not hasattr(self.__model._model.llm_params.contents, "input_precision"):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.input_precision"
            )
        d = {}
        d["input_precision"] = getattr(
            self.__model._model.llm_params.contents, "input_precision", 1
        )
        try:
            input_precision = d.get("input_precision", 8)
        except AttributeError as e:
            raise TypeError("Expected 'd' to be a dictionary") from e
        return input_precision

    def get_embedding_lookup_addr(self) -> int:
        """
        Returns the embedding_lookup_addr from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Explicit attribute existence check
        if not hasattr(
            self.__model._model.llm_params.contents, "embedding_lookup_addr"
        ):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.embedding_lookup_addr"
            )

        try:
            return self.__model._model.llm_params.contents.embedding_lookup_addr
        except Exception as e:
            raise AttributeError(f"Failed to retrieve embedding_lookup_addr: {e}")

    def get_scale_lookup_addr(self) -> int:
        """
        Returns the embedding_lookup_scale_addr from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        # Explicit attribute existence check
        if not hasattr(
            self.__model._model.llm_params.contents, "embedding_lookup_scale_addr"
        ):
            raise AttributeError(
                "Missing attribute: self.__model._model.llm_params.contents.embedding_lookup_scale_addr"
            )

        try:
            return self.__model._model.llm_params.contents.embedding_lookup_scale_addr
        except Exception as e:
            raise AttributeError(f"Failed to retrieve embedding_lookup_scale_addr: {e}")

    def print_llm_params(self) -> None:
        print(f"llm_params: {self.__model._model.llm_params.contents}")
