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
from dvapi import (
    DVEndpoint,
    DVModel,
    DVSession,
    DVTensor,
    DVInferOptions,
    dv_status_code,
    dv_stringify_status_code,
)
from dvapi import (
    dv_model_type,
    dv_model_llm_params,
    dv_infer_type,
    dv_get_client_lib_version,
    dv_version,
    dv_product_type,
)
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
        self.__endpoint_list: Optional[List[DVEndpoint]] = None
        self.__vision_model: DVModel = None
        self.infer_type = dv_infer_type
        self.llm_endpoint_ids: List[int] = [0]
        self.llm_endpoint_group: Optional[DVEndpoint] = None
        self.vision_endpoint_id: int = 0
        self.llm_params: dv_model_llm_params = None
        self.model_type: dv_model_type = dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2
        self.dv_version: dv_version = dv_get_client_lib_version()

        logging.info(
            f"ClientLib Version:{self.dv_version.major}.{self.dv_version.minor}.{self.dv_version.patch}.{self.dv_version.patch_minor}"
        )

        self.create_session(config)
        self._check_client_lib_version_compatibility()
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

    def _check_client_lib_version_compatibility(self):
        """Check if the current client library version is supported by the proxy."""
        if not hasattr(dvapi, "dv_retrieve_version_details"):
            logging.warning(
                "dv_retrieve_version_details is not available in dvapi; "
                "skipping client-lib version compatibility check"
            )
            return

        version_details = dvapi.dv_retrieve_version_details(self.__session._session)
        if version_details is None:
            logging.warning(
                "dv_retrieve_version_details returned None; "
                "skipping client-lib version compatibility check"
            )
            return

        try:
            product_versions, product_count = version_details
            product_count = product_count.value
            supported_client_lib_versions = [
                product_versions[i].product_version
                for i in range(product_count)
                if product_versions[i].product_type == dv_product_type.CLIENT_LIB
            ]

            client_ver = (
                f"{self.dv_version.major}.{self.dv_version.minor}."
                f"{self.dv_version.patch}.{self.dv_version.patch_minor}"
            )

            proxy_ver = None
            current_details = dvapi.dv_exchange_current_version_details(
                self.__session._session
            )
            if current_details is not None:
                curr_versions, curr_count = current_details
                curr_count = curr_count.value
                try:
                    for i in range(curr_count):
                        if curr_versions[i].product_type == dv_product_type.PROXY:
                            pv = curr_versions[i].product_version
                            proxy_ver = (
                                f"{pv.major}.{pv.minor}.{pv.patch}.{pv.patch_minor}"
                            )
                            break
                finally:
                    _curr_free = dvapi._dvApiObj().dv_free_version_details
                    _curr_free.argtypes = [type(curr_versions)]
                    _curr_free.restype = dvapi.dv_status_code
                    _curr_free(curr_versions)

            supported_llm_versions = [
                f"{product_versions[i].product_version.major}."
                f"{product_versions[i].product_version.minor}."
                f"{product_versions[i].product_version.patch}."
                f"{product_versions[i].product_version.patch_minor}"
                for i in range(product_count)
                if product_versions[i].product_type == dv_product_type.LLM_MODEL
            ]

            if proxy_ver:
                logging.info(
                    f"Proxy version: {proxy_ver} | ClientLib version: {client_ver}"
                )
            else:
                logging.info(f"ClientLib version: {client_ver}")

            if supported_llm_versions:
                logging.info(f"Supported LLM model versions: {supported_llm_versions}")

            if hasattr(dv_product_type, "MCS_LLM_MODEL"):
                supported_mcs_llm_versions = [
                    f"{product_versions[i].product_version.major}."
                    f"{product_versions[i].product_version.minor}."
                    f"{product_versions[i].product_version.patch}."
                    f"{product_versions[i].product_version.patch_minor}"
                    for i in range(product_count)
                    if product_versions[i].product_type == dv_product_type.MCS_LLM_MODEL
                ]

                if supported_mcs_llm_versions:
                    logging.info(
                        f"Supported MCS LLM model versions: {supported_mcs_llm_versions}"
                    )

            if not supported_client_lib_versions:
                logging.warning(
                    "No CLIENT_LIB versions in supported list; "
                    "skipping client-lib version compatibility check"
                )
                return

            matches = False
            for sv in supported_client_lib_versions:
                if (
                    self.dv_version.major == sv.major
                    and self.dv_version.minor == sv.minor
                    and self.dv_version.patch == sv.patch
                    and self.dv_version.patch_minor == sv.patch_minor
                ):
                    matches = True
                    break

            if matches:
                logging.info(
                    f"ClientLib version {client_ver} is supported by the proxy"
                )
            else:
                supported_strs = [
                    f"{v.major}.{v.minor}.{v.patch}.{v.patch_minor}"
                    for v in supported_client_lib_versions
                ]
                raise DvApiException(
                    f"ClientLib version {client_ver} is not supported by "
                    f"the proxy. Supported versions: {supported_strs}"
                )
        finally:
            _free = dvapi._dvApiObj().dv_free_version_details
            _free.argtypes = [type(product_versions)]
            _free.restype = dvapi.dv_status_code
            _free(product_versions)

    def create_session(self, config: AraConfig) -> dv_status_code:
        """
        Create session with endpoint
        Args:
            Config (AraConig): Configuration to create session
        """

        logging.info(f"Interface type provided : {config.interface_type}")

        if (
            config.interface_type != InterfaceType.SOCKET
            and config.interface_type != InterfaceType.IPV4
        ):
            raise ValueError(f"Unsupported interface type: {config.interface_type}")

        # ToDO: Add code for creating session with named_pipe here.

        ret = None
        tried_socket = False
        tried_ipv4 = False

        while True:
            if config.interface_type == InterfaceType.SOCKET:
                ret, self.__session = DVSession.create_via_unix_socket(
                    config.interface_socket_file
                )
                tried_socket = True

            if config.interface_type == InterfaceType.IPV4:
                ip = config.interface_ip_address
                port = config.interface_port
                ret, self.__session = DVSession.create_via_tcp_ipv4_socket(ip, port)
                tried_ipv4 = True

            if ret != dv_status_code.DV_SUCCESS:
                logging.warning(
                    f"could not establish session using {config.interface_type}"
                )
                if tried_socket == True and tried_ipv4 == False:
                    config.interface_type = InterfaceType.IPV4
                elif tried_socket == False and tried_ipv4 == True:
                    config.interface_type = InterfaceType.SOCKET
                else:
                    logging.error("could not establish session")
                    raise DvApiException(
                        f"Could not create session: {dv_stringify_status_code(ret)}"
                    )
                logging.warning(
                    f"try connection with interface_type {config.interface_type}"
                )
            else:
                logging.info(
                    f"Session created with interface_type {config.interface_type}"
                )
                break
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

        if self.__endpoint_list:
            return self.__endpoint_list

        ret, eplist = self.__session.get_endpoint_list()

        if ret != dv_status_code.DV_SUCCESS:
            logging.error("could not get endpoint list")
            raise DvApiException(
                f"Could not load endpoints: {dv_stringify_status_code(ret)}"
            )

        assert eplist is not None
        self.__endpoint_list = eplist[:]
        return self.__endpoint_list

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
        # assert self.__endpoint_list is not None and len(self.__endpoint_list) >= len(self.llm_endpoint_ids)
        assert self.llm_endpoint_group is not None
        return self.llm_endpoint_group

    def get_vision_endpoint(self) -> DVEndpoint:
        """
        Return endpoint decided for vision model.
        default: 0
        """
        assert (
            self.__endpoint_list is not None
            and len(self.__endpoint_list) > self.vision_endpoint_id
        )
        return self.__endpoint_list[self.vision_endpoint_id]

    def _set_llm_endpoint_id(self, endpoint_ids: List[int]) -> None:
        """
        User will set endpoint thet want to load llm model on.
        default: 0
        """
        logging.debug(f"set llm_endpoint_ids to {endpoint_ids}")
        assert self.__endpoint_list is not None and len(self.__endpoint_list) >= len(
            endpoint_ids
        )
        self.llm_endpoint_ids = endpoint_ids
        endpoint_list = [self.__endpoint_list[i] for i in endpoint_ids]
        # create endpoint group
        if hasattr(DVSession, "endpoint_create_group") and callable(
            getattr(DVSession, "endpoint_create_group")
        ):
            self.llm_endpoint_group = self.__session.endpoint_create_group(
                endpoint_list
            )
        else:
            ep_list_bare = [e._endpoint for e in endpoint_list]

            ret, ep_grp = dvapi.dv_endpoint_create_group(
                self.__session._session, ep_list_bare, len(ep_list_bare)
            )
            assert ret is dv_status_code.DV_SUCCESS
            self.llm_endpoint_group = DVEndpoint(ep_grp[0])

    def _set_vision_endpoint_id(self, endpoint_id: list[int]) -> None:
        """
        User will set endpoint thet want to load vision model on.
        default: 0
        """
        logging.debug(f"set vision_endpoint_id to {endpoint_id}")
        assert (
            self.__endpoint_list is not None
            and len(self.__endpoint_list) > endpoint_id[0]
        )
        self.vision_endpoint_id = endpoint_id[0]

    def get_infer_type(self, infer_type: dv_infer_type) -> dv_infer_type:
        # handle infer type based on model type
        if "DV_MODEL_TYPE_MCS_LLM" in dv_model_type.__members__:
            if self.model_type == dv_model_type.DV_MODEL_TYPE_MCS_LLM:
                if (
                    infer_type
                    == dv_infer_type.DV_INFER_TYPE_LLM_FOLLOWUP_PROMPT_PROCESSING
                    or infer_type == dv_infer_type.DV_INFER_TYPE_LLM_PROMPT_PROCESSING
                ):
                    infer_type = dv_infer_type.DV_INFER_TYPE_MCS_LLM_PROMPT_PROCESSING
                elif infer_type == dv_infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION:
                    infer_type = dv_infer_type.DV_INFER_TYPE_MCS_LLM_TOKEN_GENERATION
        return infer_type

    def forward(
        self,
        input_ids: np.ndarray,
        output_ids: np.ndarray,
        active_tokens: int,
        valid_tokens: int,
        infer_type,
        **kwargs,
    ) -> tuple[dv_status_code, dvapi.dv_infer_request, float]:
        if self.__model is None:
            logging.error("model is not loaded!")
            return dv_status_code.DV_ERROR, None, 0.0

        tokens_to_skip = kwargs.pop("tokens_to_skip", 0)
        # endpoint = self._endpoint()

        input_tensors = [DVTensor(input_ids, params=None)]
        output_tensors = [DVTensor(output_ids, params=None)]

        infer_type = self.get_infer_type(infer_type)

        infer_options: DVInferOptions = DVInferOptions.dv_infer_options(
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
                "Token generation failed", f"{dv_stringify_status_code(status)}"
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

    def model_get_parameters_from_file(self, model_path: str):
        logging.debug(
            f"model_get_parameters_from_file called, with model_path : {model_path}"
        )
        ret = dv_status_code.DV_FAILURE_UNKOWN
        if (
            self.dv_version.major <= 1
            and self.dv_version.minor <= 3
            and self.dv_version.patch <= 2
        ):
            logging.debug(
                f"Current Client Lib version doesn't support parameter from file function!"
            )
            return self.llm_params
        try:
            ret, model = DVModel.get_parameters_from_file(model_path)
            if ret == dv_status_code.DV_SUCCESS:
                self.llm_params = model._model.llm_params.contents
                self.model_type = model.model_type
                self.__model = model
            if ret == dv_status_code.DV_MODEL_UNSUPPORTED_VERSION:
                logging.error(
                    "Model DVM version is not supported by client_lib and proxy version."
                )
                raise DvApiException(
                    f"Model DVM Version not supported : {dv_stringify_status_code(ret)}"
                )
        except Exception as e:
            print(f"Warning: Exception getting params from file: {e}")
        if ret != dv_status_code.DV_SUCCESS:
            logging.warning(
                f"failed to read model params from model_file: {model_path}"
            )
        return self.llm_params

    def _model_load_options(self, model_path: str) -> dvapi.DVModelLoadOptions:
        """
        Returns the model load options for LLM models.

        Returns:
            dvapi.DVModelLoadOptions: The model load options.
        """

        if self.model_type == None:
            self.model_get_parameters_from_file(model_path)

        return dvapi.DVModelLoadOptions.model_load_option(
            model_name="llm_model",
            priority=dvapi.dv_model_priority_level.DV_MODEL_PRIORITY_LEVEL_DEFAULT,
            cache=False,
            model_async=False,
            model_type=self.model_type,
        )

    def load_model(
        self, model_path: Union[str, Path], endpoint_id: List[int] = [0]
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
            return dv_status_code.DV_SUCCESS

        # Check if model_path is not None
        assert model_path is not None, "model_path must not be None"
        if isinstance(model_path, Path):
            model_path = str(model_path)

        self._set_llm_endpoint_id(endpoint_id)

        model_load_options = self._model_load_options(model_path)

        ret, self.__model = self.__session.load_model_from_file_with_options(
            self.get_llm_endpoint(), model_path, options=model_load_options
        )
        if ret != dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load model: {dv_stringify_status_code(ret)}"
            )
        assert self.__model is not None

        self.llm_params = self.__model._model.llm_params.contents
        self.model_type = self.__model.model_type

        return ret

    def update_llm_params(self, generation_config) -> None:
        """
        Updates the DV model's parameters according to the generation config.

        Args:
            generation_config (AraGenerationConfig): Generation configuration.
        """

        if self.__model is None:
            raise RuntimeError("model is not loaded.")

        # dv_model_set_llm_cfg_params function failing for MCS Model
        if "DV_MODEL_TYPE_MCS_LLM" in dv_model_type.__members__:
            if self.model_type == dv_model_type.DV_MODEL_TYPE_MCS_LLM:
                logging.debug("skip update_llm_params for DV_MODEL_TYPE_MCS_LLM")
                return

        if generation_config.top_k > self.get_vocab_size():
            logging.warning(
                f"top_k should between [1, {self.get_vocab_size()}], max limit is vocab size of the model."
            )
            logging.warning(f"Updating top_k to {self.get_vocab_size()}.")
            generation_config.top_k = self.get_vocab_size()

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
        if ret != dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load LLM params to DV model: {dv_stringify_status_code(ret)}"
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

            if status != dv_status_code.DV_SUCCESS:
                print(
                    f"Warning: Failed to get device temperature: {dv_stringify_status_code(status)}"
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
        if hasattr(values, "cpu"):
            values = values.cpu().detach().numpy()
        return dvapi.DVTensor(values, params=params)

    def load_vision_model(
        self, model_path: str, endpoint_id: list[int] = [0]
    ) -> dv_status_code:
        """Load vision model from file."""

        if self.__vision_model is not None:
            logging.warning("vision model already loaded!")
            logging.warning("return success without loading again!")
            return dv_status_code.DV_SUCCESS

        assert model_path is not None, "model_path must not be None"
        if isinstance(model_path, Path):
            model_path = str(model_path)

        self._set_vision_endpoint_id(endpoint_id)
        ret, self.__vision_model = self.__session.load_model_from_file(
            self.get_vision_endpoint(), model_path
        )

        if ret != dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load_vision_model: {dv_stringify_status_code(ret)}"
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

        if ret != dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load model: {dv_stringify_status_code(ret)}"
            )

        return ret, inf_req

    def get_max_num_tokens(self) -> int:
        """
        Returns the max_num_tokens from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Check if the attribute exists
        if not hasattr(self.llm_params, "max_num_tokens"):
            raise AttributeError("Missing attribute: self.llm_params.max_num_tokens")

        try:
            return self.llm_params.max_num_tokens
        except Exception as e:
            # Unexpected error while reading value
            raise AttributeError(f"Failed to retrieve max_num_tokens: {e}")

    def get_vocab_size(self) -> int:
        """
        Returns the vocab_size from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        if not hasattr(self.llm_params, "vocab_size"):
            raise AttributeError("Missing attribute: self.llm_params.vocab_size")
        try:
            return self.llm_params.vocab_size
        except Exception as e:
            # If something unexpected goes wrong while reading it, re-raise clearly
            raise AttributeError(f"Failed to retrieve vocab_size: {e}")

    def get_pad_token_id(self) -> int:
        """
        Returns the pad_token_id from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Explicit attribute check
        if not hasattr(self.llm_params, "pad_token_id"):
            raise AttributeError("Missing attribute: self.llm_params.pad_token_id")

        try:
            return self.llm_params.pad_token_id
        except Exception as e:
            # Unexpected read error
            raise AttributeError(f"Failed to retrieve pad_token_id: {e}")

    def get_eos_token_id(self) -> int:
        """
        Returns the eos_token _id from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Explicitly check if the attribute exists
        if not hasattr(self.llm_params, "eos_token_id"):
            raise AttributeError("Missing attribute: self.llm_params.eos_token_id")

        try:
            return self.llm_params.eos_token_id
        except Exception as e:
            # Unexpected error while reading the value
            raise AttributeError(f"Failed to retrieve eos_token_id: {e}")

    def is_speculative(self) -> bool:
        """
        Returns the is_speculative flag from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Check for attribute existence
        if not hasattr(self.llm_params, "is_speculative"):
            raise AttributeError("Missing attribute: self.llm_params.is_speculative")

        try:
            return self.llm_params.is_speculative
        except Exception as e:
            # Unexpected read error
            raise AttributeError(f"Failed to retrieve is_speculative: {e}")

    def is_host_specd(self) -> bool:
        """
        Returns the is_host_specd flag from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Check attribute existence
        if not hasattr(self.llm_params, "is_host_specd"):
            raise AttributeError("Missing attribute: self.llm_params.is_host_specd")

        try:
            return self.llm_params.is_host_specd
        except Exception as e:
            # Unexpected error while accessing the value
            raise AttributeError(f"Failed to retrieve is_host_specd: {e}")

    def get_embedding_size(self) -> int:
        """
        Returns the embedding_size from the model.
        Raises AttributeError if the attribute does not exist.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Check attribute existence explicitly
        if not hasattr(self.llm_params, "embedding_size"):
            raise AttributeError("Missing attribute: self.llm_params.embedding_size")

        try:
            return self.llm_params.embedding_size
        except Exception as e:
            # If something unexpected goes wrong while reading it, re-raise clearly
            raise AttributeError(f"Failed to retrieve embedding_size: {e}")

    def get_input_precision(self) -> int:

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Ensure the target attribute exists
        if not hasattr(self.llm_params, "input_precision"):
            raise AttributeError("Missing attribute: self.llm_params.input_precision")
        d = {}
        d["input_precision"] = getattr(self.llm_params, "input_precision", 1)
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

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Explicit attribute existence check
        if not hasattr(self.llm_params, "embedding_lookup_addr"):
            raise AttributeError(
                "Missing attribute: self.llm_params.embedding_lookup_addr"
            )

        try:
            return self.llm_params.embedding_lookup_addr
        except Exception as e:
            raise AttributeError(f"Failed to retrieve embedding_lookup_addr: {e}")

    def get_scale_lookup_addr(self) -> int:
        """
        Returns the embedding_lookup_scale_addr from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Explicit attribute existence check
        if not hasattr(self.llm_params, "embedding_lookup_scale_addr"):
            raise AttributeError(
                "Missing attribute: self.llm_params.embedding_lookup_scale_addr"
            )

        try:
            return self.llm_params.embedding_lookup_scale_addr
        except Exception as e:
            raise AttributeError(f"Failed to retrieve embedding_lookup_scale_addr: {e}")

    def get_bos_token_id(self) -> int:
        """
        Returns the bos_token_id from the model.
        Raises AttributeError if the attribute is missing or inaccessible.
        """

        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        # Explicitly check if the attribute exists
        if not hasattr(self.llm_params, "bos_token_id"):
            raise AttributeError("Missing attribute: self.llm_params.bos_token_id")

        try:
            return self.llm_params.bos_token_id
        except Exception as e:
            # Unexpected error while reading the value
            raise AttributeError(f"Failed to retrieve bos_token_id: {e}")

    def get_num_shards(self) -> int:
        logging.debug("get_num_shards called!")
        if self.__model is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )
            return 1

        if not hasattr(self.__model._model, "is_mcs"):
            logging.debug("model does not have is_mcs param.")
            return 1
        if self.__model._model.is_mcs == 0:
            logging.debug("model is not a MCS (Multi Chip Shard) model.")
            return 1
        if not hasattr(self.__model._model, "mcs_params"):
            logging.debug("model does not have mcs_params param.")
            return 1
        logging.debug(f"num_shards: {self.__model._model.mcs_params.num_shards}")
        return self.__model._model.mcs_params.num_shards

    def is_mcs_model(self) -> bool:
        logging.debug("is_mcs_model called!")
        if self.__model is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )
            return False

        if not hasattr(self.__model._model, "is_mcs"):
            logging.debug("model does not have is_mcs param.")
            return False
        if self.__model._model.is_mcs == 0:
            logging.debug("model is not a MCS (Multi Chip Shard) model.")
            return False
        return True

    def print_llm_params(self) -> None:
        if self.llm_params is None:
            logging.warning(
                "Missing attribute: load_model() or model_get_parameters_from_file() first!"
            )

        print(f"llm_params: {self.llm_params}")

    def get_llm_params(self) -> dict[str, Any]:
        return {
            "vocab_size": self.get_vocab_size(),
            "embedding_size": self.get_embedding_size(),
            "max_num_tokens": self.get_max_num_tokens(),
            "pad_token_id": self.get_pad_token_id(),
            "eos_token_id": self.get_eos_token_id(),
            "bos_token_id": self.get_bos_token_id(),
            "is_speculative": self.is_speculative(),
            "is_host_specd": self.is_host_specd(),
        }
