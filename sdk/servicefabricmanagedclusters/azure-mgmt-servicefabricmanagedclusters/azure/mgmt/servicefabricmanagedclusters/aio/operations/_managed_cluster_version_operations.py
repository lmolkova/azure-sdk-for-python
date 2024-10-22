# pylint: disable=too-many-lines,too-many-statements
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import sys
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ResourceNotModifiedError,
    map_error,
)
from azure.core.pipeline import PipelineResponse
from azure.core.rest import AsyncHttpResponse, HttpRequest
from azure.core.tracing.decorator_async import distributed_trace_async
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat

from ... import models as _models
from ...operations._managed_cluster_version_operations import (
    build_get_by_environment_request,
    build_get_request,
    build_list_by_environment_request,
    build_list_request,
)

if sys.version_info >= (3, 9):
    from collections.abc import MutableMapping
else:
    from typing import MutableMapping  # type: ignore  # pylint: disable=ungrouped-imports
T = TypeVar("T")
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]


class ManagedClusterVersionOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.servicefabricmanagedclusters.aio.ServiceFabricManagedClustersManagementClient`'s
        :attr:`managed_cluster_version` attribute.
    """

    models = _models

    def __init__(self, *args, **kwargs) -> None:
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop("client")
        self._config = input_args.pop(0) if input_args else kwargs.pop("config")
        self._serialize = input_args.pop(0) if input_args else kwargs.pop("serializer")
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop("deserializer")

    @distributed_trace_async
    async def get(self, location: str, cluster_version: str, **kwargs: Any) -> _models.ManagedClusterCodeVersionResult:
        """Gets information about a Service Fabric managed cluster code version available in the specified
        location.

        Gets information about an available Service Fabric managed cluster code version.

        :param location: The location for the cluster code versions. This is different from cluster
         location. Required.
        :type location: str
        :param cluster_version: The cluster code version. Required.
        :type cluster_version: str
        :return: ManagedClusterCodeVersionResult or the result of cls(response)
        :rtype: ~azure.mgmt.servicefabricmanagedclusters.models.ManagedClusterCodeVersionResult
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[_models.ManagedClusterCodeVersionResult] = kwargs.pop("cls", None)

        _request = build_get_request(
            location=location,
            cluster_version=cluster_version,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _stream = False
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorModel, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("ManagedClusterCodeVersionResult", pipeline_response.http_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @distributed_trace_async
    async def get_by_environment(
        self,
        location: str,
        environment: Union[str, _models.ManagedClusterVersionEnvironment],
        cluster_version: str,
        **kwargs: Any
    ) -> _models.ManagedClusterCodeVersionResult:
        """Gets information about a Service Fabric cluster code version available for the specified
        environment.

        Gets information about an available Service Fabric cluster code version by environment.

        :param location: The location for the cluster code versions. This is different from cluster
         location. Required.
        :type location: str
        :param environment: The operating system of the cluster. The default means all. "Windows"
         Required.
        :type environment: str or
         ~azure.mgmt.servicefabricmanagedclusters.models.ManagedClusterVersionEnvironment
        :param cluster_version: The cluster code version. Required.
        :type cluster_version: str
        :return: ManagedClusterCodeVersionResult or the result of cls(response)
        :rtype: ~azure.mgmt.servicefabricmanagedclusters.models.ManagedClusterCodeVersionResult
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[_models.ManagedClusterCodeVersionResult] = kwargs.pop("cls", None)

        _request = build_get_by_environment_request(
            location=location,
            environment=environment,
            cluster_version=cluster_version,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _stream = False
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorModel, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("ManagedClusterCodeVersionResult", pipeline_response.http_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @distributed_trace_async
    async def list(self, location: str, **kwargs: Any) -> List[_models.ManagedClusterCodeVersionResult]:
        """Gets the list of Service Fabric cluster code versions available for the specified location.

        Gets all available code versions for Service Fabric cluster resources by location.

        :param location: The location for the cluster code versions. This is different from cluster
         location. Required.
        :type location: str
        :return: list of ManagedClusterCodeVersionResult or the result of cls(response)
        :rtype: list[~azure.mgmt.servicefabricmanagedclusters.models.ManagedClusterCodeVersionResult]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[List[_models.ManagedClusterCodeVersionResult]] = kwargs.pop("cls", None)

        _request = build_list_request(
            location=location,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _stream = False
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorModel, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("[ManagedClusterCodeVersionResult]", pipeline_response.http_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @distributed_trace_async
    async def list_by_environment(
        self, location: str, environment: Union[str, _models.ManagedClusterVersionEnvironment], **kwargs: Any
    ) -> List[_models.ManagedClusterCodeVersionResult]:
        """Gets the list of Service Fabric cluster code versions available for the specified environment.

        Gets all available code versions for Service Fabric cluster resources by environment.

        :param location: The location for the cluster code versions. This is different from cluster
         location. Required.
        :type location: str
        :param environment: The operating system of the cluster. The default means all. "Windows"
         Required.
        :type environment: str or
         ~azure.mgmt.servicefabricmanagedclusters.models.ManagedClusterVersionEnvironment
        :return: list of ManagedClusterCodeVersionResult or the result of cls(response)
        :rtype: list[~azure.mgmt.servicefabricmanagedclusters.models.ManagedClusterCodeVersionResult]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[List[_models.ManagedClusterCodeVersionResult]] = kwargs.pop("cls", None)

        _request = build_list_by_environment_request(
            location=location,
            environment=environment,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _stream = False
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorModel, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("[ManagedClusterCodeVersionResult]", pipeline_response.http_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore
