# pylint: disable=too-many-lines
# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator (autorest: 3.6.2, generator: @autorest/python@5.12.6)
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from typing import Any, AsyncIterable, Callable, Dict, Optional, TypeVar

from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    map_error,
)
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.mgmt.core.exceptions import ARMErrorFormat

from ... import models as _models
from ..._vendor import _convert_request
from ...operations._index_entities_operations import build_get_entites_cross_region_request

T = TypeVar("T")
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]


class IndexEntitiesOperations:
    """IndexEntitiesOperations async operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~index_service_apis.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """

    models = _models

    def __init__(self, client, config, serializer, deserializer) -> None:
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def get_entites_cross_region(
        self, body: Optional["_models.CrossRegionIndexEntitiesRequest"] = None, **kwargs: Any
    ) -> AsyncIterable["_models.IndexEntitiesResponse"]:
        """get_entites_cross_region.

        :param body:
        :type body: ~index_service_apis.models.CrossRegionIndexEntitiesRequest
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either IndexEntitiesResponse or the result of
         cls(response)
        :rtype:
         ~azure.core.async_paging.AsyncItemPaged[~index_service_apis.models.IndexEntitiesResponse]
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        content_type = kwargs.pop("content_type", "application/json-patch+json")  # type: Optional[str]

        cls = kwargs.pop("cls", None)  # type: ClsType["_models.IndexEntitiesResponse"]
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop("error_map", {}))

        def prepare_request(next_link=None):
            if not next_link:
                if body is not None:
                    _json = self._serialize.body(body, "CrossRegionIndexEntitiesRequest")
                else:
                    _json = None

                request = build_get_entites_cross_region_request(
                    content_type=content_type,
                    json=_json,
                    template_url=self.get_entites_cross_region.metadata["url"],
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)

            else:
                if body is not None:
                    _json = self._serialize.body(body, "CrossRegionIndexEntitiesRequest")
                else:
                    _json = None

                request = build_get_entites_cross_region_request(
                    content_type=content_type,
                    json=_json,
                    template_url=next_link,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = "GET"
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize("IndexEntitiesResponse", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return None, AsyncList(list_of_elem)

        async def get_next(next_link=None):
            request = prepare_request(next_link)

            pipeline_response = await self._client._pipeline.run(  # pylint: disable=protected-access
                request, stream=False, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response

        return AsyncItemPaged(get_next, extract_data)

    get_entites_cross_region.metadata = {"url": "/ux/v1.0/entities/crossRegion"}  # type: ignore