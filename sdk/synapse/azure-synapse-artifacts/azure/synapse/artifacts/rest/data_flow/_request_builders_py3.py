# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from typing import Any, Dict, IO, Optional

from azure.core.pipeline.transport._base import _format_url_section
from azure.synapse.artifacts.core.rest import HttpRequest
from msrest import Serializer

_SERIALIZER = Serializer()


def build_create_or_update_data_flow_request(
    data_flow_name: str,
    *,
    json: Any = None,
    content: Any = None,
    if_match: Optional[str] = None,
    **kwargs: Any
) -> HttpRequest:
    """Creates or updates a data flow.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param data_flow_name: The data flow name.
    :type data_flow_name: str
    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Data flow resource definition.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Data flow resource definition.
    :paramtype content: Any
    :keyword if_match: ETag of the data flow entity. Should only be specified for update, for which
     it should match existing entity or can be * for unconditional update.
    :paramtype if_match: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python


            # JSON input template you can fill out and use as your `json` input.
            json = {
                "properties": "properties"
            }

            # response body for status code(s): 200
            response.json() == {
                "properties": "properties"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/dataflows/{dataFlowName}')
    path_format_arguments = {
        'dataFlowName': _SERIALIZER.url("data_flow_name", data_flow_name, 'str', max_length=260, min_length=1, pattern=r'^[A-Za-z0-9_][^<>*#.%&:\\+?/]*$'),
    }
    url = _format_url_section(url, **path_format_arguments)

    # Construct parameters
    query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    if if_match is not None:
        header_parameters['If-Match'] = _SERIALIZER.header("if_match", if_match, 'str')
    if content_type is not None:
        header_parameters['Content-Type'] = _SERIALIZER.header("content_type", content_type, 'str')
    header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="PUT",
        url=url,
        params=query_parameters,
        headers=header_parameters,
        json=json,
        content=content,
        **kwargs
    )


def build_get_data_flow_request(
    data_flow_name: str,
    *,
    if_none_match: Optional[str] = None,
    **kwargs: Any
) -> HttpRequest:
    """Gets a data flow.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param data_flow_name: The data flow name.
    :type data_flow_name: str
    :keyword if_none_match: ETag of the data flow entity. Should only be specified for get. If the
     ETag matches the existing entity tag, or if * was provided, then no content will be returned.
    :paramtype if_none_match: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # response body for status code(s): 200
            response.json() == {
                "properties": "properties"
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/dataflows/{dataFlowName}')
    path_format_arguments = {
        'dataFlowName': _SERIALIZER.url("data_flow_name", data_flow_name, 'str', max_length=260, min_length=1, pattern=r'^[A-Za-z0-9_][^<>*#.%&:\\+?/]*$'),
    }
    url = _format_url_section(url, **path_format_arguments)

    # Construct parameters
    query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    if if_none_match is not None:
        header_parameters['If-None-Match'] = _SERIALIZER.header("if_none_match", if_none_match, 'str')
    header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="GET",
        url=url,
        params=query_parameters,
        headers=header_parameters,
        **kwargs
    )


def build_delete_data_flow_request(
    data_flow_name: str,
    **kwargs: Any
) -> HttpRequest:
    """Deletes a data flow.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param data_flow_name: The data flow name.
    :type data_flow_name: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/dataflows/{dataFlowName}')
    path_format_arguments = {
        'dataFlowName': _SERIALIZER.url("data_flow_name", data_flow_name, 'str', max_length=260, min_length=1, pattern=r'^[A-Za-z0-9_][^<>*#.%&:\\+?/]*$'),
    }
    url = _format_url_section(url, **path_format_arguments)

    # Construct parameters
    query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="DELETE",
        url=url,
        params=query_parameters,
        headers=header_parameters,
        **kwargs
    )


def build_rename_data_flow_request(
    data_flow_name: str,
    *,
    json: Any = None,
    content: Any = None,
    **kwargs: Any
) -> HttpRequest:
    """Renames a dataflow.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param data_flow_name: The data flow name.
    :type data_flow_name: str
    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. proposed new name.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). proposed new name.
    :paramtype content: Any
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "newName": "str (optional)"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/dataflows/{dataFlowName}/rename')
    path_format_arguments = {
        'dataFlowName': _SERIALIZER.url("data_flow_name", data_flow_name, 'str', max_length=260, min_length=1, pattern=r'^[A-Za-z0-9_][^<>*#.%&:\\+?/]*$'),
    }
    url = _format_url_section(url, **path_format_arguments)

    # Construct parameters
    query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    if content_type is not None:
        header_parameters['Content-Type'] = _SERIALIZER.header("content_type", content_type, 'str')
    header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="POST",
        url=url,
        params=query_parameters,
        headers=header_parameters,
        json=json,
        content=content,
        **kwargs
    )


def build_get_data_flows_by_workspace_request(
    **kwargs: Any
) -> HttpRequest:
    """Lists data flows.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # response body for status code(s): 200
            response.json() == {
                "nextLink": "str (optional)",
                "value": [
                    {
                        "properties": "properties"
                    }
                ]
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/dataflows')

    # Construct parameters
    query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="GET",
        url=url,
        params=query_parameters,
        headers=header_parameters,
        **kwargs
    )

