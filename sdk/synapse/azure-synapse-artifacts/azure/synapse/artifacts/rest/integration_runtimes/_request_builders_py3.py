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


def build_list_request(
    **kwargs: Any
) -> HttpRequest:
    """List Integration Runtimes.

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
    url = kwargs.pop("template_url", '/integrationRuntimes')

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


def build_get_request(
    integration_runtime_name: str,
    **kwargs: Any
) -> HttpRequest:
    """Get Integration Runtime.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param integration_runtime_name: The Integration Runtime name.
    :type integration_runtime_name: str
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
    url = kwargs.pop("template_url", '/integrationRuntimes/{integrationRuntimeName}')
    path_format_arguments = {
        'integrationRuntimeName': _SERIALIZER.url("integration_runtime_name", integration_runtime_name, 'str'),
    }
    url = _format_url_section(url, **path_format_arguments)

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

