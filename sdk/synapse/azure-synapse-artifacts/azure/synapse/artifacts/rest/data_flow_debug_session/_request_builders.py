# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING

from azure.core.pipeline.transport._base import _format_url_section
from azure.synapse.artifacts.core.rest import HttpRequest
from msrest import Serializer

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from typing import Any, Dict, IO, Optional

_SERIALIZER = Serializer()


def build_create_data_flow_debug_session_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Creates a data flow debug session.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Data flow debug session definition.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Data flow debug session definition.
    :paramtype content: Any
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "clusterTimeout": "int (optional)",
                "dataBricksLinkedService": {
                    "properties": "properties"
                },
                "dataFlowName": "str (optional)",
                "existingClusterId": "str (optional)",
                "newClusterName": "str (optional)",
                "newClusterNodeType": "str (optional)"
            }

            # response body for status code(s): 200
            response.json() == {
                "sessionId": "str (optional)"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/createDataFlowDebugSession')

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
        **kwargs
    )


def build_query_data_flow_debug_sessions_by_workspace_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Query all active data flow debug sessions.

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
                        "": {
                            "str": "object (optional)"
                        },
                        "computeType": "str (optional)",
                        "coreCount": "int (optional)",
                        "dataFlowName": "str (optional)",
                        "integrationRuntimeName": "str (optional)",
                        "lastActivityTime": "str (optional)",
                        "nodeCount": "int (optional)",
                        "sessionId": "str (optional)",
                        "startTime": "str (optional)",
                        "timeToLiveInMinutes": "int (optional)"
                    }
                ]
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/queryDataFlowDebugSessions')

    # Construct parameters
    query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="POST",
        url=url,
        params=query_parameters,
        headers=header_parameters,
        **kwargs
    )


def build_add_data_flow_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Add a data flow into debug session.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Data flow debug session definition with debug content.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Data flow debug session definition with debug content.
    :paramtype content: Any
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "": {
                    "str": "object (optional)"
                },
                "dataFlow": {
                    "properties": "properties"
                },
                "datasets": [
                    {
                        "properties": "properties"
                    }
                ],
                "debugSettings": {
                    "datasetParameters": "object (optional)",
                    "parameters": {
                        "str": "object (optional)"
                    },
                    "sourceSettings": [
                        {
                            "": {
                                "str": "object (optional)"
                            },
                            "rowLimit": "int (optional)",
                            "sourceName": "str (optional)"
                        }
                    ]
                },
                "linkedServices": [
                    {
                        "properties": "properties"
                    }
                ],
                "sessionId": "str (optional)",
                "staging": {
                    "folderPath": "str (optional)",
                    "linkedService": {
                        "parameters": {
                            "str": "object (optional)"
                        },
                        "referenceName": "str",
                        "type": "str"
                    }
                }
            }

            # response body for status code(s): 200
            response.json() == {
                "jobVersion": "str (optional)"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/addDataFlowToDebugSession')

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
        **kwargs
    )


def build_delete_data_flow_debug_session_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Deletes a data flow debug session.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Data flow debug session definition for deletion.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Data flow debug session definition for deletion.
    :paramtype content: Any
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "dataFlowName": "str (optional)",
                "sessionId": "str (optional)"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/deleteDataFlowDebugSession')

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
        **kwargs
    )


def build_execute_command_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Execute a data flow debug command.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Data flow debug command definition.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Data flow debug command definition.
    :paramtype content: Any
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "commandName": "str (optional)",
                "commandPayload": "object",
                "dataFlowName": "str (optional)",
                "sessionId": "str"
            }

            # response body for status code(s): 200
            response.json() == {
                "data": "str (optional)",
                "status": "str (optional)"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/executeDataFlowDebugCommand')

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
        **kwargs
    )

