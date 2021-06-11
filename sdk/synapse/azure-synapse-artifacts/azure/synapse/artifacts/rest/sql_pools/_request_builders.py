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


def build_list_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """List Sql Pools.

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
                        "collation": "str (optional)",
                        "createMode": "str (optional)",
                        "creationDate": "datetime (optional)",
                        "maxSizeBytes": "long (optional)",
                        "provisioningState": "str (optional)",
                        "recoverableDatabaseId": "str (optional)",
                        "restorePointInTime": "str (optional)",
                        "sku": {
                            "capacity": "int (optional)",
                            "name": "str (optional)",
                            "tier": "str (optional)"
                        },
                        "sourceDatabaseId": "str (optional)",
                        "status": "str (optional)"
                    }
                ]
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sqlPools')

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
    sql_pool_name,  # type: str
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Get Sql Pool.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param sql_pool_name: The Sql Pool name.
    :type sql_pool_name: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # response body for status code(s): 200
            response.json() == {
                "collation": "str (optional)",
                "createMode": "str (optional)",
                "creationDate": "datetime (optional)",
                "maxSizeBytes": "long (optional)",
                "provisioningState": "str (optional)",
                "recoverableDatabaseId": "str (optional)",
                "restorePointInTime": "str (optional)",
                "sku": {
                    "capacity": "int (optional)",
                    "name": "str (optional)",
                    "tier": "str (optional)"
                },
                "sourceDatabaseId": "str (optional)",
                "status": "str (optional)"
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sqlPools/{sqlPoolName}')
    path_format_arguments = {
        'sqlPoolName': _SERIALIZER.url("sql_pool_name", sql_pool_name, 'str'),
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

