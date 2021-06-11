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


def build_get_spark_job_definitions_by_workspace_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Lists spark job definitions.

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
                        "properties": {
                            "": {
                                "str": "object (optional)"
                            },
                            "description": "str (optional)",
                            "jobProperties": {
                                "": {
                                    "str": "object (optional)"
                                },
                                "archives": [
                                    "str (optional)"
                                ],
                                "args": [
                                    "str (optional)"
                                ],
                                "className": "str (optional)",
                                "conf": "object (optional)",
                                "driverCores": "int",
                                "driverMemory": "str",
                                "executorCores": "int",
                                "executorMemory": "str",
                                "file": "str",
                                "files": [
                                    "str (optional)"
                                ],
                                "jars": [
                                    "str (optional)"
                                ],
                                "name": "str (optional)",
                                "numExecutors": "int"
                            },
                            "language": "str (optional)",
                            "requiredSparkVersion": "str (optional)",
                            "targetBigDataPool": {
                                "referenceName": "str",
                                "type": "str"
                            }
                        }
                    }
                ]
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sparkJobDefinitions')

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


def build_create_or_update_spark_job_definition_request(
    spark_job_definition_name,  # type: str
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Creates or updates a Spark Job Definition.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param spark_job_definition_name: The spark job definition name.
    :type spark_job_definition_name: str
    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Spark Job Definition resource definition.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Spark Job Definition resource definition.
    :paramtype content: Any
    :keyword if_match: ETag of the Spark Job Definition entity.  Should only be specified for
     update, for which it should match existing entity or can be * for unconditional update.
    :paramtype if_match: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "properties": {
                    "": {
                        "str": "object (optional)"
                    },
                    "description": "str (optional)",
                    "jobProperties": {
                        "": {
                            "str": "object (optional)"
                        },
                        "archives": [
                            "str (optional)"
                        ],
                        "args": [
                            "str (optional)"
                        ],
                        "className": "str (optional)",
                        "conf": "object (optional)",
                        "driverCores": "int",
                        "driverMemory": "str",
                        "executorCores": "int",
                        "executorMemory": "str",
                        "file": "str",
                        "files": [
                            "str (optional)"
                        ],
                        "jars": [
                            "str (optional)"
                        ],
                        "name": "str (optional)",
                        "numExecutors": "int"
                    },
                    "language": "str (optional)",
                    "requiredSparkVersion": "str (optional)",
                    "targetBigDataPool": {
                        "referenceName": "str",
                        "type": "str"
                    }
                }
            }

            # response body for status code(s): 200
            response.json() == {
                "properties": {
                    "": {
                        "str": "object (optional)"
                    },
                    "description": "str (optional)",
                    "jobProperties": {
                        "": {
                            "str": "object (optional)"
                        },
                        "archives": [
                            "str (optional)"
                        ],
                        "args": [
                            "str (optional)"
                        ],
                        "className": "str (optional)",
                        "conf": "object (optional)",
                        "driverCores": "int",
                        "driverMemory": "str",
                        "executorCores": "int",
                        "executorMemory": "str",
                        "file": "str",
                        "files": [
                            "str (optional)"
                        ],
                        "jars": [
                            "str (optional)"
                        ],
                        "name": "str (optional)",
                        "numExecutors": "int"
                    },
                    "language": "str (optional)",
                    "requiredSparkVersion": "str (optional)",
                    "targetBigDataPool": {
                        "referenceName": "str",
                        "type": "str"
                    }
                }
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]
    if_match = kwargs.pop('if_match', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sparkJobDefinitions/{sparkJobDefinitionName}')
    path_format_arguments = {
        'sparkJobDefinitionName': _SERIALIZER.url("spark_job_definition_name", spark_job_definition_name, 'str'),
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
        **kwargs
    )


def build_get_spark_job_definition_request(
    spark_job_definition_name,  # type: str
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Gets a Spark Job Definition.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param spark_job_definition_name: The spark job definition name.
    :type spark_job_definition_name: str
    :keyword if_none_match: ETag of the Spark Job Definition entity. Should only be specified for
     get. If the ETag matches the existing entity tag, or if * was provided, then no content will be
     returned.
    :paramtype if_none_match: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # response body for status code(s): 200
            response.json() == {
                "properties": {
                    "": {
                        "str": "object (optional)"
                    },
                    "description": "str (optional)",
                    "jobProperties": {
                        "": {
                            "str": "object (optional)"
                        },
                        "archives": [
                            "str (optional)"
                        ],
                        "args": [
                            "str (optional)"
                        ],
                        "className": "str (optional)",
                        "conf": "object (optional)",
                        "driverCores": "int",
                        "driverMemory": "str",
                        "executorCores": "int",
                        "executorMemory": "str",
                        "file": "str",
                        "files": [
                            "str (optional)"
                        ],
                        "jars": [
                            "str (optional)"
                        ],
                        "name": "str (optional)",
                        "numExecutors": "int"
                    },
                    "language": "str (optional)",
                    "requiredSparkVersion": "str (optional)",
                    "targetBigDataPool": {
                        "referenceName": "str",
                        "type": "str"
                    }
                }
            }
    """

    if_none_match = kwargs.pop('if_none_match', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sparkJobDefinitions/{sparkJobDefinitionName}')
    path_format_arguments = {
        'sparkJobDefinitionName': _SERIALIZER.url("spark_job_definition_name", spark_job_definition_name, 'str'),
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


def build_delete_spark_job_definition_request(
    spark_job_definition_name,  # type: str
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Deletes a Spark Job Definition.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param spark_job_definition_name: The spark job definition name.
    :type spark_job_definition_name: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sparkJobDefinitions/{sparkJobDefinitionName}')
    path_format_arguments = {
        'sparkJobDefinitionName': _SERIALIZER.url("spark_job_definition_name", spark_job_definition_name, 'str'),
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


def build_execute_spark_job_definition_request(
    spark_job_definition_name,  # type: str
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Executes the spark job definition.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param spark_job_definition_name: The spark job definition name.
    :type spark_job_definition_name: str
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # response body for status code(s): 200, 202
            response.json() == {
                "appId": "str (optional)",
                "appInfo": {
                    "str": "str (optional)"
                },
                "artifactId": "str (optional)",
                "errorInfo": [
                    {
                        "errorCode": "str (optional)",
                        "message": "str (optional)",
                        "source": "str (optional)"
                    }
                ],
                "id": "int",
                "jobType": "str (optional)",
                "livyInfo": {
                    "currentState": "str (optional)",
                    "deadAt": "datetime (optional)",
                    "jobCreationRequest": {
                        "archives": [
                            "str (optional)"
                        ],
                        "args": [
                            "str (optional)"
                        ],
                        "className": "str (optional)",
                        "conf": {
                            "str": "str (optional)"
                        },
                        "driverCores": "int (optional)",
                        "driverMemory": "str (optional)",
                        "executorCores": "int (optional)",
                        "executorMemory": "str (optional)",
                        "file": "str (optional)",
                        "files": [
                            "str (optional)"
                        ],
                        "jars": [
                            "str (optional)"
                        ],
                        "name": "str (optional)",
                        "numExecutors": "int (optional)",
                        "pyFiles": [
                            "str (optional)"
                        ]
                    },
                    "killedAt": "datetime (optional)",
                    "notStartedAt": "datetime (optional)",
                    "recoveringAt": "datetime (optional)",
                    "runningAt": "datetime (optional)",
                    "startingAt": "datetime (optional)",
                    "successAt": "datetime (optional)"
                },
                "log": [
                    "str (optional)"
                ],
                "name": "str (optional)",
                "pluginInfo": {
                    "cleanupStartedAt": "datetime (optional)",
                    "currentState": "str (optional)",
                    "monitoringStartedAt": "datetime (optional)",
                    "preparationStartedAt": "datetime (optional)",
                    "resourceAcquisitionStartedAt": "datetime (optional)",
                    "submissionStartedAt": "datetime (optional)"
                },
                "result": "str (optional)",
                "schedulerInfo": {
                    "cancellationRequestedAt": "datetime (optional)",
                    "currentState": "str (optional)",
                    "endedAt": "datetime (optional)",
                    "scheduledAt": "datetime (optional)",
                    "submittedAt": "datetime (optional)"
                },
                "sparkPoolName": "str (optional)",
                "state": "str (optional)",
                "submitterId": "str (optional)",
                "submitterName": "str (optional)",
                "tags": {
                    "str": "str (optional)"
                },
                "workspaceName": "str (optional)"
            }
    """


    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/sparkJobDefinitions/{sparkJobDefinitionName}/execute')
    path_format_arguments = {
        'sparkJobDefinitionName': _SERIALIZER.url("spark_job_definition_name", spark_job_definition_name, 'str'),
    }
    url = _format_url_section(url, **path_format_arguments)

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


def build_rename_spark_job_definition_request(
    spark_job_definition_name,  # type: str
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Renames a sparkJobDefinition.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :param spark_job_definition_name: The spark job definition name.
    :type spark_job_definition_name: str
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
    url = kwargs.pop("template_url", '/sparkJobDefinitions/{sparkJobDefinitionName}/rename')
    path_format_arguments = {
        'sparkJobDefinitionName': _SERIALIZER.url("spark_job_definition_name", spark_job_definition_name, 'str'),
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
        **kwargs
    )


def build_debug_spark_job_definition_request(
    **kwargs  # type: Any
):
    # type: (...) -> HttpRequest
    """Debug the spark job definition.

    See https://aka.ms/azsdk/python/llcwiki for how to incorporate this request builder into your
    code flow.

    :keyword json: Pass in a JSON-serializable object (usually a dictionary). See the template in
     our example to find the input shape. Spark Job Definition resource definition.
    :paramtype json: Any
    :keyword content: Pass in binary content you want in the body of the request (typically bytes,
     a byte iterator, or stream input). Spark Job Definition resource definition.
    :paramtype content: Any
    :return: Returns an :class:`~azure.synapse.artifacts.core.rest.HttpRequest` that you will pass
     to the client's `send_request` method. See https://aka.ms/azsdk/python/protocol/quickstart for
     how to incorporate this response into your code flow.
    :rtype: ~azure.synapse.artifacts.core.rest.HttpRequest

    Example:
        .. code-block:: python

            # JSON input template you can fill out and use as your `json` input.
            json = {
                "properties": {
                    "": {
                        "str": "object (optional)"
                    },
                    "description": "str (optional)",
                    "jobProperties": {
                        "": {
                            "str": "object (optional)"
                        },
                        "archives": [
                            "str (optional)"
                        ],
                        "args": [
                            "str (optional)"
                        ],
                        "className": "str (optional)",
                        "conf": "object (optional)",
                        "driverCores": "int",
                        "driverMemory": "str",
                        "executorCores": "int",
                        "executorMemory": "str",
                        "file": "str",
                        "files": [
                            "str (optional)"
                        ],
                        "jars": [
                            "str (optional)"
                        ],
                        "name": "str (optional)",
                        "numExecutors": "int"
                    },
                    "language": "str (optional)",
                    "requiredSparkVersion": "str (optional)",
                    "targetBigDataPool": {
                        "referenceName": "str",
                        "type": "str"
                    }
                }
            }

            # response body for status code(s): 200, 202
            response.json() == {
                "appId": "str (optional)",
                "appInfo": {
                    "str": "str (optional)"
                },
                "artifactId": "str (optional)",
                "errorInfo": [
                    {
                        "errorCode": "str (optional)",
                        "message": "str (optional)",
                        "source": "str (optional)"
                    }
                ],
                "id": "int",
                "jobType": "str (optional)",
                "livyInfo": {
                    "currentState": "str (optional)",
                    "deadAt": "datetime (optional)",
                    "jobCreationRequest": {
                        "archives": [
                            "str (optional)"
                        ],
                        "args": [
                            "str (optional)"
                        ],
                        "className": "str (optional)",
                        "conf": {
                            "str": "str (optional)"
                        },
                        "driverCores": "int (optional)",
                        "driverMemory": "str (optional)",
                        "executorCores": "int (optional)",
                        "executorMemory": "str (optional)",
                        "file": "str (optional)",
                        "files": [
                            "str (optional)"
                        ],
                        "jars": [
                            "str (optional)"
                        ],
                        "name": "str (optional)",
                        "numExecutors": "int (optional)",
                        "pyFiles": [
                            "str (optional)"
                        ]
                    },
                    "killedAt": "datetime (optional)",
                    "notStartedAt": "datetime (optional)",
                    "recoveringAt": "datetime (optional)",
                    "runningAt": "datetime (optional)",
                    "startingAt": "datetime (optional)",
                    "successAt": "datetime (optional)"
                },
                "log": [
                    "str (optional)"
                ],
                "name": "str (optional)",
                "pluginInfo": {
                    "cleanupStartedAt": "datetime (optional)",
                    "currentState": "str (optional)",
                    "monitoringStartedAt": "datetime (optional)",
                    "preparationStartedAt": "datetime (optional)",
                    "resourceAcquisitionStartedAt": "datetime (optional)",
                    "submissionStartedAt": "datetime (optional)"
                },
                "result": "str (optional)",
                "schedulerInfo": {
                    "cancellationRequestedAt": "datetime (optional)",
                    "currentState": "str (optional)",
                    "endedAt": "datetime (optional)",
                    "scheduledAt": "datetime (optional)",
                    "submittedAt": "datetime (optional)"
                },
                "sparkPoolName": "str (optional)",
                "state": "str (optional)",
                "submitterId": "str (optional)",
                "submitterName": "str (optional)",
                "tags": {
                    "str": "str (optional)"
                },
                "workspaceName": "str (optional)"
            }
    """

    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    api_version = "2019-06-01-preview"
    accept = "application/json"

    # Construct URL
    url = kwargs.pop("template_url", '/debugSparkJobDefinition')

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

