# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from copy import deepcopy
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

from azure.core import PipelineClient
from azure.core.pipeline import policies
from azure.core.rest import HttpRequest, HttpResponse

from ._configuration import AzureAIClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import AgentsOperations, ConnectionsOperations, EvaluationsOperations

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential


class AzureAIClient:
    """AzureAIClient.

    :ivar agents: AgentsOperations operations
    :vartype agents: azure.ai.client.operations.AgentsOperations
    :ivar connections: ConnectionsOperations operations
    :vartype connections: azure.ai.client.operations.ConnectionsOperations
    :ivar evaluations: EvaluationsOperations operations
    :vartype evaluations: azure.ai.client.operations.EvaluationsOperations
    :param endpoint: The Azure AI Studio project endpoint, in the form
     ``https://<azure-region>.api.azureml.ms`` or
     ``https://<private-link-guid>.<azure-region>.api.azureml.ms``\\\\ , where
     :code:`<azure-region>` is the Azure region where the project is deployed (e.g. westus) and
     :code:`<private-link-guid>` is the GUID of the Enterprise private link. Required.
    :type endpoint: str
    :param subscription_id: The Azure subscription ID. Required.
    :type subscription_id: str
    :param resource_group_name: The name of the Azure Resource Group. Required.
    :type resource_group_name: str
    :param project_name: The Azure AI Studio project name. Required.
    :type project_name: str
    :param credential: Credential used to authenticate requests to the service. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword api_version: The API version to use for this operation. Default value is
     "2024-07-01-preview". Note that overriding this default value may result in unsupported
     behavior.
    :paramtype api_version: str
    """

    def __init__(
        self,
        endpoint: str,
        subscription_id: str,
        resource_group_name: str,
        project_name: str,
        credential: "TokenCredential",
        **kwargs: Any
    ) -> None:
        _endpoint = "{endpoint}/agents/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{projectName}"  # pylint: disable=line-too-long
        self._config = AzureAIClientConfiguration(
            endpoint=endpoint,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            project_name=project_name,
            credential=credential,
            **kwargs
        )
        _policies = kwargs.pop("policies", None)
        if _policies is None:
            _policies = [
                policies.RequestIdPolicy(**kwargs),
                self._config.headers_policy,
                self._config.user_agent_policy,
                self._config.proxy_policy,
                policies.ContentDecodePolicy(**kwargs),
                self._config.redirect_policy,
                self._config.retry_policy,
                self._config.authentication_policy,
                self._config.custom_hook_policy,
                self._config.logging_policy,
                policies.DistributedTracingPolicy(**kwargs),
                policies.SensitiveHeaderCleanupPolicy(**kwargs) if self._config.redirect_policy else None,
                self._config.http_logging_policy,
            ]
        self._client: PipelineClient = PipelineClient(base_url=_endpoint, policies=_policies, **kwargs)

        self._serialize = Serializer()
        self._deserialize = Deserializer()
        self._serialize.client_side_validation = False
        self.agents = AgentsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.connections = ConnectionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.evaluations = EvaluationsOperations(self._client, self._config, self._serialize, self._deserialize)

    def send_request(self, request: HttpRequest, *, stream: bool = False, **kwargs: Any) -> HttpResponse:
        """Runs the network request through the client's chained policies.

        >>> from azure.core.rest import HttpRequest
        >>> request = HttpRequest("GET", "https://www.example.org/")
        <HttpRequest [GET], url: 'https://www.example.org/'>
        >>> response = client.send_request(request)
        <HttpResponse: 200 OK>

        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request

        :param request: The network request you want to make. Required.
        :type request: ~azure.core.rest.HttpRequest
        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.
        :return: The response of your network call. Does not do error handling on your response.
        :rtype: ~azure.core.rest.HttpResponse
        """

        request_copy = deepcopy(request)
        path_format_arguments = {
            "endpoint": self._serialize.url("self._config.endpoint", self._config.endpoint, "str"),
            "subscriptionId": self._serialize.url("self._config.subscription_id", self._config.subscription_id, "str"),
            "resourceGroupName": self._serialize.url(
                "self._config.resource_group_name", self._config.resource_group_name, "str"
            ),
            "projectName": self._serialize.url("self._config.project_name", self._config.project_name, "str"),
        }

        request_copy.url = self._client.format_url(request_copy.url, **path_format_arguments)
        return self._client.send_request(request_copy, stream=stream, **kwargs)  # type: ignore

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        self._client.__exit__(*exc_details)
