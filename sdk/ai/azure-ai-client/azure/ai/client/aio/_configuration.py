# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from typing import Any, TYPE_CHECKING

from azure.core.pipeline import policies

from .._version import VERSION

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from azure.core.credentials_async import AsyncTokenCredential


class AzureAIClientConfiguration:  # pylint: disable=too-many-instance-attributes
    """Configuration for AzureAIClient.

    Note that all parameters used to create this instance are saved as instance
    attributes.

    :param endpoint: The Azure AI Studio project endpoint, in the form
     ``https://<azure-region>.api.azureml.ms`` or
     ``https://<private-link-guid>.<azure-region>.api.azureml.ms``\\ , where :code:`<azure-region>`
     is the Azure region where the project is deployed (e.g. westus) and :code:`<private-link-guid>`
     is the GUID of the Enterprise private link. Required.
    :type endpoint: str
    :param subscription_id: The Azure subscription ID. Required.
    :type subscription_id: str
    :param resource_group_name: The name of the Azure Resource Group. Required.
    :type resource_group_name: str
    :param workspace_name: The name of the Azure AI Studio hub. Required.
    :type workspace_name: str
    :param credential: Credential used to authenticate requests to the service. Required.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
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
        workspace_name: str,
        credential: "AsyncTokenCredential",
        **kwargs: Any
    ) -> None:
        api_version: str = kwargs.pop("api_version", "2024-07-01-preview")

        if endpoint is None:
            raise ValueError("Parameter 'endpoint' must not be None.")
        if subscription_id is None:
            raise ValueError("Parameter 'subscription_id' must not be None.")
        if resource_group_name is None:
            raise ValueError("Parameter 'resource_group_name' must not be None.")
        if workspace_name is None:
            raise ValueError("Parameter 'workspace_name' must not be None.")
        if credential is None:
            raise ValueError("Parameter 'credential' must not be None.")

        self.endpoint = endpoint
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.workspace_name = workspace_name
        self.credential = credential
        self.api_version = api_version
        self.credential_scopes = kwargs.pop("credential_scopes", ["https://management.azure.com/.default"])
        kwargs.setdefault("sdk_moniker", "ai-client/{}".format(VERSION))
        self.polling_interval = kwargs.get("polling_interval", 30)
        self._configure(**kwargs)

    def _configure(self, **kwargs: Any) -> None:
        self.user_agent_policy = kwargs.get("user_agent_policy") or policies.UserAgentPolicy(**kwargs)
        self.headers_policy = kwargs.get("headers_policy") or policies.HeadersPolicy(**kwargs)
        self.proxy_policy = kwargs.get("proxy_policy") or policies.ProxyPolicy(**kwargs)
        self.logging_policy = kwargs.get("logging_policy") or policies.NetworkTraceLoggingPolicy(**kwargs)
        self.http_logging_policy = kwargs.get("http_logging_policy") or policies.HttpLoggingPolicy(**kwargs)
        self.custom_hook_policy = kwargs.get("custom_hook_policy") or policies.CustomHookPolicy(**kwargs)
        self.redirect_policy = kwargs.get("redirect_policy") or policies.AsyncRedirectPolicy(**kwargs)
        self.retry_policy = kwargs.get("retry_policy") or policies.AsyncRetryPolicy(**kwargs)
        self.authentication_policy = kwargs.get("authentication_policy")
        if self.credential and not self.authentication_policy:
            self.authentication_policy = policies.AsyncBearerTokenCredentialPolicy(
                self.credential, *self.credential_scopes, **kwargs
            )
