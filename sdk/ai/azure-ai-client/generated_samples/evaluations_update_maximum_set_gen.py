# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential

from azure.ai.client import AzureAIClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-ai-client
# USAGE
    python evaluations_update_maximum_set_gen.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = AzureAIClient(
        endpoint="ENDPOINT",
        subscription_id="SUBSCRIPTION_ID",
        resource_group_name="RESOURCE_GROUP_NAME",
        workspace_name="WORKSPACE_NAME",
        credential=DefaultAzureCredential(),
    )

    response = client.evaluations.update(
        id="8y",
        update_request={
            "description": "vl",
            "displayName": "zkystmqhvncvxnxrhahhulbui",
            "tags": {"key6951": "mirtkcesgent"},
        },
    )
    print(response)


# x-ms-original-file: 2024-07-01-preview/Evaluations_Update_MaximumSet_Gen.json
if __name__ == "__main__":
    main()
