# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential

from azure.mgmt.mysqlflexibleservers import MySQLManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-mysqlflexibleservers
# USAGE
    python server_create_with_point_in_time_restore.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = MySQLManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
    )

    response = client.servers.begin_create(
        resource_group_name="TargetResourceGroup",
        server_name="targetserver",
        parameters={
            "location": "SoutheastAsia",
            "properties": {
                "createMode": "PointInTimeRestore",
                "restorePointInTime": "2021-06-24T00:00:37.467Z",
                "sourceServerResourceId": "/subscriptions/ffffffff-ffff-ffff-ffff-ffffffffffff/resourceGroups/SourceResourceGroup/providers/Microsoft.DBforMySQL/flexibleServers/sourceserver",
            },
            "sku": {"name": "Standard_D14_v2", "tier": "GeneralPurpose"},
            "tags": {"num": "1"},
        },
    ).result()
    print(response)


# x-ms-original-file: specification/mysql/resource-manager/Microsoft.DBforMySQL/FlexibleServers/preview/2024-06-01-preview/examples/ServerCreateWithPointInTimeRestore.json
if __name__ == "__main__":
    main()
