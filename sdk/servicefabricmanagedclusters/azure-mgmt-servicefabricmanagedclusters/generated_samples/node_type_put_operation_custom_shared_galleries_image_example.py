# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential

from azure.mgmt.servicefabricmanagedclusters import ServiceFabricManagedClustersManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-servicefabricmanagedclusters
# USAGE
    python node_type_put_operation_custom_shared_galleries_image_example.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = ServiceFabricManagedClustersManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="00000000-0000-0000-0000-000000000000",
    )

    response = client.node_types.begin_create_or_update(
        resource_group_name="resRg",
        cluster_name="myCluster",
        node_type_name="BE",
        parameters={
            "properties": {
                "dataDiskSizeGB": 200,
                "isPrimary": False,
                "vmInstanceCount": 10,
                "vmSharedGalleryImageId": "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-custom-image/providers/Microsoft.Compute/sharedGalleries/35349201-a0b3-405e-8a23-9f1450984307-SFSHAREDGALLERY/images/TestNoProdContainerDImage/versions/latest",
                "vmSize": "Standard_D3",
            }
        },
    ).result()
    print(response)


# x-ms-original-file: specification/servicefabricmanagedclusters/resource-manager/Microsoft.ServiceFabric/preview/2024-06-01-preview/examples/NodeTypePutOperationCustomSharedGalleriesImage_example.json
if __name__ == "__main__":
    main()
