# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential
from azure.mgmt.reservations import AzureReservationAPI

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-reservations
# USAGE
    python calculate_exchange.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = AzureReservationAPI(
        credential=DefaultAzureCredential(),
    )

    response = client.calculate_exchange.begin_post(
        body={
            "properties": {
                "reservationsToExchange": [
                    {
                        "quantity": 1,
                        "reservationId": "/providers/microsoft.capacity/reservationOrders/1f14354c-dc12-4c8d-8090-6f295a3a34aa/reservations/c8c926bd-fc5d-4e29-9d43-b68340ac23a6",
                    }
                ],
                "reservationsToPurchase": [
                    {
                        "location": "westus",
                        "properties": {
                            "appliedScopeType": "Shared",
                            "appliedScopes": None,
                            "billingPlan": "Upfront",
                            "billingScopeId": "/subscriptions/ed3a1871-612d-abcd-a849-c2542a68be83",
                            "displayName": "testDisplayName",
                            "quantity": 1,
                            "renew": False,
                            "reservedResourceProperties": {"instanceFlexibility": "On"},
                            "reservedResourceType": "VirtualMachines",
                            "term": "P1Y",
                        },
                        "sku": {"name": "Standard_B1ls"},
                    }
                ],
            }
        },
    ).result()
    print(response)


# x-ms-original-file: specification/reservations/resource-manager/Microsoft.Capacity/stable/2022-03-01/examples/CalculateExchange.json
if __name__ == "__main__":
    main()
