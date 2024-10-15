# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential

from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-postgresqlflexibleservers
# USAGE
    python migrations_create_other_source_types_validate_migrate.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = PostgreSQLManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
    )

    response = client.migrations.create(
        subscription_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
        resource_group_name="testrg",
        target_db_server_name="testtarget",
        migration_name="testmigration",
        parameters={
            "location": "westus",
            "properties": {
                "dbsToMigrate": ["db1", "db2", "db3", "db4"],
                "migrationMode": "Offline",
                "migrationOption": "ValidateAndMigrate",
                "overwriteDbsInTarget": "True",
                "secretParameters": {
                    "adminCredentials": {"sourceServerPassword": "xxxxxxxx", "targetServerPassword": "xxxxxxxx"}
                },
                "sourceDbServerResourceId": "testsource:5432@pguser",
                "sourceType": "OnPremises",
                "sslMode": "Prefer",
            },
        },
    )
    print(response)


# x-ms-original-file: specification/postgresql/resource-manager/Microsoft.DBforPostgreSQL/stable/2024-08-01/examples/Migrations_Create_Other_SourceTypes_Validate_Migrate.json
if __name__ == "__main__":
    main()
