# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.servicelinker import ServiceLinkerManagementClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestServiceLinkerManagementConnectorOperations(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(ServiceLinkerManagementClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_dryrun(self, resource_group):
        response = self.client.connector.list_dryrun(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            api_version="2024-07-01-preview",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get_dryrun(self, resource_group):
        response = self.client.connector.get_dryrun(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            dryrun_name="str",
            api_version="2024-07-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_create_dryrun(self, resource_group):
        response = self.client.connector.begin_create_dryrun(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            dryrun_name="str",
            parameters={
                "id": "str",
                "name": "str",
                "operationPreviews": [
                    {"action": "str", "description": "str", "name": "str", "operationType": "str", "scope": "str"}
                ],
                "parameters": "dryrun_parameters",
                "prerequisiteResults": ["dryrun_prerequisite_result"],
                "provisioningState": "str",
                "systemData": {
                    "createdAt": "2020-02-20 00:00:00",
                    "createdBy": "str",
                    "createdByType": "str",
                    "lastModifiedAt": "2020-02-20 00:00:00",
                    "lastModifiedBy": "str",
                    "lastModifiedByType": "str",
                },
                "type": "str",
            },
            api_version="2024-07-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_update_dryrun(self, resource_group):
        response = self.client.connector.begin_update_dryrun(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            dryrun_name="str",
            parameters={
                "operationPreviews": [
                    {"action": "str", "description": "str", "name": "str", "operationType": "str", "scope": "str"}
                ],
                "parameters": "dryrun_parameters",
                "prerequisiteResults": ["dryrun_prerequisite_result"],
                "provisioningState": "str",
            },
            api_version="2024-07-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_delete_dryrun(self, resource_group):
        response = self.client.connector.delete_dryrun(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            dryrun_name="str",
            api_version="2024-07-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list(self, resource_group):
        response = self.client.connector.list(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            api_version="2024-07-01-preview",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get(self, resource_group):
        response = self.client.connector.get(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            connector_name="str",
            api_version="2024-07-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_create_or_update(self, resource_group):
        response = self.client.connector.begin_create_or_update(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            connector_name="str",
            parameters={
                "authInfo": "auth_info_base",
                "clientType": "str",
                "configurationInfo": {
                    "action": "str",
                    "additionalConfigurations": {"str": "str"},
                    "additionalConnectionStringProperties": {"str": "str"},
                    "configurationStore": {"appConfigurationId": "str"},
                    "customizedKeys": {"str": "str"},
                    "daprProperties": {
                        "bindingComponentDirection": "str",
                        "componentType": "str",
                        "metadata": [
                            {"description": "str", "name": "str", "required": "str", "secretRef": "str", "value": "str"}
                        ],
                        "runtimeVersion": "str",
                        "scopes": ["str"],
                        "secretStoreComponent": "str",
                        "version": "str",
                    },
                    "deleteOrUpdateBehavior": "str",
                },
                "id": "str",
                "name": "str",
                "provisioningState": "str",
                "publicNetworkSolution": {
                    "action": "str",
                    "deleteOrUpdateBehavior": "str",
                    "firewallRules": {"azureServices": "str", "callerClientIP": "str", "ipRanges": ["str"]},
                },
                "scope": "str",
                "secretStore": {"keyVaultId": "str", "keyVaultSecretName": "str"},
                "systemData": {
                    "createdAt": "2020-02-20 00:00:00",
                    "createdBy": "str",
                    "createdByType": "str",
                    "lastModifiedAt": "2020-02-20 00:00:00",
                    "lastModifiedBy": "str",
                    "lastModifiedByType": "str",
                },
                "targetService": "target_service_base",
                "type": "str",
                "vNetSolution": {"deleteOrUpdateBehavior": "str", "type": "str"},
            },
            api_version="2024-07-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_delete(self, resource_group):
        response = self.client.connector.begin_delete(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            connector_name="str",
            api_version="2024-07-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_update(self, resource_group):
        response = self.client.connector.begin_update(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            connector_name="str",
            parameters={
                "authInfo": "auth_info_base",
                "clientType": "str",
                "configurationInfo": {
                    "action": "str",
                    "additionalConfigurations": {"str": "str"},
                    "additionalConnectionStringProperties": {"str": "str"},
                    "configurationStore": {"appConfigurationId": "str"},
                    "customizedKeys": {"str": "str"},
                    "daprProperties": {
                        "bindingComponentDirection": "str",
                        "componentType": "str",
                        "metadata": [
                            {"description": "str", "name": "str", "required": "str", "secretRef": "str", "value": "str"}
                        ],
                        "runtimeVersion": "str",
                        "scopes": ["str"],
                        "secretStoreComponent": "str",
                        "version": "str",
                    },
                    "deleteOrUpdateBehavior": "str",
                },
                "provisioningState": "str",
                "publicNetworkSolution": {
                    "action": "str",
                    "deleteOrUpdateBehavior": "str",
                    "firewallRules": {"azureServices": "str", "callerClientIP": "str", "ipRanges": ["str"]},
                },
                "scope": "str",
                "secretStore": {"keyVaultId": "str", "keyVaultSecretName": "str"},
                "targetService": "target_service_base",
                "vNetSolution": {"deleteOrUpdateBehavior": "str", "type": "str"},
            },
            api_version="2024-07-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_validate(self, resource_group):
        response = self.client.connector.begin_validate(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            connector_name="str",
            api_version="2024-07-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_generate_configurations(self, resource_group):
        response = self.client.connector.generate_configurations(
            subscription_id="str",
            resource_group_name=resource_group.name,
            location="str",
            connector_name="str",
            api_version="2024-07-01-preview",
        )

        # please add some check logic here by yourself
        # ...
