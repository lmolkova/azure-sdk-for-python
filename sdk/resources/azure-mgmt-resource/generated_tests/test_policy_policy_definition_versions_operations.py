# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.resource import PolicyClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestPolicyPolicyDefinitionVersionsOperations(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(PolicyClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_all_builtins(self, resource_group):
        response = self.client.policy_definition_versions.list_all_builtins(
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_all_at_management_group(self, resource_group):
        response = self.client.policy_definition_versions.list_all_at_management_group(
            management_group_name="str",
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_all(self, resource_group):
        response = self.client.policy_definition_versions.list_all(
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_create_or_update(self, resource_group):
        response = self.client.policy_definition_versions.create_or_update(
            policy_definition_name="str",
            policy_definition_version="str",
            parameters={
                "description": "str",
                "displayName": "str",
                "id": "str",
                "metadata": {},
                "mode": "Indexed",
                "name": "str",
                "parameters": {
                    "str": {
                        "allowedValues": [{}],
                        "defaultValue": {},
                        "metadata": {
                            "assignPermissions": bool,
                            "description": "str",
                            "displayName": "str",
                            "strongType": "str",
                        },
                        "schema": {},
                        "type": "str",
                    }
                },
                "policyRule": {},
                "policyType": "str",
                "systemData": {
                    "createdAt": "2020-02-20 00:00:00",
                    "createdBy": "str",
                    "createdByType": "str",
                    "lastModifiedAt": "2020-02-20 00:00:00",
                    "lastModifiedBy": "str",
                    "lastModifiedByType": "str",
                },
                "type": "str",
                "version": "str",
            },
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_delete(self, resource_group):
        response = self.client.policy_definition_versions.delete(
            policy_definition_name="str",
            policy_definition_version="str",
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get(self, resource_group):
        response = self.client.policy_definition_versions.get(
            policy_definition_name="str",
            policy_definition_version="str",
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get_built_in(self, resource_group):
        response = self.client.policy_definition_versions.get_built_in(
            policy_definition_name="str",
            policy_definition_version="str",
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_create_or_update_at_management_group(self, resource_group):
        response = self.client.policy_definition_versions.create_or_update_at_management_group(
            management_group_name="str",
            policy_definition_name="str",
            policy_definition_version="str",
            parameters={
                "description": "str",
                "displayName": "str",
                "id": "str",
                "metadata": {},
                "mode": "Indexed",
                "name": "str",
                "parameters": {
                    "str": {
                        "allowedValues": [{}],
                        "defaultValue": {},
                        "metadata": {
                            "assignPermissions": bool,
                            "description": "str",
                            "displayName": "str",
                            "strongType": "str",
                        },
                        "schema": {},
                        "type": "str",
                    }
                },
                "policyRule": {},
                "policyType": "str",
                "systemData": {
                    "createdAt": "2020-02-20 00:00:00",
                    "createdBy": "str",
                    "createdByType": "str",
                    "lastModifiedAt": "2020-02-20 00:00:00",
                    "lastModifiedBy": "str",
                    "lastModifiedByType": "str",
                },
                "type": "str",
                "version": "str",
            },
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_delete_at_management_group(self, resource_group):
        response = self.client.policy_definition_versions.delete_at_management_group(
            management_group_name="str",
            policy_definition_name="str",
            policy_definition_version="str",
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get_at_management_group(self, resource_group):
        response = self.client.policy_definition_versions.get_at_management_group(
            management_group_name="str",
            policy_definition_name="str",
            policy_definition_version="str",
            api_version="2023-04-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list(self, resource_group):
        response = self.client.policy_definition_versions.list(
            policy_definition_name="str",
            api_version="2023-04-01",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_built_in(self, resource_group):
        response = self.client.policy_definition_versions.list_built_in(
            policy_definition_name="str",
            api_version="2023-04-01",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_by_management_group(self, resource_group):
        response = self.client.policy_definition_versions.list_by_management_group(
            management_group_name="str",
            policy_definition_name="str",
            api_version="2023-04-01",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...
