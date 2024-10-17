# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.redisenterprise import RedisEnterpriseManagementClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestRedisEnterpriseManagementAccessPolicyAssignmentOperations(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(RedisEnterpriseManagementClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_create_update(self, resource_group):
        response = self.client.access_policy_assignment.begin_create_update(
            resource_group_name=resource_group.name,
            cluster_name="str",
            database_name="str",
            access_policy_assignment_name="str",
            parameters={
                "accessPolicyName": "str",
                "id": "str",
                "name": "str",
                "provisioningState": "str",
                "type": "str",
                "user": {"objectId": "str"},
            },
            api_version="2024-09-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get(self, resource_group):
        response = self.client.access_policy_assignment.get(
            resource_group_name=resource_group.name,
            cluster_name="str",
            database_name="str",
            access_policy_assignment_name="str",
            api_version="2024-09-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_delete(self, resource_group):
        response = self.client.access_policy_assignment.begin_delete(
            resource_group_name=resource_group.name,
            cluster_name="str",
            database_name="str",
            access_policy_assignment_name="str",
            api_version="2024-09-01-preview",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list(self, resource_group):
        response = self.client.access_policy_assignment.list(
            resource_group_name=resource_group.name,
            cluster_name="str",
            database_name="str",
            api_version="2024-09-01-preview",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...
