# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.computeschedule.aio import ComputeScheduleMgmtClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer
from devtools_testutils.aio import recorded_by_proxy_async

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestComputeScheduleMgmtScheduledActionsOperationsAsync(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(ComputeScheduleMgmtClient, is_async=True)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_submit_deallocate(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_submit_deallocate(
            locationparameter="str",
            request_body={
                "correlationid": "str",
                "executionParameters": {
                    "optimizationPreference": "str",
                    "retryPolicy": {"retryCount": 0, "retryWindowInMinutes": 0},
                },
                "resources": {"ids": ["str"]},
                "schedule": {"deadLine": "2020-02-20 00:00:00", "deadlineType": "str", "timeZone": "str"},
            },
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_submit_hibernate(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_submit_hibernate(
            locationparameter="str",
            request_body={
                "correlationid": "str",
                "executionParameters": {
                    "optimizationPreference": "str",
                    "retryPolicy": {"retryCount": 0, "retryWindowInMinutes": 0},
                },
                "resources": {"ids": ["str"]},
                "schedule": {"deadLine": "2020-02-20 00:00:00", "deadlineType": "str", "timeZone": "str"},
            },
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_submit_start(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_submit_start(
            locationparameter="str",
            request_body={
                "correlationid": "str",
                "executionParameters": {
                    "optimizationPreference": "str",
                    "retryPolicy": {"retryCount": 0, "retryWindowInMinutes": 0},
                },
                "resources": {"ids": ["str"]},
                "schedule": {"deadLine": "2020-02-20 00:00:00", "deadlineType": "str", "timeZone": "str"},
            },
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_execute_deallocate(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_execute_deallocate(
            locationparameter="str",
            request_body={
                "correlationid": "str",
                "executionParameters": {
                    "optimizationPreference": "str",
                    "retryPolicy": {"retryCount": 0, "retryWindowInMinutes": 0},
                },
                "resources": {"ids": ["str"]},
            },
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_execute_hibernate(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_execute_hibernate(
            locationparameter="str",
            request_body={
                "correlationid": "str",
                "executionParameters": {
                    "optimizationPreference": "str",
                    "retryPolicy": {"retryCount": 0, "retryWindowInMinutes": 0},
                },
                "resources": {"ids": ["str"]},
            },
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_execute_start(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_execute_start(
            locationparameter="str",
            request_body={
                "correlationid": "str",
                "executionParameters": {
                    "optimizationPreference": "str",
                    "retryPolicy": {"retryCount": 0, "retryWindowInMinutes": 0},
                },
                "resources": {"ids": ["str"]},
            },
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_get_operation_status(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_get_operation_status(
            locationparameter="str",
            request_body={"correlationid": "str", "operationIds": ["str"]},
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_cancel_operations(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_cancel_operations(
            locationparameter="str",
            request_body={"correlationid": "str", "operationIds": ["str"]},
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_scheduled_actions_virtual_machines_get_operation_errors(self, resource_group):
        response = await self.client.scheduled_actions.virtual_machines_get_operation_errors(
            locationparameter="str",
            request_body={"operationIds": ["str"]},
        )

        # please add some check logic here by yourself
        # ...
