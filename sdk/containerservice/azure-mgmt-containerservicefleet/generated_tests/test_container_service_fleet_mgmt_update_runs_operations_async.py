# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.containerservicefleet.aio import ContainerServiceFleetMgmtClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer
from devtools_testutils.aio import recorded_by_proxy_async

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestContainerServiceFleetMgmtUpdateRunsOperationsAsync(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(ContainerServiceFleetMgmtClient, is_async=True)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_list_by_fleet(self, resource_group):
        response = self.client.update_runs.list_by_fleet(
            resource_group_name=resource_group.name,
            fleet_name="str",
            api_version="2024-05-02-preview",
        )
        result = [r async for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_get(self, resource_group):
        response = await self.client.update_runs.get(
            resource_group_name=resource_group.name,
            fleet_name="str",
            update_run_name="str",
            api_version="2024-05-02-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_begin_create_or_update(self, resource_group):
        response = await (
            await self.client.update_runs.begin_create_or_update(
                resource_group_name=resource_group.name,
                fleet_name="str",
                update_run_name="str",
                resource={
                    "eTag": "str",
                    "id": "str",
                    "managedClusterUpdate": {
                        "upgrade": {"type": "str", "kubernetesVersion": "str"},
                        "nodeImageSelection": {"type": "str", "customNodeImageVersions": [{"version": "str"}]},
                    },
                    "name": "str",
                    "provisioningState": "str",
                    "status": {
                        "nodeImageSelection": {"selectedNodeImageVersions": [{"version": "str"}]},
                        "stages": [
                            {
                                "afterStageWaitStatus": {
                                    "status": {
                                        "completedTime": "2020-02-20 00:00:00",
                                        "error": {
                                            "additionalInfo": [{"info": {}, "type": "str"}],
                                            "code": "str",
                                            "details": [...],
                                            "message": "str",
                                            "target": "str",
                                        },
                                        "startTime": "2020-02-20 00:00:00",
                                        "state": "str",
                                    },
                                    "waitDurationInSeconds": 0,
                                },
                                "groups": [
                                    {
                                        "members": [
                                            {
                                                "clusterResourceId": "str",
                                                "message": "str",
                                                "name": "str",
                                                "operationId": "str",
                                                "status": {
                                                    "completedTime": "2020-02-20 00:00:00",
                                                    "error": {
                                                        "additionalInfo": [{"info": {}, "type": "str"}],
                                                        "code": "str",
                                                        "details": [...],
                                                        "message": "str",
                                                        "target": "str",
                                                    },
                                                    "startTime": "2020-02-20 00:00:00",
                                                    "state": "str",
                                                },
                                            }
                                        ],
                                        "name": "str",
                                        "status": {
                                            "completedTime": "2020-02-20 00:00:00",
                                            "error": {
                                                "additionalInfo": [{"info": {}, "type": "str"}],
                                                "code": "str",
                                                "details": [...],
                                                "message": "str",
                                                "target": "str",
                                            },
                                            "startTime": "2020-02-20 00:00:00",
                                            "state": "str",
                                        },
                                    }
                                ],
                                "name": "str",
                                "status": {
                                    "completedTime": "2020-02-20 00:00:00",
                                    "error": {
                                        "additionalInfo": [{"info": {}, "type": "str"}],
                                        "code": "str",
                                        "details": [...],
                                        "message": "str",
                                        "target": "str",
                                    },
                                    "startTime": "2020-02-20 00:00:00",
                                    "state": "str",
                                },
                            }
                        ],
                        "status": {
                            "completedTime": "2020-02-20 00:00:00",
                            "error": {
                                "additionalInfo": [{"info": {}, "type": "str"}],
                                "code": "str",
                                "details": [...],
                                "message": "str",
                                "target": "str",
                            },
                            "startTime": "2020-02-20 00:00:00",
                            "state": "str",
                        },
                    },
                    "strategy": {
                        "stages": [{"name": "str", "afterStageWaitInSeconds": 0, "groups": [{"name": "str"}]}]
                    },
                    "systemData": {
                        "createdAt": "2020-02-20 00:00:00",
                        "createdBy": "str",
                        "createdByType": "str",
                        "lastModifiedAt": "2020-02-20 00:00:00",
                        "lastModifiedBy": "str",
                        "lastModifiedByType": "str",
                    },
                    "type": "str",
                    "updateStrategyId": "str",
                },
                api_version="2024-05-02-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_begin_delete(self, resource_group):
        response = await (
            await self.client.update_runs.begin_delete(
                resource_group_name=resource_group.name,
                fleet_name="str",
                update_run_name="str",
                api_version="2024-05-02-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_begin_skip(self, resource_group):
        response = await (
            await self.client.update_runs.begin_skip(
                resource_group_name=resource_group.name,
                fleet_name="str",
                update_run_name="str",
                body={"targets": [{"name": "str", "type": "str"}]},
                api_version="2024-05-02-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_begin_start(self, resource_group):
        response = await (
            await self.client.update_runs.begin_start(
                resource_group_name=resource_group.name,
                fleet_name="str",
                update_run_name="str",
                api_version="2024-05-02-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_begin_stop(self, resource_group):
        response = await (
            await self.client.update_runs.begin_stop(
                resource_group_name=resource_group.name,
                fleet_name="str",
                update_run_name="str",
                api_version="2024-05-02-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...
