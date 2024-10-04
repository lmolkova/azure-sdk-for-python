# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from devtools_testutils.aio import recorded_by_proxy_async
from testpreparer import AzureAIPreparer
from testpreparer_async import AzureAIClientTestBaseAsync


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestAzureAIEvaluationsOperationsAsync(AzureAIClientTestBaseAsync):
    @AzureAIPreparer()
    @recorded_by_proxy_async
    async def test_evaluations_create(self, azureai_endpoint):
        client = self.create_async_client(endpoint=azureai_endpoint)
        response = await client.evaluations.create(
            evaluation={
                "data": "input_data",
                "evaluators": {"str": {"id": "str", "dataMapping": {"str": "str"}, "initParams": {"str": {}}}},
                "description": "str",
                "displayName": "str",
                "id": "str",
                "properties": {"str": "str"},
                "status": "str",
                "systemData": {
                    "createdAt": "2020-02-20 00:00:00",
                    "createdBy": "str",
                    "createdByType": "str",
                    "lastModifiedAt": "2020-02-20 00:00:00",
                },
                "tags": {"str": "str"},
            },
        )

        # please add some check logic here by yourself
        # ...

    @AzureAIPreparer()
    @recorded_by_proxy_async
    async def test_evaluations_list(self, azureai_endpoint):
        client = self.create_async_client(endpoint=azureai_endpoint)
        response = client.evaluations.list()
        result = [r async for r in response]
        # please add some check logic here by yourself
        # ...

    @AzureAIPreparer()
    @recorded_by_proxy_async
    async def test_evaluations_update(self, azureai_endpoint):
        client = self.create_async_client(endpoint=azureai_endpoint)
        response = await client.evaluations.update(
            id="str",
            update_request={"description": "str", "displayName": "str", "tags": {"str": "str"}},
        )

        # please add some check logic here by yourself
        # ...

    @AzureAIPreparer()
    @recorded_by_proxy_async
    async def test_evaluations_get(self, azureai_endpoint):
        client = self.create_async_client(endpoint=azureai_endpoint)
        response = await client.evaluations.get(
            id="str",
        )

        # please add some check logic here by yourself
        # ...
