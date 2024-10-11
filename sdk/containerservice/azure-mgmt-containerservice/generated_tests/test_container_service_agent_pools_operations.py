# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.containerservice import ContainerServiceClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestContainerServiceAgentPoolsOperations(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(ContainerServiceClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_abort_latest_operation(self, resource_group):
        response = self.client.agent_pools.begin_abort_latest_operation(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            api_version="2024-08-01",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list(self, resource_group):
        response = self.client.agent_pools.list(
            resource_group_name=resource_group.name,
            resource_name="str",
            api_version="2024-08-01",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get(self, resource_group):
        response = self.client.agent_pools.get(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            api_version="2024-08-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_create_or_update(self, resource_group):
        response = self.client.agent_pools.begin_create_or_update(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            parameters={
                "availabilityZones": ["str"],
                "capacityReservationGroupID": "str",
                "count": 0,
                "creationData": {"sourceResourceId": "str"},
                "currentOrchestratorVersion": "str",
                "enableAutoScaling": bool,
                "enableEncryptionAtHost": bool,
                "enableFIPS": bool,
                "enableNodePublicIP": bool,
                "enableUltraSSD": bool,
                "gpuInstanceProfile": "str",
                "hostGroupID": "str",
                "id": "str",
                "kubeletConfig": {
                    "allowedUnsafeSysctls": ["str"],
                    "containerLogMaxFiles": 0,
                    "containerLogMaxSizeMB": 0,
                    "cpuCfsQuota": bool,
                    "cpuCfsQuotaPeriod": "str",
                    "cpuManagerPolicy": "str",
                    "failSwapOn": bool,
                    "imageGcHighThreshold": 0,
                    "imageGcLowThreshold": 0,
                    "podMaxPids": 0,
                    "topologyManagerPolicy": "str",
                },
                "kubeletDiskType": "str",
                "linuxOSConfig": {
                    "swapFileSizeMB": 0,
                    "sysctls": {
                        "fsAioMaxNr": 0,
                        "fsFileMax": 0,
                        "fsInotifyMaxUserWatches": 0,
                        "fsNrOpen": 0,
                        "kernelThreadsMax": 0,
                        "netCoreNetdevMaxBacklog": 0,
                        "netCoreOptmemMax": 0,
                        "netCoreRmemDefault": 0,
                        "netCoreRmemMax": 0,
                        "netCoreSomaxconn": 0,
                        "netCoreWmemDefault": 0,
                        "netCoreWmemMax": 0,
                        "netIpv4IpLocalPortRange": "str",
                        "netIpv4NeighDefaultGcThresh1": 0,
                        "netIpv4NeighDefaultGcThresh2": 0,
                        "netIpv4NeighDefaultGcThresh3": 0,
                        "netIpv4TcpFinTimeout": 0,
                        "netIpv4TcpKeepaliveProbes": 0,
                        "netIpv4TcpKeepaliveTime": 0,
                        "netIpv4TcpMaxSynBacklog": 0,
                        "netIpv4TcpMaxTwBuckets": 0,
                        "netIpv4TcpTwReuse": bool,
                        "netIpv4TcpkeepaliveIntvl": 0,
                        "netNetfilterNfConntrackBuckets": 0,
                        "netNetfilterNfConntrackMax": 0,
                        "vmMaxMapCount": 0,
                        "vmSwappiness": 0,
                        "vmVfsCachePressure": 0,
                    },
                    "transparentHugePageDefrag": "str",
                    "transparentHugePageEnabled": "str",
                },
                "maxCount": 0,
                "maxPods": 0,
                "minCount": 0,
                "mode": "str",
                "name": "str",
                "networkProfile": {
                    "allowedHostPorts": [{"portEnd": 0, "portStart": 0, "protocol": "str"}],
                    "applicationSecurityGroups": ["str"],
                    "nodePublicIPTags": [{"ipTagType": "str", "tag": "str"}],
                },
                "nodeImageVersion": "str",
                "nodeLabels": {"str": "str"},
                "nodePublicIPPrefixID": "str",
                "nodeTaints": ["str"],
                "orchestratorVersion": "str",
                "osDiskSizeGB": 0,
                "osDiskType": "str",
                "osSKU": "str",
                "osType": "Linux",
                "podSubnetID": "str",
                "powerState": {"code": "str"},
                "provisioningState": "str",
                "proximityPlacementGroupID": "str",
                "scaleDownMode": "str",
                "scaleSetEvictionPolicy": "Delete",
                "scaleSetPriority": "Regular",
                "securityProfile": {"enableSecureBoot": bool, "enableVTPM": bool},
                "spotMaxPrice": -1,
                "tags": {"str": "str"},
                "type": "str",
                "upgradeSettings": {"drainTimeoutInMinutes": 0, "maxSurge": "str", "nodeSoakDurationInMinutes": 0},
                "vmSize": "str",
                "vnetSubnetID": "str",
                "windowsProfile": {"disableOutboundNat": bool},
                "workloadRuntime": "str",
            },
            api_version="2024-08-01",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_delete(self, resource_group):
        response = self.client.agent_pools.begin_delete(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            api_version="2024-08-01",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get_upgrade_profile(self, resource_group):
        response = self.client.agent_pools.get_upgrade_profile(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            api_version="2024-08-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_delete_machines(self, resource_group):
        response = self.client.agent_pools.begin_delete_machines(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            machines={"machineNames": ["str"]},
            api_version="2024-08-01",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_get_available_agent_pool_versions(self, resource_group):
        response = self.client.agent_pools.get_available_agent_pool_versions(
            resource_group_name=resource_group.name,
            resource_name="str",
            api_version="2024-08-01",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_begin_upgrade_node_image_version(self, resource_group):
        response = self.client.agent_pools.begin_upgrade_node_image_version(
            resource_group_name=resource_group.name,
            resource_name="str",
            agent_pool_name="str",
            api_version="2024-08-01",
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...
