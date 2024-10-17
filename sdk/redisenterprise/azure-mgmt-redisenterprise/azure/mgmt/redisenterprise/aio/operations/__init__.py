# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._operations import Operations
from ._operations_status_operations import OperationsStatusOperations
from ._redis_enterprise_operations import RedisEnterpriseOperations
from ._databases_operations import DatabasesOperations
from ._access_policy_assignment_operations import AccessPolicyAssignmentOperations
from ._private_endpoint_connections_operations import PrivateEndpointConnectionsOperations
from ._private_link_resources_operations import PrivateLinkResourcesOperations

from ._patch import __all__ as _patch_all
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "Operations",
    "OperationsStatusOperations",
    "RedisEnterpriseOperations",
    "DatabasesOperations",
    "AccessPolicyAssignmentOperations",
    "PrivateEndpointConnectionsOperations",
    "PrivateLinkResourcesOperations",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
