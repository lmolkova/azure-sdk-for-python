# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._models import AudioNotificationContent
from ._models import DocumentNotificationContent
from ._models import ImageNotificationContent
from ._models import MessageReceipt
from ._models import MessageTemplate
from ._models import MessageTemplateBindings
from ._models import MessageTemplateDocument
from ._models import MessageTemplateImage
from ._models import MessageTemplateItem
from ._models import MessageTemplateLocation
from ._models import MessageTemplateQuickAction
from ._models import MessageTemplateText
from ._models import MessageTemplateValue
from ._models import MessageTemplateVideo
from ._models import NotificationContent
from ._models import SendMessageResult
from ._models import TemplateNotificationContent
from ._models import TextNotificationContent
from ._models import VideoNotificationContent
from ._models import WhatsAppMessageTemplateBindings
from ._models import WhatsAppMessageTemplateBindingsButton
from ._models import WhatsAppMessageTemplateBindingsComponent
from ._models import WhatsAppMessageTemplateItem

from ._enums import CommunicationMessageKind
from ._enums import CommunicationMessagesChannel
from ._enums import MessageTemplateBindingsKind
from ._enums import MessageTemplateStatus
from ._enums import MessageTemplateValueKind
from ._enums import RepeatabilityResult
from ._enums import WhatsAppMessageButtonSubType
from ._patch import __all__ as _patch_all
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "AudioNotificationContent",
    "DocumentNotificationContent",
    "ImageNotificationContent",
    "MessageReceipt",
    "MessageTemplate",
    "MessageTemplateBindings",
    "MessageTemplateDocument",
    "MessageTemplateImage",
    "MessageTemplateItem",
    "MessageTemplateLocation",
    "MessageTemplateQuickAction",
    "MessageTemplateText",
    "MessageTemplateValue",
    "MessageTemplateVideo",
    "NotificationContent",
    "SendMessageResult",
    "TemplateNotificationContent",
    "TextNotificationContent",
    "VideoNotificationContent",
    "WhatsAppMessageTemplateBindings",
    "WhatsAppMessageTemplateBindingsButton",
    "WhatsAppMessageTemplateBindingsComponent",
    "WhatsAppMessageTemplateItem",
    "CommunicationMessageKind",
    "CommunicationMessagesChannel",
    "MessageTemplateBindingsKind",
    "MessageTemplateStatus",
    "MessageTemplateValueKind",
    "RepeatabilityResult",
    "WhatsAppMessageButtonSubType",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
