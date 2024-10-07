# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


class AgentsApiResponseFormatMode(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Represents the mode in which the model will handle the return format of a tool call."""

    AUTO = "auto"
    """Default value. Let the model handle the return format."""
    NONE = "none"
    """Setting the value to ``none``\\ , will result in a 400 Bad request."""


class AgentsApiToolChoiceOptionMode(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Specifies how the tool choice will be used."""

    NONE = "none"
    """The model will not call a function and instead generates a message."""
    AUTO = "auto"
    """The model can pick between generating a message or calling a function."""


class AgentsNamedToolChoiceType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Available tool types for agents named tools."""

    FUNCTION = "function"
    """Tool type ``function``"""
    CODE_INTERPRETER = "code_interpreter"
    """Tool type ``code_interpreter``"""
    FILE_SEARCH = "file_search"
    """Tool type ``file_search``"""


class AgentStreamEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Each event in a server-sent events stream has an ``event`` and ``data`` property:

    .. code-block::

       event: thread.created
       data: {"id": "thread_123", "object": "thread", ...}

    We emit events whenever a new object is created, transitions to a new state, or is being
    streamed in parts (deltas). For example, we emit ``thread.run.created`` when a new run
    is created, ``thread.run.completed`` when a run completes, and so on. When an Agent chooses
    to create a message during a run, we emit a ``thread.message.created event``\\ , a
    ``thread.message.in_progress`` event, many ``thread.message.delta`` events, and finally a
    ``thread.message.completed`` event.

    We may add additional events over time, so we recommend handling unknown events gracefully
    in your code.
    """

    THREAD_CREATED = "thread.created"
    """Event sent when a new thread is created. The data of this event is of type AgentThread"""
    THREAD_RUN_CREATED = "thread.run.created"
    """Event sent when a new run is created. The data of this event is of type ThreadRun"""
    THREAD_RUN_QUEUED = "thread.run.queued"
    """Event sent when a run moves to ``queued`` status. The data of this event is of type ThreadRun"""
    THREAD_RUN_IN_PROGRESS = "thread.run.in_progress"
    """Event sent when a run moves to ``in_progress`` status. The data of this event is of type
    ThreadRun"""
    THREAD_RUN_REQUIRES_ACTION = "thread.run.requires_action"
    """Event sent when a run moves to ``requires_action`` status. The data of this event is of type
    ThreadRun"""
    THREAD_RUN_COMPLETED = "thread.run.completed"
    """Event sent when a run is completed. The data of this event is of type ThreadRun"""
    THREAD_RUN_FAILED = "thread.run.failed"
    """Event sent when a run fails. The data of this event is of type ThreadRun"""
    THREAD_RUN_CANCELLING = "thread.run.cancelling"
    """Event sent when a run moves to ``cancelling`` status. The data of this event is of type
    ThreadRun"""
    THREAD_RUN_CANCELLED = "thread.run.cancelled"
    """Event sent when a run is cancelled. The data of this event is of type ThreadRun"""
    THREAD_RUN_EXPIRED = "thread.run.expired"
    """Event sent when a run is expired. The data of this event is of type ThreadRun"""
    THREAD_RUN_STEP_CREATED = "thread.run.step.created"
    """Event sent when a new thread run step is created. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_IN_PROGRESS = "thread.run.step.in_progress"
    """Event sent when a run step moves to ``in_progress`` status. The data of this event is of type
    RunStep"""
    THREAD_RUN_STEP_DELTA = "thread.run.step.delta"
    """Event sent when a run stepis being streamed. The data of this event is of type
    RunStepDeltaChunk"""
    THREAD_RUN_STEP_COMPLETED = "thread.run.step.completed"
    """Event sent when a run step is completed. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_FAILED = "thread.run.step.failed"
    """Event sent when a run step fails. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_CANCELLED = "thread.run.step.cancelled"
    """Event sent when a run step is cancelled. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_EXPIRED = "thread.run.step.expired"
    """Event sent when a run step is expired. The data of this event is of type RunStep"""
    THREAD_MESSAGE_CREATED = "thread.message.created"
    """Event sent when a new message is created. The data of this event is of type ThreadMessage"""
    THREAD_MESSAGE_IN_PROGRESS = "thread.message.in_progress"
    """Event sent when a message moves to ``in_progress`` status. The data of this event is of type
    ThreadMessage"""
    THREAD_MESSAGE_DELTA = "thread.message.delta"
    """Event sent when a message is being streamed. The data of this event is of type
    MessageDeltaChunk"""
    THREAD_MESSAGE_COMPLETED = "thread.message.completed"
    """Event sent when a message is completed. The data of this event is of type ThreadMessage"""
    THREAD_MESSAGE_INCOMPLETE = "thread.message.incomplete"
    """Event sent before a message is completed. The data of this event is of type ThreadMessage"""
    ERROR = "error"
    """Event sent when an error occurs, such as an internal server error or a timeout."""
    DONE = "done"
    """Event sent when the stream is done."""


class ApiResponseFormat(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible API response formats."""

    TEXT = "text"
    """``text`` format should be used for requests involving any sort of ToolCall."""
    JSON_OBJECT = "json_object"
    """Using ``json_object`` format will limit the usage of ToolCall to only functions."""


class AuthenticationType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """to do."""

    API_KEY = "ApiKey"
    """API Key authentication"""
    AAD = "AAD"
    """Entra ID authentication"""
    SAS = "SAS"
    """Shared Access Signature (SAS) authentication"""


class DoneEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Terminal event indicating the successful end of a stream."""

    DONE = "done"
    """Event sent when the stream is done."""


class EndpointType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The Type (or category) of the connection."""

    AZURE_OPEN_AI = "AzureOpenAI"
    """Azure OpenAI"""
    SERVERLESS = "Serverless"
    """Serverless API"""
    AGENT = "Agent"
    """Agent"""


class ErrorEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Terminal event indicating a server side error while streaming."""

    ERROR = "error"
    """Event sent when an error occurs, such as an internal server error or a timeout."""


class FilePurpose(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The possible values denoting the intended usage of a file."""

    FINE_TUNE = "fine-tune"
    """Indicates a file is used for fine tuning input."""
    FINE_TUNE_RESULTS = "fine-tune-results"
    """Indicates a file is used for fine tuning results."""
    AGENTS = "assistants"
    """Indicates a file is used as input to agents."""
    AGENTS_OUTPUT = "assistants_output"
    """Indicates a file is used as output by agents."""
    BATCH = "batch"
    """Indicates a file is used as input to ."""
    BATCH_OUTPUT = "batch_output"
    """Indicates a file is used as output by a vector store batch operation."""
    VISION = "vision"
    """Indicates a file is used as input to a vision operation."""


class FileState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The state of the file."""

    UPLOADED = "uploaded"
    """The file has been uploaded but it's not yet processed. This state is not returned by Azure
    OpenAI and exposed only for
    compatibility. It can be categorized as an inactive state."""
    PENDING = "pending"
    """The operation was created and is not queued to be processed in the future. It can be
    categorized as an inactive state."""
    RUNNING = "running"
    """The operation has started to be processed. It can be categorized as an active state."""
    PROCESSED = "processed"
    """The operation has successfully processed and is ready for consumption. It can be categorized as
    a terminal state."""
    ERROR = "error"
    """The operation has completed processing with a failure and cannot be further consumed. It can be
    categorized as a terminal state."""
    DELETING = "deleting"
    """The entity is in the process to be deleted. This state is not returned by Azure OpenAI and
    exposed only for compatibility.
    It can be categorized as an active state."""
    DELETED = "deleted"
    """The entity has been deleted but may still be referenced by other entities predating the
    deletion. It can be categorized as a
    terminal state."""


class IncompleteRunDetails(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The reason why the run is incomplete. This will point to which specific token limit was reached
    over the course of the run.
    """

    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    """Maximum completion tokens exceeded"""
    MAX_PROMPT_TOKENS = "max_prompt_tokens"
    """Maximum prompt tokens exceeded"""


class ListSortOrder(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The available sorting options when requesting a list of response objects."""

    ASCENDING = "asc"
    """Specifies an ascending sort order."""
    DESCENDING = "desc"
    """Specifies a descending sort order."""


class MessageIncompleteDetailsReason(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """A set of reasons describing why a message is marked as incomplete."""

    CONTENT_FILTER = "content_filter"
    """The run generating the message was terminated due to content filter flagging."""
    MAX_TOKENS = "max_tokens"
    """The run generating the message exhausted available tokens before completion."""
    RUN_CANCELLED = "run_cancelled"
    """The run generating the message was cancelled before completion."""
    RUN_FAILED = "run_failed"
    """The run generating the message failed."""
    RUN_EXPIRED = "run_expired"
    """The run generating the message expired."""


class MessageRole(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The possible values for roles attributed to messages in a thread."""

    USER = "user"
    """The role representing the end-user."""
    AGENT = "assistant"
    """The role representing the agent."""


class MessageStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The possible execution status values for a thread message."""

    IN_PROGRESS = "in_progress"
    """A run is currently creating this message."""
    INCOMPLETE = "incomplete"
    """This message is incomplete. See incomplete_details for more information."""
    COMPLETED = "completed"
    """This message was successfully completed by a run."""


class MessageStreamEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Message operation related streaming events."""

    THREAD_MESSAGE_CREATED = "thread.message.created"
    """Event sent when a new message is created. The data of this event is of type ThreadMessage"""
    THREAD_MESSAGE_IN_PROGRESS = "thread.message.in_progress"
    """Event sent when a message moves to ``in_progress`` status. The data of this event is of type
    ThreadMessage"""
    THREAD_MESSAGE_DELTA = "thread.message.delta"
    """Event sent when a message is being streamed. The data of this event is of type
    MessageDeltaChunk"""
    THREAD_MESSAGE_COMPLETED = "thread.message.completed"
    """Event sent when a message is completed. The data of this event is of type ThreadMessage"""
    THREAD_MESSAGE_INCOMPLETE = "thread.message.incomplete"
    """Event sent before a message is completed. The data of this event is of type ThreadMessage"""


class RunStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible values for the status of an agent thread run."""

    QUEUED = "queued"
    """Represents a run that is queued to start."""
    IN_PROGRESS = "in_progress"
    """Represents a run that is in progress."""
    REQUIRES_ACTION = "requires_action"
    """Represents a run that needs another operation, such as tool output submission, to continue."""
    CANCELLING = "cancelling"
    """Represents a run that is in the process of cancellation."""
    CANCELLED = "cancelled"
    """Represents a run that has been cancelled."""
    FAILED = "failed"
    """Represents a run that failed."""
    COMPLETED = "completed"
    """Represents a run that successfully completed."""
    EXPIRED = "expired"
    """Represents a run that expired before it could otherwise finish."""


class RunStepErrorCode(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible error code values attributable to a failed run step."""

    SERVER_ERROR = "server_error"
    """Represents a server error."""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    """Represents an error indicating configured rate limits were exceeded."""


class RunStepStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible values for the status of a run step."""

    IN_PROGRESS = "in_progress"
    """Represents a run step still in progress."""
    CANCELLED = "cancelled"
    """Represents a run step that was cancelled."""
    FAILED = "failed"
    """Represents a run step that failed."""
    COMPLETED = "completed"
    """Represents a run step that successfully completed."""
    EXPIRED = "expired"
    """Represents a run step that expired before otherwise finishing."""


class RunStepStreamEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Run step operation related streaming events."""

    THREAD_RUN_STEP_CREATED = "thread.run.step.created"
    """Event sent when a new thread run step is created. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_IN_PROGRESS = "thread.run.step.in_progress"
    """Event sent when a run step moves to ``in_progress`` status. The data of this event is of type
    RunStep"""
    THREAD_RUN_STEP_DELTA = "thread.run.step.delta"
    """Event sent when a run stepis being streamed. The data of this event is of type
    RunStepDeltaChunk"""
    THREAD_RUN_STEP_COMPLETED = "thread.run.step.completed"
    """Event sent when a run step is completed. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_FAILED = "thread.run.step.failed"
    """Event sent when a run step fails. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_CANCELLED = "thread.run.step.cancelled"
    """Event sent when a run step is cancelled. The data of this event is of type RunStep"""
    THREAD_RUN_STEP_EXPIRED = "thread.run.step.expired"
    """Event sent when a run step is expired. The data of this event is of type RunStep"""


class RunStepType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The possible types of run steps."""

    MESSAGE_CREATION = "message_creation"
    """Represents a run step to create a message."""
    TOOL_CALLS = "tool_calls"
    """Represents a run step that calls tools."""


class RunStreamEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Run operation related streaming events."""

    THREAD_RUN_CREATED = "thread.run.created"
    """Event sent when a new run is created. The data of this event is of type ThreadRun"""
    THREAD_RUN_QUEUED = "thread.run.queued"
    """Event sent when a run moves to ``queued`` status. The data of this event is of type ThreadRun"""
    THREAD_RUN_IN_PROGRESS = "thread.run.in_progress"
    """Event sent when a run moves to ``in_progress`` status. The data of this event is of type
    ThreadRun"""
    THREAD_RUN_REQUIRES_ACTION = "thread.run.requires_action"
    """Event sent when a run moves to ``requires_action`` status. The data of this event is of type
    ThreadRun"""
    THREAD_RUN_COMPLETED = "thread.run.completed"
    """Event sent when a run is completed. The data of this event is of type ThreadRun"""
    THREAD_RUN_FAILED = "thread.run.failed"
    """Event sent when a run fails. The data of this event is of type ThreadRun"""
    THREAD_RUN_CANCELLING = "thread.run.cancelling"
    """Event sent when a run moves to ``cancelling`` status. The data of this event is of type
    ThreadRun"""
    THREAD_RUN_CANCELLED = "thread.run.cancelled"
    """Event sent when a run is cancelled. The data of this event is of type ThreadRun"""
    THREAD_RUN_EXPIRED = "thread.run.expired"
    """Event sent when a run is expired. The data of this event is of type ThreadRun"""


class ThreadStreamEvent(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Thread operation related streaming events."""

    THREAD_CREATED = "thread.created"
    """Event sent when a new thread is created. The data of this event is of type AgentThread"""


class TruncationStrategy(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible truncation strategies for the thread."""

    AUTO = "auto"
    """Default value. Messages in the middle of the thread will be dropped to fit the context length
    of the model."""
    LAST_MESSAGES = "last_messages"
    """The thread will truncate to the ``lastMessages`` count of recent messages."""


class VectorStoreChunkingStrategyRequestType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Type of chunking strategy."""

    AUTO = "auto"
    STATIC = "static"


class VectorStoreChunkingStrategyResponseType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Type of chunking strategy."""

    OTHER = "other"
    STATIC = "static"


class VectorStoreExpirationPolicyAnchor(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Describes the relationship between the days and the expiration of this vector store."""

    LAST_ACTIVE_AT = "last_active_at"
    """The expiration policy is based on the last time the vector store was active."""


class VectorStoreFileBatchStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The status of the vector store file batch."""

    IN_PROGRESS = "in_progress"
    """The vector store is still processing this file batch."""
    COMPLETED = "completed"
    """the vector store file batch is ready for use."""
    CANCELLED = "cancelled"
    """The vector store file batch was cancelled."""
    FAILED = "failed"
    """The vector store file batch failed to process."""


class VectorStoreFileErrorCode(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Error code variants for vector store file processing."""

    INTERNAL_ERROR = "internal_error"
    """An internal error occurred."""
    FILE_NOT_FOUND = "file_not_found"
    """The file was not found."""
    PARSING_ERROR = "parsing_error"
    """The file could not be parsed."""
    UNHANDLED_MIME_TYPE = "unhandled_mime_type"
    """The file has an unhandled mime type."""


class VectorStoreFileStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Vector store file status."""

    IN_PROGRESS = "in_progress"
    """The file is currently being processed."""
    COMPLETED = "completed"
    """The file has been successfully processed."""
    FAILED = "failed"
    """The file has failed to process."""
    CANCELLED = "cancelled"
    """The file was cancelled."""


class VectorStoreFileStatusFilter(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Query parameter filter for vector store file retrieval endpoint."""

    IN_PROGRESS = "in_progress"
    """Retrieve only files that are currently being processed"""
    COMPLETED = "completed"
    """Retrieve only files that have been successfully processed"""
    FAILED = "failed"
    """Retrieve only files that have failed to process"""
    CANCELLED = "cancelled"
    """Retrieve only files that were cancelled"""


class VectorStoreStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Vector store possible status."""

    EXPIRED = "expired"
    """expired status indicates that this vector store has expired and is no longer available for use."""
    IN_PROGRESS = "in_progress"
    """in_progress status indicates that this vector store is still processing files."""
    COMPLETED = "completed"
    """completed status indicates that this vector store is ready for use."""
