# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

from enum import Enum
import json
from typing import List, Any, Optional, Union

from azure.ai.client import _types

# from zoneinfo import ZoneInfo
from ..models._models import MessageDeltaChunk, RunStep, RunStepDeltaChunk, ThreadMessage, ThreadRun
from .. import models as _models

from azure.core.tracing import SpanKind  # type: ignore
from azure.core.settings import settings  # type: ignore
_Unset: Any = object()


_GEN_AI_MESSAGE_ID = "gen_ai.message.id"
_GEN_AI_MESSAGE_STATUS = "gen_ai.message.status"
_GEN_AI_THREAD_ID = "gen_ai.thread.id"
_GEN_AI_THREAD_RUN_ID = "gen_ai.thread.run.id"
_GEN_AI_AGENT_ID = "gen_ai.agent.id"
_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
_GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_THREAD_RUN_STATUS = "gen_ai.thread.run.status"
_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
_GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
_GEN_AI_REQUEST_MAX_INPUT_TOKENS = "gen_ai.request.max_input_tokens"
_GEN_AI_REQUEST_MAX_OUTPUT_TOKENS = "gen_ai.request.max_output_tokens"
_GEN_AI_SYSTEM = "gen_ai.system"
_SERVER_ADDRESS = "server.address"
_AZ_AI_AGENT_SYSTEM = "az.ai.agent" # TODO: decide
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
_GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
_GEN_AI_REQUEST_RESPONSE_FORMAT = "gen_ai.request.response_format"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
_GEN_AI_EVENT_CONTENT = "gen_ai.event.content"

try:
    # pylint: disable = no-name-in-module
    from azure.core.tracing import AbstractSpan, SpanKind  # type: ignore
    from opentelemetry.trace import StatusCode, Span

    _span_impl_type = settings.tracing_implementation()
except ModuleNotFoundError:
    _span_impl_type = None

_trace_agent_content: bool = True # TODO - read from somewhere
class _OperationName(Enum):
    CREATE_AGENT = "create_agent"
    CREATE_THREAD = "create_thread"
    CREATE_MESSAGE = "create_message"
    CREATE_THREAD_RUN = "create_thread_run"
    THREAD_RUN = "thread_run"
    SUBMIT_TOOL_OUTPUT = "submit_tool_output" # TODO: instrument
    EXECUTE_TOOL = "execute_tool"

def wrap_handler(handler: _models.AgentEventHandler, span: "AbstractSpan") -> "_models.AgentEventHandler":
    if span and span.span_instance.is_recording:
        return _EventHandlerWrapper(handler, span)
    return handler

def start_thread_run_span(
    operation_name: _OperationName,
    config: Any, # TODO: type AzureAIClientConfiguration
    thread_id: str,
    assistant_id: str,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    additional_instructions: Optional[str] = None,
    additional_messages: Optional[List[_models.ThreadMessage]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[List[_models.ToolDefinition]] = None,
    max_prompt_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional["_types.AgentsApiResponseFormatOption"] = None,
) -> "AbstractSpan":
    span = _start_span(operation_name,
                    config.project_name,
                    thread_id=thread_id,
                    agent_id=assistant_id,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_prompt_tokens=max_prompt_tokens,
                    max_completion_tokens=max_completion_tokens,
                    response_format=response_format)
    if span and span.span_instance.is_recording:
        _add_instructions_event(span, instructions, additional_instructions)

        if additional_messages:
            for message in additional_messages:
                _add_thread_message_event(span, message)
    return span

def start_create_agent_span(
    config: Any, # TODO: type AzureAIClientConfiguration
    model: str = _Unset,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[_models.ToolDefinition]] = None,
    tool_resources: Optional[_models.ToolResources] = None,
    toolset: Optional[_models.ToolSet] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    response_format: Optional["_types.AgentsApiResponseFormatOption"] = None,
) -> "AbstractSpan":
    span = _start_span(_OperationName.CREATE_AGENT,
                    config.project_name,
                    span_name=f"{_OperationName.CREATE_AGENT.value} {name}",
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format)
    if span and span.span_instance.is_recording:
        if name:
            span.add_attribute(_GEN_AI_AGENT_NAME, name)
        if description:
            span.add_attribute(_GEN_AI_AGENT_DESCRIPTION, description)
        _add_instructions_event(span, instructions, None)

    return span

def set_end_run(span: "AbstractSpan", run: ThreadRun) -> None:
    if span and span.span_instance.is_recording:
        span.add_attribute(_GEN_AI_THREAD_RUN_STATUS, run.status)
        if run.usage:
            span.add_attribute(_GEN_AI_USAGE_INPUT_TOKENS, run.usage.prompt_tokens)
            span.add_attribute(_GEN_AI_USAGE_OUTPUT_TOKENS, run.usage.completion_tokens)

def trace_tool_execution(
    tool_call_id: str,
    tool_name: str,
    thread_id: Optional[str] = None, # TODO: would be nice to have this, but need to propagate somehow
    agent_id: Optional[str] = None, # TODO: would be nice to have this, but need to propagate somehow
    run_id: Optional[str] = None # TODO: would be nice to have this, but need to propagate somehow
) -> "AbstractSpan":
    span = _start_span(_OperationName.EXECUTE_TOOL,
                    span_name=tool_name, thread_id=thread_id, agent_id=agent_id, run_id=run_id)
    if span is not None and span.span_instance.is_recording:
        span.add_attribute(_GEN_AI_TOOL_CALL_ID, tool_call_id)
        span.add_attribute(_GEN_AI_TOOL_NAME, tool_name)

    return span

def start_create_thread_span(
    config: Any, # TODO: type AzureAIClientConfiguration
    messages: Optional[List[_models.ThreadMessageOptions]] = None,
    tool_resources: Optional[_models.ToolResources] = None,
) -> "AbstractSpan":
    span = _start_span(_OperationName.CREATE_THREAD, config.project_name)
    if span and span.span_instance.is_recording:
        for message in messages or []:
            _add_thread_message_event(span, message)

    return span

# TODO: list messages is important to instrument too

def start_create_message_span(
        config: Any, # TODO: type AzureAIClientConfiguration
        thread_id: str,
        content: str,
        role: Union[str, _models.MessageRole] = _Unset,
        attachments: Optional[List[_models.MessageAttachment]] = None
) -> "AbstractSpan":
    role_str = _get_role(role)
    span = _start_span(_OperationName.CREATE_MESSAGE, config.project_name, thread_id=thread_id)
    if span and span.span_instance.is_recording:
        _add_message_event(span, role_str, content, attachments=attachments, thread_id=thread_id)
    return span

def _start_span(
        operation_name: _OperationName,
        server_address: str,
        span_name: str = None,
        thread_id: str = None,
        agent_id: str = None,
        model: str = None,
        temperature: str = None,
        top_p: str = None,
        max_prompt_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        response_format: Optional["_types.AgentsApiResponseFormatOption"] = None,
) -> "AbstractSpan":
    if _span_impl_type is None:
        return None

    span = _span_impl_type(name=span_name or operation_name.value, kind=SpanKind.CLIENT)

    if span and span.span_instance.is_recording:
        span.add_attribute(_GEN_AI_SYSTEM, _AZ_AI_AGENT_SYSTEM)
        span.add_attribute(_GEN_AI_OPERATION_NAME, operation_name.value)

        if server_address:
            span.add_attribute(_SERVER_ADDRESS, server_address)

        if thread_id:
            span.add_attribute(_GEN_AI_THREAD_ID, thread_id)

        if agent_id:
            span.add_attribute(_GEN_AI_AGENT_ID, agent_id)

        if model:
            span.add_attribute(_GEN_AI_REQUEST_MODEL, model)

        if temperature:
            span.add_attribute(_GEN_AI_REQUEST_TEMPERATURE, temperature)

        if top_p:
            span.add_attribute(_GEN_AI_REQUEST_TOP_P, top_p)

        if max_prompt_tokens:
            span.add_attribute(_GEN_AI_REQUEST_MAX_INPUT_TOKENS, max_prompt_tokens)

        if max_completion_tokens:
            span.add_attribute(_GEN_AI_REQUEST_MAX_OUTPUT_TOKENS, max_completion_tokens)

        if response_format:
            span.add_attribute(_GEN_AI_REQUEST_RESPONSE_FORMAT, response_format.value)

    return span

def _create_event_attributes(thread_id: str = None,
                            agent_id: str = None,
                            thread_run_id: str = None,
                            message_id: str = None,
                            message_status: str = None) -> dict:
    attrs = {_GEN_AI_SYSTEM: _AZ_AI_AGENT_SYSTEM}
    if thread_id:
        attrs[_GEN_AI_THREAD_ID] = thread_id

    if agent_id:
        attrs[_GEN_AI_AGENT_ID] = agent_id

    if thread_run_id:
        attrs[_GEN_AI_THREAD_RUN_ID] = thread_run_id

    if message_id:
        attrs[_GEN_AI_MESSAGE_ID] = message_id

    if message_status:
        attrs[_GEN_AI_MESSAGE_STATUS] = message_status

    return attrs

def _add_thread_message_event(span, message: _models.ThreadMessage) -> None:
    content_body = {}
    if _trace_agent_content:
        for content in message.content:
            typed_content = content.get(content.type, None)
            if typed_content:
                content_body[content.type] = typed_content.as_dict()

    _add_message_event(span,
                    _get_role(message.role),
                    content_body,
                    attachments=message.attachments,
                    thread_id=message.thread_id,
                    agent_id=message.assistant_id,
                    message_id=message.id,
                    thread_run_id=message.run_id,
                    message_status=message.status)

def _add_message_event(span,
                    role: str,
                    content: Any,
                    attachments: Optional[List[_models.MessageAttachment]] = None,
                    thread_id: Optional[str] = None,
                    agent_id: Optional[str] = None,
                    message_id: Optional[str] = None,
                    thread_run_id: Optional[str] = None,
                    message_status: Optional[str] = None) -> None:
    # TODO document new fields

    event_body = {}
    if _trace_agent_content:
        event_body["content"] = content
        if attachments:
            event_body["attachments"] = [attachment.as_dict() for attachment in attachments]

    event_body["role"] = role

    # TODO update attributes in semconv
    attributes = _create_event_attributes(thread_id=thread_id, agent_id=agent_id, message_id=message_id, thread_run_id=thread_run_id, message_status=message_status)
    attributes[_GEN_AI_EVENT_CONTENT] = json.dumps(event_body)
    span.span_instance.add_event(name=f"gen_ai.{role}.message", attributes=attributes)

def _add_instructions_event(span: "AbstractSpan", instructions: str, additional_instructions: str) -> None:
    if not instructions:
        return

    event_body = {}
    if _trace_agent_content and (instructions or additional_instructions):
        if instructions and additional_instructions:
            event_body["content"] = f"{instructions} {additional_instructions}"
        else:
            event_body["content"] = instructions or additional_instructions

    attributes = _create_event_attributes()
    attributes[_GEN_AI_EVENT_CONTENT] = json.dumps(event_body)
    span.span_instance.add_event(name=_GEN_AI_SYSTEM_MESSAGE, attributes=attributes)


def _get_role(role: Union[str, _models.MessageRole]) -> str:

    if role is None or role is _Unset:
        return _models.MessageRole.USER.value

    if isinstance(role, _models.MessageRole):
        return role.value

    return role

class _EventHandlerWrapper(_models.AgentEventHandler):
    def __init__(self, inner_handler: _models.AgentEventHandler, span: "AbstractSpan"):
        super().__init__()
        self.span = span
        self.inner_handler = inner_handler

    def on_message_delta(self, delta: "MessageDeltaChunk") -> None:
        if self.inner_handler:
            self.inner_handler.on_message_delta(delta)

    def on_thread_message(self, message: "ThreadMessage") -> None:
        if self.inner_handler:
            self.inner_handler.on_thread_message(message)
        if message.status == "completed" or message.status == "incomplete":
            _add_thread_message_event(self.span, message)

    def on_thread_run(self, run: "ThreadRun") -> None:
        if self.inner_handler:
            self.inner_handler.on_thread_run(run)
        # TODO: is it possible that it's called before run has ended?
        # TODO: make use of last error?
        set_end_run(self.span, run)

    def on_run_step(self, step: "RunStep") -> None:
        if self.inner_handler:
            self.inner_handler.on_run_step(step)

    def on_run_step_delta(self, delta: "RunStepDeltaChunk") -> None:
        if self.inner_handler:
            self.inner_handler.on_run_step_delta(delta)

    def on_error(self, data: str) -> None:
        if self.inner_handler:
            self.inner_handler.on_error(data)
        # TODO how can we get error type or any other useful information?
        self.span.span_instance.set_status(2, data)
        self.span.span_instance.end()

    def on_done(self) -> None:
        if self.inner_handler:
            self.inner_handler.on_done()
        self.span.span_instance.end()

    def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
        if self.inner_handler:
            self.inner_handler.on_unhandled_event(event_type, event_data)
