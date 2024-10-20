# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

from enum import Enum
import json
from typing import List, Any, Optional, Union

from azure.ai.client import _types

from azure.ai.client._instrumentation._utils import * # pylint: disable=unused-wildcard-import
from azure.ai.client.models import _models
from azure.ai.client.models._enums import MessageRole, RunStepStatus
from azure.ai.client.models._models import MessageAttachment, MessageDeltaChunk, RunStep, RunStepDeltaChunk, RunStepFunctionToolCall, RunStepToolCallDetails, SubmitToolOutputsAction, ThreadMessage, ThreadMessageOptions, ThreadRun, ToolDefinition, ToolOutput, ToolResources
from azure.ai.client.models._patch import AgentEventHandler, ToolSet

_Unset: Any = object()

_trace_agent_content: bool = True # TODO - read from somewhere

def wrap_handler(handler: "_models.AgentEventHandler", span: "AbstractSpan") -> "_models.AgentEventHandler":
    if isinstance(handler, _EventHandlerWrapper):
        return handler

    if span and span.span_instance.is_recording:
        return _EventHandlerWrapper(handler, span)

    return handler

def start_thread_run_span(
    operation_name: OperationName,
    config: Any, # TODO: type AzureAIClientConfiguration
    thread_id: str,
    assistant_id: str,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    additional_instructions: Optional[str] = None,
    additional_messages: Optional[List[ThreadMessage]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[List[ToolDefinition]] = None,
    max_prompt_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional["_types.AgentsApiResponseFormatOption"] = None,
) -> "AbstractSpan":
    span = start_span(operation_name,
                    config.project_name,
                    thread_id=thread_id,
                    agent_id=assistant_id,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_prompt_tokens=max_prompt_tokens,
                    max_completion_tokens=max_completion_tokens,
                    response_format=response_format.value if response_format else None)
    if span and span.span_instance.is_recording:
        _add_instructions_event(span, instructions, additional_instructions)

        if additional_messages:
            for message in additional_messages:
                _add_thread_message_event(span, message)
    return span

def start_submit_tool_output_span(
    config: Any, # TODO: type AzureAIClientConfiguration
    thread_id: str,
    run_id: str,
    tool_outputs: List[ToolOutput] = _Unset,
    event_handler: Optional[AgentEventHandler] = None,
) -> "AbstractSpan":

    run_span = event_handler.span if isinstance(event_handler, _EventHandlerWrapper) else None
    recorded = _add_tool_message_events(run_span, tool_outputs)

    span = start_span(OperationName.SUBMIT_TOOL_OUTPUT,
                    config.project_name,
                    thread_id=thread_id,
                    run_id=run_id)
    if not recorded:
        _add_tool_message_events(span, tool_outputs)
    return span


def _add_tool_message_events(span, tool_outputs: List[ToolOutput]) -> bool:
    if span and span.span_instance.is_recording:
        for tool_output in tool_outputs:
            body = {"content": tool_output["output"], "id": tool_output["tool_call_id"]}
            span.span_instance.add_event("gen_ai.tool.message", {"gen_ai.event.content": json.dumps(body)})
        return True

    return False


def start_create_agent_span(
    config: Any, # TODO: type AzureAIClientConfiguration
    model: str = _Unset,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[ToolDefinition]] = None,
    tool_resources: Optional[ToolResources] = None,
    toolset: Optional[ToolSet] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    response_format: Optional["_types.AgentsApiResponseFormatOption"] = None,
) -> "AbstractSpan":
    span = start_span(OperationName.CREATE_AGENT,
                    config.project_name,
                    span_name=f"{OperationName.CREATE_AGENT.value} {name}",
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format.value if response_format else None)
    if span and span.span_instance.is_recording:
        if name:
            span.add_attribute(GEN_AI_AGENT_NAME, name)
        if description:
            span.add_attribute(GEN_AI_AGENT_DESCRIPTION, description)
        _add_instructions_event(span, instructions, None)

    return span

def set_end_run(span: "AbstractSpan", run: ThreadRun) -> None:
    if span and span.span_instance.is_recording:
        span.add_attribute(GEN_AI_THREAD_RUN_STATUS, run.status)
        if run.usage:
            span.add_attribute(GEN_AI_USAGE_INPUT_TOKENS, run.usage.prompt_tokens)
            span.add_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, run.usage.completion_tokens)

def start_create_thread_span(
    config: Any, # TODO: type AzureAIClientConfiguration
    messages: Optional[List[ThreadMessageOptions]] = None,
    tool_resources: Optional[ToolResources] = None,
) -> "AbstractSpan":
    span = start_span(OperationName.CREATE_THREAD, config.project_name)
    if span and span.span_instance.is_recording:
        for message in messages or []:
            _add_thread_message_event(span, message)

    return span

# TODO: list messages is important to instrument too

def start_create_message_span(
        config: Any, # TODO: type AzureAIClientConfiguration
        thread_id: str,
        content: str,
        role: Union[str, MessageRole] = _Unset,
        attachments: Optional[List[MessageAttachment]] = None
) -> "AbstractSpan":
    role_str = _get_role(role)
    span = start_span(OperationName.CREATE_MESSAGE, config.project_name, thread_id=thread_id)
    if span and span.span_instance.is_recording:
        _add_message_event(span, role_str, content, attachments=attachments, thread_id=thread_id)
    return span

def _create_event_attributes(thread_id: str = None,
                            agent_id: str = None,
                            thread_run_id: str = None,
                            message_id: str = None,
                            message_status: str = None,
                            usage: Optional[_models.RunStepCompletionUsage] = None) -> dict:
    attrs = {GEN_AI_SYSTEM: AZ_AI_AGENT_SYSTEM}
    if thread_id:
        attrs[GEN_AI_THREAD_ID] = thread_id

    if agent_id:
        attrs[GEN_AI_AGENT_ID] = agent_id

    if thread_run_id:
        attrs[GEN_AI_THREAD_RUN_ID] = thread_run_id

    if message_id:
        attrs[GEN_AI_MESSAGE_ID] = message_id

    if message_status:
        attrs[GEN_AI_MESSAGE_STATUS] = message_status

    if usage:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = usage.prompt_tokens
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = usage.completion_tokens

    return attrs

def _add_thread_message_event(span, message: ThreadMessage, usage: Optional[_models.RunStepCompletionUsage] = None) -> None:
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
                    message_status=message.status,
                    usage=usage)

def _add_message_event(span,
                    role: str,
                    content: Any,
                    attachments: Optional[List[MessageAttachment]] = None,
                    thread_id: Optional[str] = None,
                    agent_id: Optional[str] = None,
                    message_id: Optional[str] = None,
                    thread_run_id: Optional[str] = None,
                    message_status: Optional[str] = None,
                    usage: Optional[_models.RunStepCompletionUsage] = None) -> None:
    # TODO document new fields

    event_body = {}
    if _trace_agent_content:
        event_body["content"] = content
        if attachments:
            event_body["attachments"] = [attachment.as_dict() for attachment in attachments]

    event_body["role"] = role

    # TODO update attributes in semconv
    attributes = _create_event_attributes(thread_id=thread_id,
                                        agent_id=agent_id,
                                        message_id=message_id,
                                        thread_run_id=thread_run_id,
                                        message_status=message_status,
                                        usage=usage)
    attributes[GEN_AI_EVENT_CONTENT] = json.dumps(event_body)
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
    attributes[GEN_AI_EVENT_CONTENT] = json.dumps(event_body)
    span.span_instance.add_event(name=GEN_AI_SYSTEM_MESSAGE, attributes=attributes)


def _get_role(role: Union[str, MessageRole]) -> str:

    if role is None or role is _Unset:
        return "user"

    if isinstance(role, MessageRole):
        return role.value

    return role


def _add_tool_assistant_message_event(span, step: RunStep) -> None:
    # we prob want a new event for it
    tool_calls = [{"id": t.id,
                    "type": t.type,
                    "function" : {
                        "name": t.function.name,
                        "arguments": json.loads(t.function.arguments)
                    } if isinstance(t, RunStepFunctionToolCall) else None,
                } for t in step.step_details.tool_calls]

    attributes = _create_event_attributes(thread_id=step.thread_id,
                                        agent_id=step.assistant_id,
                                        thread_run_id=step.run_id,
                                        message_status=step.status,
                                        usage=step.usage)

    attributes[GEN_AI_EVENT_CONTENT] = json.dumps({"tool_calls": tool_calls})
    span.span_instance.add_event(name=f"gen_ai.assistant.message", attributes=attributes)

class _EventHandlerWrapper(AgentEventHandler):
    def __init__(self, inner_handler: AgentEventHandler, span: "AbstractSpan"):
        super().__init__()
        self.span = span
        self.inner_handler = inner_handler
        self.ended = False
        self.last_run = None
        self.last_message = None

    def on_message_delta(self, delta: "MessageDeltaChunk") -> None:
        if self.inner_handler:
            self.inner_handler.on_message_delta(delta)

    def on_thread_message(self, message: "ThreadMessage") -> None:
        if self.inner_handler:
            self.inner_handler.on_thread_message(message)

        if message.status == "completed" or message.status == "incomplete":
            self.last_message = message

    def on_thread_run(self, run: "ThreadRun") -> None:
        if self.inner_handler:
            self.inner_handler.on_thread_run(run)
        self.last_run = run

    def on_run_step(self, step: "RunStep") -> None:
        if self.inner_handler:
            self.inner_handler.on_run_step(step)

        if step.status == RunStepStatus.IN_PROGRESS:
            return

        # todo - report errors for failure statuses here and in run ?
        if step.type == "tool_calls" and isinstance(step.step_details, RunStepToolCallDetails):
            _add_tool_assistant_message_event(self.span, step)
        elif step.type == "message_creation" and step.status == RunStepStatus.COMPLETED:
            _add_thread_message_event(self.span, self.last_message, step.usage)
            self.last_message = None

    def on_run_step_delta(self, delta: "RunStepDeltaChunk") -> None:
        if self.inner_handler:
            self.inner_handler.on_run_step_delta(delta)

    def on_error(self, data: str) -> None:
        if self.inner_handler:
            self.inner_handler.on_error(data)

    def on_done(self) -> None:
        if self.inner_handler:
            self.inner_handler.on_done()
        # it could be called multiple tines (for each step) __exit__

    def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
        if self.inner_handler:
            self.inner_handler.on_unhandled_event(event_type, event_data)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.ended:
            self.ended = True
            set_end_run(self.span, self.last_run)

            if self.last_run.last_error:
                self.span.set_status(StatusCode.ERROR, self.last_run.last_error.message)
                self.span.add_attribute(ERROR_TYPE, self.last_run.last_error.code)

            self.span.__exit__(exc_type, exc_val, exc_tb)
            self.span.finish()
