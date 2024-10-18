# pylint: disable=too-many-lines
# # ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
import os
import json
import time
import functools
import datetime
import logging
import sys

from azure.ai.client import AzureAIClient
from azure.ai.client.models import FunctionTool, CodeInterpreterTool, FileSearchTool, ToolSet
from azure.core.pipeline.transport import RequestsTransport
from devtools_testutils import AzureRecordedTestCase, EnvironmentVariableLoader, recorded_by_proxy
from azure.core.exceptions import AzureError, ServiceRequestError, HttpResponseError
from azure.ai.client.models import FunctionTool
from azure.identity import DefaultAzureCredential

# TODO clean this up / get rid of anything not in use

"""
issues I've noticed with the code: 
    delete_thread(thread.id) fails
    cancel_thread(thread.id) expires/times out occasionally
    added time.sleep() to the beginning of my last few tests to avoid limits
    when using the endpoint from Howie, delete_agent(agent.id) did not work but would not cause an error
"""

# Set to True to enable SDK logging
LOGGING_ENABLED = True

if LOGGING_ENABLED:
    # Create a logger for the 'azure' SDK
    # See https://docs.python.org/3/library/logging.html
    logger = logging.getLogger("azure")
    logger.setLevel(logging.DEBUG)  # INFO or DEBUG

    # Configure a console output
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)


agentClientPreparer = functools.partial(
    EnvironmentVariableLoader,
    "azure_ai_client",
    project_connection_string_agents_tests="https://foo.bar.some-domain.ms;00000000-0000-0000-0000-000000000000;rg-resour-cegr-oupfoo1;abcd-abcdabcdabcda-abcdefghijklm",
)
"""
agentClientPreparer = functools.partial(
    EnvironmentVariableLoader,
    'azure_ai_client',
    azure_ai_client_host_name="https://foo.bar.some-domain.ms",
    azure_ai_client_subscription_id="00000000-0000-0000-0000-000000000000",
    azure_ai_client_resource_group_name="rg-resour-cegr-oupfoo1",
    azure_ai_client_workspace_name="abcd-abcdabcdabcda-abcdefghijklm",
)
"""


# create tool for agent use
def fetch_current_datetime_live():
    """
    Get the current time as a JSON string.

    :return: Static time string so that test recordings work.
    :rtype: str
    """
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_json = json.dumps({"current_time": current_datetime})
    return time_json


# create tool for agent use
def fetch_current_datetime_recordings():
    """
    Get the current time as a JSON string.

    :return: Static time string so that test recordings work.
    :rtype: str
    """
    time_json = json.dumps({"current_time": "2024-10-10 12:30:19"})
    return time_json


# Statically defined user functions for fast reference
user_functions_recording = {"fetch_current_datetime": fetch_current_datetime_recordings}
user_functions_live = {"fetch_current_datetime": fetch_current_datetime_live}


# The test class name needs to start with "Test" to get collected by pytest
class TestagentClient(AzureRecordedTestCase):

    # helper function: create client and using environment variables
    def create_client(self, **kwargs):
        # fetch environment variables
        connection_string = kwargs.pop("project_connection_string_agents_tests")
        credential = self.get_credential(AzureAIClient, is_async=False)

        # create and return client
        client = AzureAIClient.from_connection_string(
            credential=credential,
            connection=connection_string,
        )

        return client

    # for debugging purposes: if a test fails and its agent has not been deleted, it will continue to show up in the agents list
    """
    # NOTE: this test should not be run against a shared resource, as it will delete all agents
    @agentClientPreparer()
    @recorded_by_proxy
    def test_clear_client(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # clear agent list
        agents = client.agents.list_agents().data
        for agent in agents:
            client.agents.delete_agent(agent.id)
        assert client.agents.list_agents().data.__len__() == 0
       
        # close client
        client.close()
    """

    #     # **********************************************************************************
    #     #
    #     #                               UNIT TESTS
    #     #
    #     # **********************************************************************************

    # # **********************************************************************************
    # #
    # #                      HAPPY PATH SERVICE TESTS - agent APIs
    # #
    # # **********************************************************************************

    # test client creation
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_client(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # close client
        client.close()

    # test agent creation and deletion
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_delete_agent(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)
        print("Created client")

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test agent creation with tools
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_agent_with_tools(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # initialize agent functions
        functions = FunctionTool(functions=user_functions_recording)

        # create agent with tools
        agent = client.agents.create_agent(
            model="gpt-4o", name="my-agent", instructions="You are helpful agent", tools=functions.definitions
        )
        assert agent.id
        print("Created agent, agent ID", agent.id)
        assert agent.tools
        assert agent.tools[0]["function"]["name"] == functions.definitions[0]["function"]["name"]
        print("Tool successfully submitted:", functions.definitions[0]["function"]["name"])

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    @agentClientPreparer()
    @recorded_by_proxy
    def test_update_agent(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id

        # update agent and confirm changes went through
        agent.update(name="my-agent2", instructions="You are helpful agent")
        assert agent.name
        assert agent.name == "my-agent2"

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    """
    DISABLED: can't perform consistently on shared resource
    @agentClientPreparer()
    @recorded_by_proxy
    def test_agent_list(self, **kwargs):
        # create client and ensure there are no previous agents
        client = self.create_client(**kwargs)
        list_length = client.agents.list_agents().data.__len__()

        # create agent and check that it appears in the list
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert client.agents.list_agents().data.__len__() == list_length + 1
        assert client.agents.list_agents().data[0].id == agent.id 

        # create second agent and check that it appears in the list
        agent2 = client.agents.create_agent(model="gpt-4o", name="my-agent2", instructions="You are helpful agent")
        assert client.agents.list_agents().data.__len__() == list_length + 2
        assert client.agents.list_agents().data[0].id == agent.id or client.agents.list_agents().data[1].id == agent.id 

        # delete agents and check list 
        client.agents.delete_agent(agent.id)
        assert client.agents.list_agents().data.__len__() == list_length + 1
        assert client.agents.list_agents().data[0].id == agent2.id 

        client.agents.delete_agent(agent2.id)
        assert client.agents.list_agents().data.__len__() == list_length 
        print("Deleted agents")

        # close client
        client.close()
        """

    # **********************************************************************************
    #
    #                      HAPPY PATH SERVICE TESTS - Thread APIs
    #
    # **********************************************************************************

    # test creating thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_thread(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        # assert isinstance(thread, agentThread) TODO finish this ! need to import agentThread from _models
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test getting thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_get_thread(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # get thread
        thread2 = client.agents.get_thread(thread.id)
        assert thread2.id
        assert thread.id == thread2.id
        print("Got thread, thread ID", thread2.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    """
    TODO what  can I update a thread with? 
    # test updating thread  
    @agentClientPreparer()
    @recorded_by_proxy
    def test_update_thread(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # update thread
        client.agents.update_thread(thread.id, ) # TODO what can we update it with? 
        assert not thread

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()
    """

    """
    # TODO this test is failing?  client.agents.delete_thread(thread.id) isn't working
    # status_code = 404, response = <HttpResponse: 404 Resource Not Found, Content-Type: application/json>
    # error_map = {304: <class 'azure.core.exceptions.ResourceNotModifiedError'>, 401: <class 'azure.core.exceptions.ClientAuthenticatio..., 404: <class 'azure.core.exceptions.ResourceNotFoundError'>, 409: <class 'azure.core.exceptions.ResourceExistsError'>}
    
    # test deleting thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_delete_thread(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        # assert isinstance(thread, agentThread) TODO finish this ! need to import agentThread from _models
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # delete thread
        deletion_status = client.agents.delete_thread(thread.id)
        # assert not thread

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()
    """

    # # **********************************************************************************
    # #
    # #                      HAPPY PATH SERVICE TESTS - Message APIs
    # #
    # # **********************************************************************************

    # test creating message in a thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_message(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a joke")
        assert message.id
        print("Created message, message ID", message.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test creating multiple messages in a thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_multiple_messages(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create messages
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a joke")
        assert message.id
        print("Created message, message ID", message.id)
        message2 = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me another joke")
        assert message2.id
        print("Created message, message ID", message2.id)
        message3 = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a third joke")
        assert message3.id
        print("Created message, message ID", message3.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test listing messages in a thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_list_messages(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # check that initial message list is empty
        messages0 = client.agents.list_messages(thread_id=thread.id)
        print(messages0.data)
        assert messages0.data.__len__() == 0

        # create messages and check message list for each one
        message1 = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a joke")
        assert message1.id
        print("Created message, message ID", message1.id)
        messages1 = client.agents.list_messages(thread_id=thread.id)
        assert messages1.data.__len__() == 1
        assert messages1.data[0].id == message1.id

        message2 = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me another joke")
        assert message2.id
        print("Created message, message ID", message2.id)
        messages2 = client.agents.list_messages(thread_id=thread.id)
        assert messages2.data.__len__() == 2
        assert messages2.data[0].id == message2.id or messages2.data[1].id == message2.id

        message3 = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a third joke")
        assert message3.id
        print("Created message, message ID", message3.id)
        messages3 = client.agents.list_messages(thread_id=thread.id)
        assert messages3.data.__len__() == 3
        assert (
            messages3.data[0].id == message3.id
            or messages3.data[1].id == message2.id
            or messages3.data[2].id == message2.id
        )

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test getting message in a thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_get_message(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a joke")
        assert message.id
        print("Created message, message ID", message.id)

        # get message
        message2 = client.agents.get_message(thread_id=thread.id, message_id=message.id)
        assert message2.id
        assert message.id == message2.id
        print("Got message, message ID", message.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    """
    TODO format the updated body
    # test updating message in a thread
    @agentClientPreparer()
    @recorded_by_proxy
    def test_update_message(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a joke")
        assert message.id
        print("Created message, message ID", message.id)

        # update message
        body_json = json.dumps # TODO format body into json -- figure out what the message looks like so I can update it (might be in that picture) 
        client.agents.update_message(thread_id=thread.id, message_id=message.id, body=)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()
    """

    # # **********************************************************************************
    # #
    # #                      HAPPY PATH SERVICE TESTS - Run APIs
    # #
    # # **********************************************************************************

    # test creating run
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_run(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test getting run
    @agentClientPreparer()
    @recorded_by_proxy
    def test_get_run(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        # get run
        run2 = client.agents.get_run(thread_id=thread.id, run_id=run.id)
        assert run2.id
        assert run.id == run2.id
        print("Got run, run ID", run2.id)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # TODO fix bc sometimes it works? and sometimes it doesnt?
    # test sucessful run status     TODO test for cancelled/unsucessful runs
    @agentClientPreparer()
    @recorded_by_proxy
    def test_run_status(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a joke")
        assert message.id
        print("Created message, message ID", message.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        # check status
        assert run.status in [
            "queued",
            "in_progress",
            "requires_action",
            "cancelling",
            "cancelled",
            "failed",
            "completed",
            "expired",
        ]
        while run.status in ["queued", "in_progress", "requires_action"]:
            # wait for a second
            time.sleep(1)
            run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
            print("Run status:", run.status)

        assert run.status in ["cancelled", "failed", "completed", "expired"]
        print("Run completed with status:", run.status)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    """
    # TODO another, but check that the number of runs decreases after cancelling runs
    # TODO can each thread only support one run? 
    # test listing runs
    @agentClientPreparer()
    @recorded_by_proxy
    def test_list_runs(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # check list for current runs
        runs0 = client.agents.list_runs(thread_id=thread.id)
        assert runs0.data.__len__() == 0

        # create run and check list
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)
        runs1 = client.agents.list_runs(thread_id=thread.id)
        assert runs1.data.__len__() == 1
        assert runs1.data[0].id == run.id

        # create second run
        run2 = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run2.id
        print("Created run, run ID", run2.id)
        runs2 = client.agents.list_runs(thread_id=thread.id)
        assert runs2.data.__len__() == 2
        assert runs2.data[0].id == run2.id or runs2.data[1].id == run2.id

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()
    """

    """
    # TODO figure out what to update the run with
    # test updating run
    @agentClientPreparer()
    @recorded_by_proxy
    def test_update_run(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        # update run
        body = json.dumps({'todo': 'placeholder'})
        client.agents.update_run(thread_id=thread.id, run_id=run.id, body=body)

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()
    """

    # test submitting tool outputs to run
    @agentClientPreparer()
    @recorded_by_proxy
    def test_submit_tool_outputs_to_run(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # Initialize agent tools
        functions = FunctionTool(user_functions_recording)
        code_interpreter = CodeInterpreterTool()

        toolset = ToolSet()
        toolset.add(functions)
        toolset.add(code_interpreter)

        # create agent
        agent = client.agents.create_agent(
            model="gpt-4o", name="my-agent", instructions="You are helpful agent", toolset=toolset
        )
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, what time is it?")
        assert message.id
        print("Created message, message ID", message.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        # check that tools are uploaded
        assert run.tools
        assert run.tools[0]["function"]["name"] == functions.definitions[0]["function"]["name"]
        print("Tool successfully submitted:", functions.definitions[0]["function"]["name"])

        # check status
        assert run.status in [
            "queued",
            "in_progress",
            "requires_action",
            "cancelling",
            "cancelled",
            "failed",
            "completed",
            "expired",
        ]
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = client.agents.get_run(thread_id=thread.id, run_id=run.id)

            # check if tools are needed
            if run.status == "requires_action" and run.required_action.submit_tool_outputs:
                print("Requires action: submit tool outputs")
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                if not tool_calls:
                    print(
                        "No tool calls provided - cancelling run"
                    )  # TODO how can i make sure that it wants tools? should i have some kind of error message?
                    client.agents.cancel_run(thread_id=thread.id, run_id=run.id)
                    break

                # submit tool outputs to run
                tool_outputs = toolset.execute_tool_calls(tool_calls)  # TODO issue somewhere here
                print("Tool outputs:", tool_outputs)
                if tool_outputs:
                    client.agents.submit_tool_outputs_to_run(
                        thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                    )

            print("Current run status:", run.status)

        print("Run completed with status:", run.status)

        # check that messages used the tool
        messages = client.agents.list_messages(thread_id=thread.id, run_id=run.id)
        tool_message = messages["data"][0]["content"][0]["text"]["value"]
        hour12 = time.strftime("%H")
        hour24 = time.strftime("%I")
        minute = time.strftime("%M")
        assert hour12 + ":" + minute in tool_message or hour24 + ":" + minute
        print("Used tool_outputs")

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    """
    # DISABLED: rewrite to ensure run is not complete when cancel_run is called
    # test cancelling run
    @agentClientPreparer()
    @recorded_by_proxy
    def test_cancel_run(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, what time is it?")
        assert message.id
        print("Created message, message ID", message.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        # check status and cancel
        assert run.status in ["queued", "in_progress", "requires_action"]
        client.agents.cancel_run(thread_id=thread.id, run_id=run.id)

        while run.status in ["queued", "cancelling"]:
            time.sleep(1)
            run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
            print("Current run status:", run.status)
        assert run.status == "cancelled"
        print("Run cancelled")

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()
        """

    # test create thread and run
    @agentClientPreparer()
    @recorded_by_proxy
    def test_create_thread_and_run(self, **kwargs):
        time.sleep(26)
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread and run
        run = client.agents.create_thread_and_run(assistant_id=agent.id)
        assert run.id
        assert run.thread_id
        print("Created run, run ID", run.id)

        # get thread
        thread = client.agents.get_thread(run.thread_id)
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # check status
        assert run.status in [
            "queued",
            "in_progress",
            "requires_action",
            "cancelling",
            "cancelled",
            "failed",
            "completed",
            "expired",
        ]
        while run.status in ["queued", "in_progress", "requires_action"]:
            # wait for a second
            time.sleep(1)
            run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
            # assert run.status in ["queued", "in_progress", "requires_action", "completed"]
            print("Run status:", run.status)

        assert run.status == "completed"
        print("Run completed")

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test listing run steps
    @agentClientPreparer()
    @recorded_by_proxy
    def test_list_run_step(self, **kwargs):

        time.sleep(50)
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(thread_id=thread.id, role="user", content="Hello, what time is it?")
        assert message.id
        print("Created message, message ID", message.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        steps = client.agents.list_run_steps(thread_id=thread.id, run_id=run.id)
        # commenting assertion out below, do we know exactly when run starts?
        # assert steps['data'].__len__() == 0

        # check status
        assert run.status in ["queued", "in_progress", "requires_action", "completed"]
        while run.status in ["queued", "in_progress", "requires_action"]:
            # wait for a second
            time.sleep(1)
            run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
            assert run.status in ["queued", "in_progress", "requires_action", "completed"]
            print("Run status:", run.status)
            steps = client.agents.list_run_steps(thread_id=thread.id, run_id=run.id)
            assert steps["data"].__len__() > 0  # TODO what else should we look at?

        assert run.status == "completed"
        print("Run completed")

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # test getting run step
    # TODO where are step ids from
    @agentClientPreparer()
    @recorded_by_proxy
    def test_get_run_step(self, **kwargs):
        # create client
        client = self.create_client(**kwargs)
        assert isinstance(client, AzureAIClient)

        # create agent
        agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        assert agent.id
        print("Created agent, agent ID", agent.id)

        # create thread
        thread = client.agents.create_thread()
        assert thread.id
        print("Created thread, thread ID", thread.id)

        # create message
        message = client.agents.create_message(
            thread_id=thread.id, role="user", content="Hello, can you tell me a joke?"
        )
        assert message.id
        print("Created message, message ID", message.id)

        # create run
        run = client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        assert run.id
        print("Created run, run ID", run.id)

        if run.status == "failed":
            assert run.last_error
            print(run.last_error)
            print("FAILED HERE")

        # check status
        assert run.status in ["queued", "in_progress", "requires_action", "completed"]
        while run.status in ["queued", "in_progress", "requires_action"]:
            # wait for a second
            time.sleep(1)
            run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
            if run.status == "failed":
                assert run.last_error
                print(run.last_error)
                print("FAILED HERE")
            assert run.status in ["queued", "in_progress", "requires_action", "completed"]
            print("Run status:", run.status)

        # list steps, check that get_run_step works with first step_id
        steps = client.agents.list_run_steps(thread_id=thread.id, run_id=run.id)
        assert steps["data"].__len__() > 0
        step = steps["data"][0]
        get_step = client.agents.get_run_step(thread_id=thread.id, run_id=run.id, step_id=step.id)
        assert step == get_step

        # delete agent and close client
        client.agents.delete_agent(agent.id)
        print("Deleted agent")
        client.close()

    # # **********************************************************************************
    # #
    # #                      HAPPY PATH SERVICE TESTS - Streaming APIs
    # #
    # # **********************************************************************************

    # # **********************************************************************************
    # #
    # #         NEGATIVE TESTS - TODO idk what goes here
    # #
    # # **********************************************************************************

    """
    # DISABLED, PASSES LIVE ONLY: recordings don't capture DNS lookup errors
    # test agent creation and deletion
    @agentClientPreparer()
    @recorded_by_proxy
    def test_negative_create_delete_agent(self, **kwargs):
        # create client using bad endpoint
        bad_connection_string = "https://foo.bar.some-domain.ms;00000000-0000-0000-0000-000000000000;rg-resour-cegr-oupfoo1;abcd-abcdabcdabcda-abcdefghijklm"

        credential = self.get_credential(AzureAIClient, is_async=False)
        client = AzureAIClient.from_connection_string(
            credential=credential,
            connection=bad_connection_string,
        )
        
        # attempt to create agent with bad client
        exception_caught = False
        try:
            agent = client.agents.create_agent(model="gpt-4o", name="my-agent", instructions="You are helpful agent")
        # check for error (will not have a status code since it failed on request -- no response was recieved)
        except (ServiceRequestError, HttpResponseError) as e:
            exception_caught = True
            if type(e) == ServiceRequestError:
                assert e.message
                assert "failed to resolve 'foo.bar.some-domain.ms'" in e.message.lower()
            else:
                assert "No such host is known" and "foo.bar.some-domain.ms" in str(e)
        
        # close client and confirm an exception was caught
        client.close()
        assert exception_caught
    """
