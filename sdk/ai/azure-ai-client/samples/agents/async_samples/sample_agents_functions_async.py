# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
FILE: sample_agents_functions_async.py

DESCRIPTION:
    This sample demonstrates how to use agent operations with custom functions from
    the Azure Agents service using a asynchronous client.

USAGE:
    python sample_agents_functions_async.py

    Before running the sample:

    pip install azure.ai.client azure-identity

    Set this environment variables with your own values:
    AI_CLIENT_CONNECTION_STRING - the Azure AI Project connection string, as found in your AI Studio Project.
"""
import asyncio
import time

from azure.ai.client.aio import AzureAIClient
from azure.ai.client.models import AsyncFunctionTool, RequiredFunctionToolCall, SubmitToolOutputsAction
from azure.identity import DefaultAzureCredential

import os

from user_async_functions import user_async_functions


async def main():
    # Create an Azure AI Client from a connection string, copied from your AI Studio project.
    # At the moment, it should be in the format "<HostName>;<AzureSubscriptionId>;<ResourceGroup>;<HubName>"
    # Customer needs to login to Azure subscription via Azure CLI and set the environment variables

    ai_client = AzureAIClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ["AI_CLIENT_CONNECTION_STRING"]
    )

    # Or, you can create the Azure AI Client by giving all required parameters directly
    """
    ai_client = AzureAIClient(
        credential=DefaultAzureCredential(),
        host_name=os.environ["AI_CLIENT_HOST_NAME"],
        subscription_id=os.environ["AI_CLIENT_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AI_CLIENT_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["AI_CLIENT_WORKSPACE_NAME"],
        logging_enable=True, # Optional. Remove this line if you don't want to show how to enable logging
    )
    """    
    
    async with ai_client:
        # Initialize assistant functions
        functions = AsyncFunctionTool(functions=user_async_functions)

        # Create agent
        agent = await ai_client.agents.create_agent(
            model="gpt-4-1106-preview", name="my-assistant", instructions="You are helpful assistant", tools=functions.definitions
        )
        print(f"Created agent, agent ID: {agent.id}")

        # Create thread for communication
        thread = await ai_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        # Create and send message
        message = await ai_client.agents.create_message(thread_id=thread.id, role="user", content="Hello, what's the time?")
        print(f"Created message, ID: {message.id}")

        # Create and run assistant task
        run = await ai_client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        print(f"Created run, ID: {run.id}")

        # Polling loop for run status
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(4)
            run = await ai_client.agents.get_run(thread_id=thread.id, run_id=run.id)

            if run.status == "requires_action" and  isinstance(run.required_action, SubmitToolOutputsAction):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                if not tool_calls:
                    print("No tool calls provided - cancelling run")
                    await ai_client.agents.cancel_run(thread_id=thread.id, run_id=run.id)
                    break

                tool_outputs = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, RequiredFunctionToolCall):
                        try:
                            output = functions.execute(tool_call)
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": output,
                            })
                        except Exception as e:
                            print(f"Error executing tool_call {tool_call.id}: {e}")

                print(f"Tool outputs: {tool_outputs}")
                if tool_outputs:
                    await ai_client.agents.submit_tool_outputs_to_run(
                        thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                    )

            print(f"Current run status: {run.status}")

        print(f"Run completed with status: {run.status}")

        # Delete the assistant when done
        await ai_client.agents.delete_agent(agent.id)
        print("Deleted assistant")

        # Fetch and log all messages
        messages = await ai_client.agents.list_messages(thread_id=thread.id)
        print(f"Messages: {messages}")


if __name__ == "__main__":
    asyncio.run(main());