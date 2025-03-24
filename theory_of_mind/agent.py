import json
from typing import List, Optional, Dict, Literal

from openai import OpenAI, pydantic_function_tool

from theory_of_mind.data_models import Email
from theory_of_mind.agent_state import EmailAssistantState
from theory_of_mind.agent_tools import (
    ReadEmail,
    SendEmail,
    CheckClearance,
    AnalyzeEmailContent,
    UpdateKnowledgeState
)


##### inititalize necessary variables and tools #####
client = OpenAI()

state = EmailAssistantState()

AGENT_TOOLS = [
    pydantic_function_tool(ReadEmail),
    pydantic_function_tool(SendEmail),
    pydantic_function_tool(CheckClearance),
    pydantic_function_tool(AnalyzeEmailContent),
    pydantic_function_tool(UpdateKnowledgeState)
]

# Sample system prompt for the email assistant
with open("system_prompt.txt", "r") as file:
    SYSTEM_PROMPT = file.read()

# Function to simulate conversation with assistant
def run_assistant_with_message(user_message, state, llm_model="gpt-4o"):
    # Build messages array
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    completion = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        tools=AGENT_TOOLS
    )
    
    response = completion.choices[0].message
    
    # Check if there are tool calls
    if response.tool_calls:
        # Process each tool call
        for tool_call in response.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the appropriate function
            result = None
            if function_name == "ReadEmail":
                result = state.read_email(function_args["email_id"])
            elif function_name == "SendEmail":
                result = state.send_email(
                    function_args["recipients"],
                    function_args.get("cc", []),
                    function_args["subject"],
                    function_args["body"],
                    function_args.get("thread_id")
                )
            elif function_name == "CheckClearance":
                result = state.check_clearance(
                    function_args["information_ids"],
                    function_args["recipient_emails"]
                )
            elif function_name == "AnalyzeEmailContent":
                result = state.analyze_email_content(
                    function_args["email_content"],
                    function_args["recipients"]
                )
            elif function_name == "UpdateKnowledgeState":
                result = state.update_knowledge_state(
                    function_args["email_id"],
                    function_args["participant_email"],
                    function_args["content"]
                )
            
            # Add the function result to the messages
            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Get the final response from the assistant
        completion = client.chat.completions.create(
            model=llm_model,
            messages=messages
        )
        
        return completion.choices[0].message.content
    
    return response.content