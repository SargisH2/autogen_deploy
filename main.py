# Import necessary libraries
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import autogen

# Import custom modules for handling chat and research functionality
from assistants_chat import (
    researcher, research_manager, director,
    CustomGroupChatManager, CustomUserProxyAgent,
    web_scraping, google_search
)

# Set up the configuration for the model
config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.environ['OPENAI_API_KEY']
    }
]

# Register functions for web scraping and Google search
researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search
    }
)

# Initialize a proxy agent for user interactions
user_proxy = CustomUserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS"
)

# Create a group chat environment
groupchat = autogen.GroupChat(
    agents=[director, user_proxy, researcher, research_manager],
    messages=[],
    max_round=10 ### 10 by default
)

# Set up a custom group chat manager
group_chat_manager = CustomGroupChatManager(
    human_input_mode="ALWAYS",
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# Initialize the FastAPI app
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates", auto_reload=True)

# Initialize a list to store messages
messages: List[str] = []

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    # Render the chat interface using a template
    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive a message from the client
        user_message = await websocket.receive_text()

        # Set up the websocket for the group chat manager and user proxy
        await group_chat_manager.set_websocket(websocket)
        await user_proxy.set_websocket(websocket)

        # Initiate a chat session using the user's message
        await user_proxy.a_initiate_chat(
            group_chat_manager,
            message=user_message,
            clear_history=True
        )
