import os
import requests
import json
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
import autogen
from fastapi import WebSocket

# Retrieve environment variables
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
researcher_assistant_id = 'asst_3IqSfuUyXt5TyKWCNQnpa7zf'
research_manager_assistant_id = 'asst_EMBek1VyF5V89PDOg1Q35O2M'
director_assistant_id = 'asst_Hn4Gl6H4uEdk64lMgkIzZsnH'

# Configuration for the language model
config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.environ['OPENAI_API_KEY']
    }
]

# ------------------ Create functions ------------------ #

# Function for Google search
def google_search(search_keyword):
    """Performs a Google search and returns the results."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_keyword})
    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text

# Function for summarizing content
def summary(objective, content):
    """Generates a summary for the provided content based on a given objective."""
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = "Write a summary of the following text for {objective}:\n\"{text}\"\nSUMMARY:"
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template, combine_prompt=map_prompt_template, verbose=False)
    return summary_chain.run(input_documents=docs, objective=objective)

# Function for web scraping
def web_scraping(objective: str, url: str):
    """Scrapes a website and summarizes the content based on the provided objective if the content is large."""
    print("Scraping website...")
    headers = {'Cache-Control': 'no-cache', 'Content-Type': 'application/json'}
    data_json = json.dumps({"url": url})
    response = requests.post(f"https://chrome.browserless.io/content?token={brwoserless_api_key}", headers=headers, data=data_json)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)
        return summary(objective, text) if len(text) > 10000 else text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return None

# ------------------ Create agents ------------------ #

# Create AI agents with specific roles and configurations
researcher = GPTAssistantAgent(name="researcher", llm_config={"model": 'gpt-4-1106-preview', "config_list": config_list, "assistant_id": researcher_assistant_id})
research_manager = GPTAssistantAgent(name="research_manager", instructions="be kind", llm_config={"model": 'gpt-4-1106-preview', "config_list": config_list, "assistant_id": research_manager_assistant_id})
director = GPTAssistantAgent(name="director", llm_config={"model": 'gpt-4-1106-preview', "config_list": config_list, "assistant_id": director_assistant_id})

# Custom Group Chat Manager class
class CustomGroupChatManager(autogen.GroupChatManager):
    """Manages group chats with custom behavior, integrating WebSocket communication."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = None

    async def set_websocket(self, websocket: WebSocket):
        """Sets the WebSocket for real-time communication."""
        self.websocket = websocket

    async def a_receive(self, message, sender, request_reply, silent=False):
        """Handles message reception and forwards it to the WebSocket client."""
        await super().a_receive(message, sender, request_reply, silent)
        if self.websocket and type(message) is not str:
            await self.websocket.send_text(f"{sender.name}: {message['content']}")

# Custom User Proxy Agent class
class CustomUserProxyAgent(UserProxyAgent):
    """Extends the UserProxyAgent to integrate WebSocket communication for user input."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = None

    async def set_websocket(self, websocket: WebSocket):
        """Sets the WebSocket for receiving user input."""
        self.websocket = websocket

    async def a_get_human_input(self, prompt):
        """Sends a prompt to the user and waits for the input via WebSocket."""
        await self.websocket.send_text("system: " + prompt)
        user_message = await self.websocket.receive_text()
        return user_message

