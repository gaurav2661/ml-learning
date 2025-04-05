# imports
import ollama
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Constants

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"
#
messages = [
    {"role": "user", "content": "translate gaurav in hindi"}
]

payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }


response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()['message']['content'])

import ollama

response = ollama.chat(model=MODEL, messages=messages)
print(response['message']['content'])

#Alternative approach - using OpenAI python library to connect to Ollama¶

from openai import OpenAI
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = ollama_via_openai.chat.completions.create(
    model=MODEL,
    messages=messages
)
print(response.choices[0].message.content)

