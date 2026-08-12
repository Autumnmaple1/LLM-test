from openai import OpenAI
import dotenv
import os
import json
import time


class chat_memory:
    def __init__(self):
        self.memory = []

    def add_message(self, role, content):
        self.memory.append({"role": role, "content": content})

    def clear_memory(self):
        self.memory = []


class LLMClient:
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            timeout=60.0,
        )
        self.model = os.getenv("MODEL")

    def generate_response(self, chat_history):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_history.memory,
            temperature=0,
            max_tokens=204800,
            stream=False,
        )
        return response.choices[0].message.content
            

    def stream_response(self, chat_history):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=chat_history.memory,
            temperature=0,
            max_tokens=204800,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


if __name__ == "__main__":
    llm_client = LLMClient()
    chat_history = chat_memory()
    while True:
        user_input = input("User: ")
        chat_history.add_message("user", user_input)
        print("llm: ", end="")
        for chunk in llm_client.stream_response(chat_history):
            print(chunk, end="")
            response = chunk
        print()
        chat_history.add_message("assistant", response)
