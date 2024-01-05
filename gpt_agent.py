#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:52:56 2023

@author: lunar
"""

#%%

import os
from openai import OpenAI
from lunar_tools.utils import read_api_key
import time

api_key = read_api_key('OPEN_AI_KEY') 
client = OpenAI(api_key=api_key)
#%%
# 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
# assistant id: asst_KrxchcAzg9rO4T0If7wwg2sx

class GPTAgent:
    def __init__(self, model = None, name = None, instructions = None, assistant_id = None):
        api_key = read_api_key('OPEN_AI_KEY')
        self.client = OpenAI(api_key=api_key)
        self.current_run_id = None
        self.message_ids = set()
    
        if assistant_id:
            self.init_with_assistant_id(assistant_id)
        else:
            self.assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=[{"type": "code_interpreter"}],
                model=model
            )
        
        self.thread = self.client.beta.threads.create()
        self.messages = []
    
    def init_with_assistant_id(self, assistant_id):
        self.assistant = self.client.beta.assistants.retrieve(assistant_id)


    def send_message(self, content, role="user"):
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=content
        )
        return message

    def execute_run(self, instructions):
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )
        self.current_run_id = run.id
        return run

    def is_run_complete(self):
        if self.current_run_id is None:
            raise ValueError("No run has been initiated.")
        run = self.client.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=self.current_run_id
        )
        return run.status == 'completed'

    def wait_for_run_completion(self, check_interval=1):
        while not self.is_run_complete():
            time.sleep(check_interval)

    def list_messages(self):
        self.messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        ).data
        return self.messages

    def get_all_replies(self):
        self.list_messages()  # Ensure the messages are up-to-date
        replies = []
        for message in self.messages:
            if message.role == 'assistant':
                for content in message.content:
                    if content.type == 'text':
                        replies.append(content.text.value)
        return replies
    
    def get_last_reply(self):
        return self.get_all_replies()[0]




#%%
# Custom model name and instructions - create a new agent
model_name = "gpt-4-1106-preview"
agent_name = "Ocean Painter"
instructions = "change the sentence 'an ocean seen from above' so as to portray the emotion of the prompt. for example, if you receive the message 'I am angry' you should output 'a red ocean seen from above'. only give the output description of the ocean- prompt: "
#%%
# use the assistant ID if you already created it previously
assistant_id = 'asst_KrxchcAzg9rO4T0If7wwg2sx'
gpt = GPTAgent(model=model_name, name = agent_name, instructions=instructions, assistant_id = assistant_id)

#%%
message = gpt.send_message("I feel furious and filled with imense rage")
run_instructions = ""
run = gpt.execute_run(run_instructions)
gpt.wait_for_run_completion()
last_reply = gpt.get_all_replies()[0]






