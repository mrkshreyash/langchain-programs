import os

import gradio as gr
from langchain.llms import OpenAI

openai_api_key = os.environ.get("OPENAI_API_KEY")

gpt3 = OpenAI(model_name="gpt-3.5-turbo-instruct")


def chatbot(inp):
    return gpt3(inp)


demo = gr.Interface(fn=chatbot, inputs="text", outputs="text")

demo.launch(server_name="localhost", server_port=1234)
