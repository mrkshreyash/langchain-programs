import os

import gradio
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Setting up the LLM
openai_api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(model_name="gpt-3.5-turbo",temperature=0.9)


def handle_complaint(complaint: str) -> str:
    prompt = PromptTemplate(input_variables=["complaint"],
                            template="I am customer service representative. I received the following complaint: {complaint}. My response is:")

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(complaint)

iface = gradio.Interface(fn=handle_complaint, inputs="text", outputs="text")
iface.launch()
