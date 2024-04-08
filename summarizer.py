import os

import gradio as gr
import wget
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX02OLEN/state_of_the_union.txt"

output_path = "state_of_the_union.txt"

if not os.path.exists(output_path):
    wget.download(url, out=output_path)

loader = TextLoader(output_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")

data = loader.load()

index = VectorstoreIndexCreator().from_loaders([loader])


def summarize(query):
    return index.query(query)


gr.Interface(fn=summarize, inputs="text", outputs="text").launch()
