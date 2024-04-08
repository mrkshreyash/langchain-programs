import gradio as gr
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-3.5-turbo"

# Use ChatOpenAI for initializing the model
llm = OpenAI(model_name=model, openai_api_key=openai_api_key)

# Updated the input_variables key
prompt = PromptTemplate(
    input_variables=["DishName"],
    template="Write a step-by-step procedure to perform {DishName} recipe.",
)

inputs = gr.Textbox(label="Dish Name")
output = gr.Textbox(label="Recipe")


def formatted_text(dish_name: str) -> str:
    fmt_text = prompt.format(DishName=dish_name)
    response = llm(fmt_text)
    return response


app = gr.Interface(fn=formatted_text, inputs=inputs, outputs=output)
app.launch(server_name="localhost", server_port=1234)
