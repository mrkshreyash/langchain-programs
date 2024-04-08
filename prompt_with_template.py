import os

import gradio as gr
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

openai_api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

prompt = PromptTemplate(
    input_variables=["position", "company", "skills"],
    template="Dear Hiring Manger,\n\nI am writing to apply for the {position} position at {company}. I have experience in {skills}.\n\nThank you for considering my application.\n\nSincerely,\n[Your Name]",
)


def generate_cover_letter(position: str, company: str, skills: str) -> str:
    formatted_prompt = prompt.format(position=position, company=company, skills=skills)
    response = llm(formatted_prompt)
    return response


inputs = [
    gr.Textbox(label="Position"),
    gr.Textbox(label="Company"),
    gr.Textbox(label="Skills")
]

output = gr.Textbox(label="Cover Letter")

app = gr.Interface(fn=generate_cover_letter, inputs=inputs, outputs=output)

app.launch(server_name="localhost", server_port=1234)
