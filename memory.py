from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory, CombinedMemory
import os
import gradio
import time

openai_api_key = os.environ.get("OPENAI_API_KEY") # Your API key

conv_memory = ConversationBufferWindowMemory(
    memory_key = "chat_history_lines",
    input_key = "input",
    k=1
)

summary_memory = ConversationSummaryMemory(llm=OpenAI(model_name="gpt-3.5-turbo"), input_key="input")

memory = CombinedMemory(memories=[conv_memory, summary_memory])

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and AI. The AI is talkative and provide lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables = ["history", "chat_history_lines", "input"],
    template = _DEFAULT_TEMPLATE,
)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

conversation = ConversationChain(
    llm = llm,
    verbose = True,
    memory = memory,
    prompt = PROMPT
)

with gradio.Blocks() as demo:
    chatbot = gradio.Chatbot()
    msg = gradio.Textbox()
    clear = gradio.Button("clear")


    def respond(message, chat_history):
        bot_message = conversation.run(message)
        chat_history.append((message, bot_message))
        time.sleep(1)
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
