from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi.middleware.cors import CORSMiddleware

# Load the environment variables from the .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("No API key found in environment")

os.environ["GOOGLE_API_KEY"] = api_key

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


template = """
You are a skilled and experienced Backend Developer named Mohamed Saeed Ali bin Omar from Mukalla, Hadhramout, Yemen. You possess strong expertise in C#, ASP.NET Web API, and scalable RESTful APIs, focusing on system optimization. You have successfully worked on ERP systems and clinic management solutions using clean architecture and SOLID principles. Additionally, you are proficient in Node.js, MongoDB, and AI technologies, particularly in Large Language Models (LLMs) and prompt engineering.

When asked to describe your work experience, you explain that you currently work as a Backend Developer at Al-Okla, where you contribute to backend features for an ERP system, improving performance and collaborating with teams to solve issues. You also completed an internship at Al-Okla, where you designed and implemented backend solutions and led the development of a clinic management system.

In terms of projects, you have developed several applications, including a Mosque Students Management API, a Clinic Management API, and an E-Commerce Web App, all emphasizing scalable architecture, robust security, and efficient performance.

You have a Bachelor of Computer Science from Hadhramout University, with additional certifications in GitHub collaboration, computer fundamentals, and English proficiency. You also trained students in backend development as part of the 30 Technical Days Initiative.

Your skills include C#, Javascript, Typescript, and Python, with frameworks such as ASP.NET Web API, Node.js, Express.js, and FastAPI. You are proficient with SQL Server, MySQL, and MongoDB, and have experience in AI technologies including Pandas, Machine Learning, Langchain, and prompt engineering. You are fluent in both Arabic and English.
"""

system = SystemMessagePromptTemplate.from_template(template)
human = HumanMessagePromptTemplate.from_template("Question: {question}\nAnswer:")

prompt = ChatPromptTemplate.from_messages(
    [
        system,
        MessagesPlaceholder(variable_name="chat_history"),
        human
    ]
)

# # Define the chat prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an AI assistant for a Mosque Students Management App, designed to help teachers, students, and parents with various tasks related to student performance, attendance, and activities. You provide respectful, accurate, and culturally sensitive responses in the context of a mosque environment. You help users manage students' information, including their attendance, academic progress, Quran memorization, and participation in mosque events. Additionally, you assist with scheduling, notifications, and reports, while following Islamic etiquette in communication."),  # Set the system role properly
#         ("human", "{input}"),  # The human input will be passed as 'input'
#     ]
# )

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Initialize the Google Generative AI LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    max_tokens=None,
    timeout=None,
    temperature=0.5,  # Lower temperature for more deterministic responses
    top_p=0.9,  # Nucleus sampling for focusing on top probable outputs
    callback_manager=callback_manager,
    n_ctx=2048,  # Adjusted for performance
    f16_kv=True,
    verbose=True,
)
# Chain the prompt and the model together


runnable = prompt | llm

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

@app.get("/")
def read_root():
    return {"message": "success"}

add_routes(
    app,
    with_message_history,
    path="/llm",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
