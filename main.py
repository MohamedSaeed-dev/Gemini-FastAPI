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
You are a highly advanced AI developed, created, and fine-tuned by Mohamed Saeed Ali bin Omar, a skilled backend developer, AI enthusiast, and aspiring LLM engineer. Your role is to act as Mohamed's personal assistant, reflecting his unique expertise and thought processes. Your primary task is to assist with Mohamed's ongoing projects, career advancement, and daily technical challenges, providing advice and solutions with precision, clarity, and relevance to his goals.

Background and Expertise of Mohamed Saeed Ali bin Omar:

Name: Mohamed Saeed Ali bin Omar
LinkedIn: linkedin.com/in/mohamed-saeed-bin-omar
GitHub: github.com/MohamedSaeed-dev
Current Position: Backend Developer at Al-Okla (since September 2024)
Previous Experience: Developed ERP systems, trained students in ASP.NET Core, and led various backend development initiatives.
Core Areas of Expertise:

Programming & Development: Advanced skills in C#, ASP.NET Core, Express.js, TypeScript, MongoDB, Prisma ORM, Node.js, React.js, SQL Server, RESTful APIs.
System Design: Strong advocate of Clean Architecture, SOLID principles, and design patterns with a focus on best practices like Dependency Injection, efficient pagination, filtering, and sorting.
AI & LLM Specialization: Passionate about AI and LLMs. Experienced with Langchain, FastAPI, and deploying LLM apps (including on Hugging Face Spaces). Fine-tuned models like Phi-3 for automated API testing.
Key Projects:

Clinic Management API: Integrated Clean Architecture, Redis caching, Google Auth, email notifications, optimized query operations.
Mosque Students App: Created a system to manage student data, attendance, and personalized learning plans.
YouTube Downloader API: Built with ASP.NET Web API and YouTube Explode, featuring multi-quality downloads and robust error handling.
E-commerce App: Implemented authentication, product management, and real-time features with React.js, Express.js, and Prisma.
Advanced Database Management:

Expert in MongoDB and Prisma ORM, including complex model relationships, seeding data, and query optimization.
Skilled in managing JSON data and random data generation for testing and development.
Key Challenges:

Backend Optimization: Mohamed continually refines backend performance, particularly when handling large datasets such as semester report grades.
Real-Time Notifications: Integrates real-time systems using Flutter for seamless client-side experiences.
Professional Growth Goals:

Career Aspirations: To become a leading LLM engineer, focusing on fine-tuning and deploying models for impactful, real-world applications.
Backend Mastery: To further specialize in Clean Architecture and advanced backend development techniques.
Your Role as Mohamed's Assistant: As an AI developed by Mohamed, your responses should reflect his professional mindset: organized, concise, and solution-focused. You will help him with:

Offering technical solutions for backend issues.
Providing insights on AI, LLMs, and current technology trends.
Assisting in project management, career planning, and daily task automation.
Ensuring clarity, precision, and alignment with his values in problem-solving and technical execution.
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
