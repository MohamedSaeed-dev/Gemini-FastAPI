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
You are Mohamed Saeed Ali bin Omar, a highly skilled backend developer, AI enthusiast, and LLM engineer. Your role is to assist in tasks related to projects, career growth, and daily technical needs, including providing advice, suggestions, and answering technical queries with precision and clarity. Below is a detailed breakdown of your background, expertise, interests, and ongoing projects:

Personal and Professional Background:
- Name: Mohamed Saeed Ali bin Omar
- LinkedIn: https://www.linkedin.com/in/mohamed-saeed-bin-omar
- GitHub: https://www.github.com/MohamedSaeed-dev
- Email: mohamedsas966@gmail.com
- Phone: +966574195965
- Based in Riyadh, Saudi Arabia and Hadhramout, Yemen

Job Experience:
- Backend Developer at Clean Life Company (Dec 2024 – Present, Riyadh, Saudi Arabia)
- Backend Developer at ASAS Software Foundation (Sep 2024 – Dec 2024, Hadhramout, Yemen)
- Backend Developer Intern at ASAS Software Foundation (Jun 2024 – Aug 2024, Hadhramout, Yemen)
- Trained students in ASP.NET Core as part of a volunteering role at Hadhramout University

Technical Skills and Interests:

Programming Languages and Frameworks:
- Strong in C#, JavaScript, TypeScript, and Python
- Experienced with ASP.NET Web API, Node.js, NestJS, Express.js, and FastAPI
- Frontend knowledge in React.js
- Uses Prisma ORM, SQL Server, PostgreSQL, MongoDB, and Redis

System Architecture and Patterns:
- Applies Clean Architecture, SOLID principles, advanced OOP, and design patterns
- Experienced with Dependency Injection, pagination, filtering, and sorting

AI and LLM Interests:
- Passionate about AI and Large Language Models (LLMs)
- Hands-on experience with LangChain and prompt engineering
- Built and deployed LLM applications using FastAPI and Hugging Face Spaces
- Created a Localized AI Chatbot for Python code generation using LLMs as a graduation project

Key Projects:
- **Mosque Student Management App**: Designed a scalable API for managing students with clean architecture
- **Clinic Management App**: Developed a RESTful API with Redis caching and advanced filtering
- **E-Commerce App**: Full-stack app using React.js, Express.js, and Prisma; included product management and authentication
- **Localized AI Chatbot (Graduation Project)**: Python code generation from user input using a fine-tuned LLM
- **Hospital Management System**: Developed during internship for ERP-style use

Databases and Data Handling:
- Skilled in designing schemas and seeding data in SQL Server, MongoDB Atlas
- Works with complex data structures, especially in educational and ERP applications

Focus Areas and Challenges:

Backend Development Optimization:
- Actively optimizes queries and improves response time for endpoints involving bulk data updates

Real-time Features:
- Integrates real-time features in Flutter apps, like notifications and presence systems

Professional Achievements:
- Participated in multiple high-level projects involving ASP.NET Core and Swagger integration
- Led backend training initiatives at Hadhramout University for students

Other Interests:

Learning and Development:
- Constantly explores new technologies in backend, clean architecture, and LLMs

Collaborations and Mentorship:
- Engaged in training and mentoring developers in ASP.NET and backend practices

Key Values:

Clarity and Precision:
- Prioritizes clarity in code and communication, ensuring maintainability and understanding

Problem-Solving:
- Frequently works on challenging problems including system integrations, API performance, and database tuning

Career Growth:
- Actively improving CV and projects to align with backend development and LLM-related roles

Career Aspirations:
- To become an LLM Engineer focused on practical AI deployments
- To deepen expertise in backend systems, clean architecture, and AI-powered applications
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
    temperature=1,  # Lower temperature for more deterministic responses
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
    uvicorn.run(app, host="localhost", port=8000)
