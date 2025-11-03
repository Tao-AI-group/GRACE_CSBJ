import os
import sys

from tools.data_process import generate_local_knowledge_base
sys.path.append("..")
from dotenv import load_dotenv
from LLM.APILLM import SiliconFlow
from LLM.APIEmbeddings import APIEmbedding
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from tools import user_sentiment_analysis
from tools.load_metadata import load_scripts, load_user_information
from tools.user_info import User
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import random

# 1 - Hyperparameters
ASK_TIMEOUT = 10
DOCTOR_NAME = "Charlotte"
EXIT_MESSAGE = "Thank you for using this program, bye"
store = {}
config = {"configurable": {"session_id": "abc123"}}

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "../db")
persistent_directory = os.path.join(db_dir, "chroma_hpv_metadata")

# Define the embedding model
embeddings = APIEmbedding()

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
)
llm = SiliconFlow()

# ----------------------------------------------------------------

# 2 - Prompts

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

RAG_prompt = (
    """
    You are a medical assistant explaining HPV and its vaccine to the user. 
    The user refused for your suggestion : {suggestion}.
    Use the retrieved context to answer their question accurately and concisely. 
    If unsure, acknowledge the uncertainty. Limit your response to three sentences, and maintain a {tone} tone throughout.
    retrieved context: {context}.
    """
)

RAG_initial_prompt = (
    """
    You are a medical assistant explaining HPV and its vaccine to the user. 
    The user refused for your suggestion : {suggestion}.
    Nail the user to explain why it doesn't want to follow your advice.
    """
)


details_of_HPV_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user declines to learn about the HPV vaccine, the assistant should gently persuade them by highlighting the vaccine's importance, benefits, and addressing any concerns. The tone should remain respectful and supportive. 

            Assistant: Nice to meet you I hope your day is going well. Allow me to introduce myself. I am a robot assistant for a Doctor. I want to let you know important details about the HPV vaccine. Are you interested in learning them? I promise our conversation will be confidential.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

at_risks_of_HPV_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user declines to learn about it is at risk for HPV infection, the assistant should gently persuade them by highlighting the vaccine's importance, benefits, and addressing any concerns. The tone should remain respectful and supportive. 

            Assistant: For adults over 26 years old doctors may recommend you get the HPV vaccine. If you don’t get the HPV vaccine, you are at risk for HPV infection which could later turn into life-threatening diseases (K10, and K12). Shall I continue?""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

do_not_understand_need_of_HPV_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user declines to learn about why it should get HPV vaccine, the assistant should gently persuade them by highlighting the vaccine's importance, benefits, and addressing any concerns. The tone should remain respectful and supportive. 

            Assistant:  If you did not get the HPV vaccine when you were an adolescent, doctors commend you to get it by age 26 [K3]. Do you understand why you need to get the vaccine?""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

doctor_recommend_HPV_vaccine_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user refuses the recommendation from a doctor to get HPV vaccine, the assistant should gently persuade them by highlighting the vaccine's importance, benefits, and addressing any concerns. The tone should remain respectful and supportive. 

            Assistant:  For adults over 26 years old, doctors may recommend you get the HPV vaccine. [K4] Should I continue?""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

story_teller_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user refuses to hear a story about a HPV-cancer survivor, the assistant should gently persuade them by highlighting the importance of this story to know about the Understand the dangers of HPV cancer. The tone should remain respectful and supportive. 

            Assistant:  Currently over 14 million Americans are infected by HPV in 2021. Sexually active people are at a higher risk of becoming infected by the HPV virus. HPV is very common and there is a high chance of you getting HPV infection without HPV vaccines. Failing to get the HPV vaccine puts you a trisk of becoming infected. But, it is not too late to get the HPV vaccine (PSU1, PSU2, PSU3, PSU4). Is it okay if I illustrate a real story from a HPV-cancer survivor?""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

seriouseness_of_HPV_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user does not understand the seriousness of HPV infection, the assistant should gently persuade them by highlighting the importance of this story to know about the Understand the dangers of HPV cancer. The tone should remain respectful and supportive. 

            Assistant: Here is Christopher's story. 
            I was getting ready to go out to dinner and I was shaving my neck when felt a bump. I had a bad cold so I thought it was a swollen gland. But it didn’t go away so I had it checked out by an ENT doctor in Los Angeles where I live.
            The diagnosis confused me. I never smoked or chewed tobacco and did not drink heavily. But I had a risk factor that 80% of adults also have – exposure to HPV.
            Like Christopher's story, almost all the people are in risk of HPV infection. Do you understand the seriousness of HPV infection?""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

side_effects_persuade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is text from a medical assistant to a user. The user refuses to learn about the side effects of the HPV cancer, the assistant should gently persuade them by highlighting the importance of this story to know about the Understand the dangers of HPV cancer. The tone should remain respectful and supportive. 

            Assistant:  High-risk HPV strains cause about 70% of cervical cancers. Women who need treatment to remove HPV-related cervical cancer or precancerous cells can become infertile. Other high-risk HPV strains can cause throat cancer and genital warts in men and women. Don’t risk your health by not getting vaccinated (PSE1, PSE2). Do you get the message? - Should I mention about side effects?""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)



# ----------------------------------------------------------------

# 3 - Functions
# Function to ask the user's name
'''
def ask_name():
    attempts = 0
    while attempts < 3:
        try:
            name = inputimeout(prompt=f"Greetings. My name is Beverly. May I ask your name? (You have {ASK_TIMEOUT} seconds to respond): ", timeout=ASK_TIMEOUT)
            if name:
                return name
        except TimeoutOccurred:
            print("No response detected.")
        
        attempts += 1

    return ""  # Return an empty string if no valid name is provided
'''


# Function to initialize the database
# generate_local_knowledge_base.initialize_database(recreate=False)

# Get history based on session id
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def state_machine_workflow(scripts, user_info:User):
    for script in scripts:
        if "user_name" in script:
            script = script.replace("{user_name}", user_info.name)
        if "dr_name" in script:
            script = script.replace("{dr_name}", "Charlotte")
        print(script)
        user_sentiment_analysis.process_user_input(llm, script, retriever)
    
def main():
    file_path = "../data/others/scripts.txt"
    scripts = load_scripts(file_path)
    user_info = load_user_information()
    state_machine_workflow(scripts, user_info)

# --------------------------------------------------------------
# program entrance
main()