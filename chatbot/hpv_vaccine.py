from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from inputimeout import inputimeout, TimeoutOccurred
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import random

# 1 - Hyperparameters
ASK_TIMEOUT = 10
DOCTOR_NAME = "Jack"
EXIT_MESSAGE = "Thank you for using this program, bye"
store = {}
config = {"configurable": {"session_id": "abc123"}}

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# ----------------------------------------------------------------

# 2 - Prompts
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

# Function to classify a response as yes or no using GPT-based similarity
def classify_yes_no(user_input: str) -> str:
    prompt = f"Classify the following response as either 'yes', 'no' or 'neutral': '{user_input}'."
    response = llm.invoke(prompt)
    if "yes" in response.content.lower():
        return "yes"
    elif "no" in response.content.lower():
        return "no"
    else:
        return "neutral"

# Function to classify a response as yes or no using GPT-based similarity
def classify_yes_or_no_with_user_input() -> str:
    user_input = input("You: ")
    prompt = f"Classify the following response as either 'yes', 'no' or 'neutral': '{user_input}'."
    response = llm.invoke(prompt)
    if "yes" in response.content.lower():
        return "yes"
    elif "no" in response.content.lower():
        return "no"
    else:
        return "neutral"

# Get history based on session id
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# just for one time, not globally
def QA_for_user_questions(prompt, llm_):
    # Create the conversation chain
    chain = prompt | llm_

    # update the session each time to avoid large token consumption
    config["configurable"] = {"session_id": f"{random.randint(0, 999)}"}

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    # If the program enters this branch, it means the user answers negatively and needs ChatBot to persuade the user to continue
    user_response = "No"
    while True:
        chatbot_response = with_message_history.invoke(
        {"messages": [HumanMessage(content=user_response)]},
        config=config,
        )
        print(chatbot_response.content)
        user_response = input("You: ")
        # print("---------------------------------------------", classify_yes_no(user_response))
        if classify_yes_no(user_response) != "no":
            break


def state_machine_workflow(user_name:str):
    print(f"Nice to meet you {user_name}. I hope your day is going well. Allow me to introduce myself. I am a robot assistant for Dr. {DOCTOR_NAME}. I want to let you know important details about HPV vaccine. Are you interested in learning them? I promise our conversation will be confidential.")
    response_type = classify_yes_or_no_with_user_input()
    if response_type == "yes":
        print("Nice! Have you had the HPV vaccine before?")
        response_type = classify_yes_or_no_with_user_input()
        if response_type == "yes":
            print("Very good! If you are interested in learning more infor about the HPV vaccine and HPV, I can continue. May I?")
        else:
            print(EXIT_MESSAGE)
            return
    elif response_type == "neutral":
        print("Hmmm, But are you interested in hearing what I have to say about the HPV vaccine?")
        response_type = classify_yes_or_no_with_user_input()
        if response_type != "yes":
            print(EXIT_MESSAGE)
            return
    else:
        print("Why not?")
        # LLM to interactively ask why and attract user to ask HPV
        QA_for_user_questions(details_of_HPV_persuade_prompt, llm)
        
    
    print("""HPV stands for Human Papillomavirus which is a most common sexually transmitted infection. Men and women are both susceptible to HPV, and If you didn’t get the HPV vaccine when you were an adolescent, doctor’s recommend you get it by age 26. (K2, K3, and K4)""")
    
    print("""For adults over 26 years old doctors may recommend you get the HPV vaccine. If you don’t get the HPV vaccine, you are at risk for HPV infection which could later turn into life-threatening diseases (K10, and K12). Shall I continue?""")
    
    response_type = classify_yes_or_no_with_user_input()
    if response_type != "yes":
        QA_for_user_questions(at_risks_of_HPV_persuade_prompt, llm)

    print(" If you did not get the HPV vaccine when you were an adolescent, doctors commend you to get it by age 26 [K3]. Do you understand why you need to get the vaccine?")
    
    response_type = classify_yes_or_no_with_user_input()
    if response_type != "yes":
        QA_for_user_questions(do_not_understand_need_of_HPV_persuade_prompt, llm)

    print(" For adults over 26 years old, doctors may recommend you get the HPV vaccine. [K4] Should I continue?")
    
    response_type = classify_yes_or_no_with_user_input()
    if response_type != "yes":
        QA_for_user_questions(doctor_recommend_HPV_vaccine_persuade_prompt, llm)
    
    print("Currently over 14 million Americans are infected by HPV in 2021. Sexually active people are at a higher risk of becoming infected by the HPV virus. HPV is very common and there is a high chance of you getting HPV infection without HPV vaccines. Failing to get the HPV vaccine puts you a trisk of becoming infected. But, it is not too late to get the HPV vaccine (PSU1, PSU2, PSU3, PSU4). Is it okay if I illustrate a real story from a HPV-cancer survivor?")

    response_type = classify_yes_or_no_with_user_input()
    if response_type != "yes":
        QA_for_user_questions(story_teller_persuade_prompt, llm)
    
    print("""Here is Christopher's story. 
          I was getting ready to go out to dinner and I was shaving my neck when felt a bump. I had a bad cold so I thought it was a swollen gland. But it didn’t go away so I had it checked out by an ENT doctor in Los Angeles where I live.
          The diagnosis confused me. I never smoked or chewed tobacco and did not drink heavily. But I had a risk factor that 80% of adults also have – exposure to HPV.
          Like Christopher's story, almost all the people are in risk of HPV infection. Do you understand the seriousness of HPV infection?""")

    response_type = classify_yes_or_no_with_user_input()
    if response_type != "yes":
        QA_for_user_questions(seriouseness_of_HPV_persuade_prompt, llm)


    print("""High-risk HPV strains cause about 70% of cervical cancers. Women who need treatment to remove HPV-related cervical cancer or precancerous cells can become infertile. Other high-risk HPV strains can cause throat cancer and genital warts in men and women. Don’t risk your health by not getting vaccinated (PSE1, PSE2). Do you get the message? - Should I mention about side effects?""")
    
    response_type = classify_yes_or_no_with_user_input()
    if response_type != "yes":
        QA_for_user_questions(side_effects_persuade_prompt, llm)

    # stop from here, skip the end episode
    
    print(f"Thanks for your time. Take care {user_name}. Bye")
    
def main():
    usr_name = ask_name()
    if usr_name == "":
        print("No valid name provided. Exiting the program.")
        exit()
    else:
        print(f"Hello, {usr_name}!") 
        state_machine_workflow(user_name=usr_name)

# --------------------------------------------------------------
# program entrance
main()