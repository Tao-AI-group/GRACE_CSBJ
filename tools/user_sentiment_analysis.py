import sys
import streamlit as st
sys.path.append("..")
from LLM.APILLM import SiliconFlow
from tools.chatbot_answer_question import RAG_based_QA_process, RAG_based_QA_process_streamlit, get_user_question_streamlit, update_script
from backend.RAG_QA import QA_process

# Function to classify a response as yes or no
def classify_yes_no(llm, user_input: str) -> str:
    prompt = f"""
        Classify the user's response as one of the following categories: 'yes', 'no', or 'neutral'.

        User input: "{user_input}"

        Classification:
    """
    response = llm.invoke(prompt)
    if "yes" in response.content.lower():
        return "yes"
    elif "no" in response.content.lower():
        return "no"
    else:
        return "neutral"

# Function to classify a response as yes or no
def classify_yes_or_no_with_user_input(llm) -> str:
    user_input = input("You: ")
    prompt = f"Classify the following response as either 'yes', 'no' or 'neutral': '{user_input}'."
    response = llm.invoke(prompt)
    print("----------------------------------------------")
    print(response)
    if "yes" in response.content.lower():
        return "yes"
    elif "no" in response.content.lower():
        return "no"
    else:
        return "neutral"

def process_user_input(llm, chatbot_question, retriever):
    user_input = input("You: ")
    prompt = f"""
        You asked the user: "{chatbot_question}"  
        They responded: "{user_input}"

        Classify their response based on the following options:
        - "continue" — if the user wants to keep going.  
        - "QA_with_question" — if the user doesn’t want to continue and asks a question.  
        - "QA_without_question" — if the user doesn’t want to continue and does not ask a question.

        Return only the most likely classification. Do not include any explanations.
    """
    response = llm.invoke(prompt)
    print("----------------------------------------------")
    print(response)
    if "continue" in response: # if the user does not have questions, move to the next episode
        print("***************************", "continue, return")
        pass
    elif "QA_with_question" in response: # if the user has questions, move to the RAG-based QA process
        RAG_based_QA_process(llm, chatbot_question, retriever, user_input)
    else: # if the user does not have questions and does not want to move forward, ask user's concern first, then move to the RAG-based QA process
        RAG_based_QA_process(llm, chatbot_question, retriever)

def modify_script_based_on_user_background(llm, script, user_background_info):
    prompt = f"""
        You are a chatbot that educates users about the importance of the HPV vaccine.

        User background: {user_background_info}

        Please tailor the following script to suit the user's background:  
        {script}

        - Keep the final question unchanged.  
        - Use a polite tone.  
        - Return only the revised script with no additional explanations.

        Tailored script:
    """
    return llm.invoke(prompt)

def process_user_input_streamlit(llm, chatbot_question, retriever):
    user_response = ""
    if st.session_state["input_control"] == "process_original_user_response":
        # if user_input := get_user_input("user_question_initial_response"):
        # user_input = st.session_state["chat_value"]
        # st.session_state["chat_value"] = ""
        # st.chat_message("user").write(user_input)
        # st.session_state.messages.append({"role": "user", "content": user_input})

        prompt = f"""
            You are a helpful assistant designed to classify user feedback in a script-based chatbot conversation.

            You will be given:
            1. A script message from the chatbot (e.g., informational content and a follow-up question).
            2. A user response to that message.

            Your task is to classify the user’s response into one of the following three categories:
            - continue: The user acknowledges or agrees with the information and wants to proceed.
            - skip: The user wants to skip this part or feels it's unnecessary.
            - pause: The user asks a question, expresses confusion, disagreement, emotional reaction, or otherwise wants to pause or discuss further before continuing.

            Return ONLY the label: `continue`, `skip`, or `pause`.

            Here is the input:

            Chatbot: {chatbot_question}

            User: {st.session_state["chat_value"]}

            Classification:
        """

        user_response = llm.invoke(prompt).strip().lower()
        # print(prompt)
        print("----------------------------------------------")
        print(st.session_state["chat_value"], " and the response is : ", user_response)
        
        if "continue" in user_response.lower(): # if the user does not have questions, move to the next episode
            # st.session_state["user_last_response"] = ""
            
            st.chat_message("user").write(st.session_state["chat_value"])
            st.session_state.messages.append({"role": "user", "content": st.session_state["chat_value"]})
            update_script()
            return
        elif "skip" in user_response.lower():
            # st.session_state["user_last_response"] = ""
            st.chat_message("user").write(st.session_state["chat_value"])
            st.session_state.messages.append({"role": "user", "content": st.session_state["chat_value"]})
            update_script(skip_next_script=True)
            return
        else:
            # st.session_state["user_last_response"] = user_input
            st.session_state["chatbot_last_response"] = ""
            st.session_state["input_control"] = "QA"
            st.rerun()
    else:
        QA_process(llm=llm, retriever=retriever)
    #     if "QA_with_question" in user_response: # if the user has questions, move to the RAG-based QA process
    #         st.session_state["user_last_response"] = user_input

    # # if the user does not provide a valid question
    # if st.session_state["input_control"] == "QA" and st.session_state["user_last_response"] == "":
    #     get_user_question_streamlit(llm, chatbot_question)
    
    # # if the chatbot does not solve user's question
    # if st.session_state["input_control"] == "QA" and st.session_state["user_last_response"] != "":
    #     RAG_based_QA_process_streamlit(llm, chatbot_question, retriever, user_initial_question=st.session_state["user_last_response"])


if __name__ == "__main__":
    llm = SiliconFlow()
    
    chatbot_question = "Doctors may recommend the HPV vaccine for adults over 26 years old. Without the vaccine, you may be at risk for HPV infection, which can lead to serious diseases like cervical cancer. Shall I continue?"
    process_user_input(llm=llm, chatbot_question=chatbot_question)