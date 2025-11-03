import os
import streamlit as st
import sys
sys.path.append("..")
from LLM.APILLM import SiliconFlow
from LLM.APIEmbeddings import APIEmbedding
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

def chat_message(role, message):
    st.chat_message(role).write(message)

def update_script(skip_next_script=False):
    st.session_state.waiting_for_user = False
    st.session_state["user_last_response"] = ""
    st.session_state["chatbot_last_response"] = ""
    if skip_next_script and st.session_state.current_script_index + 2 < st.session_state.script_length:
            st.session_state.current_script_index += 2
    else:   
        st.session_state.current_script_index += 1
    st.session_state["input_control"] = "script"
    st.session_state["chat_value"] = ""
    st.session_state["ask_why_user_does_not_want_continue"] = False
    st.session_state["answered_user_question"] = False
    print("---------------------Current scirpt ended-------------------------")
    print(st.session_state.current_script_index)
    st.rerun()

ask_why_user_does_not_want_continue_prompt = """
    The user refused to continue after being asked: "{}".  
    However, they did not explain why.  

    Generate a polite sentence asking for their reason.  
    Respond only with the sentence.
"""

contain_questions_prompt = """
    Does the following user response contain a question?  

    User response: "{}"  

    Respond with "yes" or "no".
"""

does_user_want_to_ask_more_questions = """
    You asked the user: "{}"  
    They responded: "{}"  

    Based on their response, does the user want to ask more questions?  
    Respond only with "yes" or "no".
"""
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


# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant designed to help with question-answering tasks. "
    "Use the following retrieved context to respond accurately. "
    "If you're unsure of the answer, respond with 'I don't know.' "
    "Keep your answer concise and limited to a maximum of three sentences. "
    "After your answer, include a natural-sounding follow-up question to check if the user has more questions."
    "\n\n"
    "{context}"
)


# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chat_history = []  # Collect chat history here (a sequence of messages)


def RAG_based_QA_process(llm, chatbot_question, retriever, user_initial_question=""):
    if user_initial_question == "":
        while True:
            response = llm.invoke(ask_why_user_does_not_want_continue_prompt.format(chatbot_question))
            print(response)
            user_initial_question = input("You :")
            user_response_situation = llm.invoke(contain_questions_prompt.format(user_initial_question))
            print("user_response_situation:", user_response_situation)
            if "yes" in user_response_situation.lower():
                break
    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create a chain to combine documents for question answering
    # create_stuff_documents_chain feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    print("I understand that you might have concerns or doubts. I will try to address your them.")
    query = user_initial_question
    chat_history.append(SystemMessage(content=chatbot_question))
    chat_history.append(HumanMessage(content=user_initial_question))
    
    while True:
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
        query = input("You: (type exit to end your QA process)")
        if query.lower() == "exit":
            break

def get_user_question_streamlit(llm, chatbot_question):
    if not st.session_state["ask_why_user_does_not_want_continue"]:
        response = llm.invoke(ask_why_user_does_not_want_continue_prompt.format(chatbot_question))
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state["ask_why_user_does_not_want_continue"] = True
    if st.session_state["chat_value"] != "":
        user_question = st.session_state["chat_value"]
        st.session_state["chat_value"] = ""
        st.chat_message("user").write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        user_response_situation = llm.invoke(contain_questions_prompt.format(user_question))
        if "yes" in user_response_situation.lower():
            st.session_state["user_last_response"] = user_question
        else:
            st.session_state["user_last_response"] = "" 
            # user does not provide a valid question, ask again
            st.session_state["ask_why_user_does_not_want_continue"] = False

def RAG_based_QA_process_streamlit(llm, chatbot_question, retriever, user_initial_question=""):
    # if user_initial_question == "":
    #     while True:
    #         response = llm.invoke(ask_why_user_does_not_want_continue_prompt.format(chatbot_question))
    #         st.chat_message("assistant").write(response)
    #         st.session_state.messages.append({"role": "assistant", "content": response})
    #         if user_initial_question := get_user_input("initial_question"):
    #             st.chat_message("user").write(user_initial_question)
    #             st.session_state.messages.append({"role": "user", "content": user_initial_question})
    #             user_response_situation = llm.invoke(contain_questions_prompt.format(user_initial_question))
    #             # print("user_response_situation:", user_response_situation)
    #             if "yes" in user_response_situation.lower():
    #                 st.session_state["user_last_response"] = user_initial_question
    #                 break
    #             else:
    #                 st.session_state["user_last_response"] = ""
    
    if not st.session_state["answered_user_question"]:
        # update the context first
        chat_history.append(HumanMessage(content=user_initial_question))
        chat_history.append(SystemMessage(content=chatbot_question))
        # Create a history-aware retriever
        # This uses the LLM to help reformulate the question based on chat history
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Create a chain to combine documents for question answering
        # `create_stuff_documents_chain` feeds all retrieved context into the LLM
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create a retrieval chain that combines the history-aware retriever and the question answering chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        query = user_initial_question
        
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        # Display the AI's response
        st.chat_message("assistant").write(f"{result['answer']}")
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
        st.session_state["answered_user_question"] = True
        st.session_state["chat_value"] = ""
    if st.session_state["chat_value"] != "":
        query = st.session_state["chat_value"]
        st.chat_message("user").write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        does_user_have_more_questions = llm.invoke(contain_questions_prompt.format(query))
        print("--------------------QA Process--------------------------")
        print("Does user has more question: ", does_user_have_more_questions)
        if "no" in does_user_have_more_questions.lower():
            update_script()
        else:
            print("---------------------new question!-------------------------")
            print("follow up question : ", query)
            st.session_state["user_last_response"] = query
            st.session_state["answered_user_question"] = False
            st.rerun()
            

if __name__ == "__main__":
    llm = SiliconFlow()
    chatbot_question = "Doctors may recommend the HPV vaccine for adults over 26 years old. Without the vaccine, you may be at risk for HPV infection, which can lead to serious diseases like cervical cancer. Shall I continue?"
    # Define the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "../db")
    persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")
    # Define the embedding model
    embeddings = APIEmbedding()

    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Create a retriever for querying the vector store
    # `search_type` specifies the type of search (e.g., similarity)
    # `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    RAG_based_QA_process(llm=llm, chatbot_question=chatbot_question,retriever=retriever, user_initial_question="")