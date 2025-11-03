import streamlit as st
import os
import sys
from config import BASE_DIR
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.load_metadata import load_scripts, load_user_information, save_user_info
from tools.user_sentiment_analysis import process_user_input_streamlit, modify_script_based_on_user_background
from tools.user_info import User
from LLM.MayoAPILLM import MayoLLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "../db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(
    collection_name="hpv_chatbot",
    embedding_function=embeddings,
    persist_directory=BASE_DIR / "db", 
)

retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
)
llm = MayoLLM()

file_path = BASE_DIR / "data/others/scripts_full.txt"
scripts = load_scripts(file_path)

if 'current_script_index' not in st.session_state:
    st.session_state.current_script_index = 0
if 'script_length' not in st.session_state:
    st.session_state.script_length = len(scripts)
if 'waiting_for_user' not in st.session_state:
    st.session_state.waiting_for_user = False
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "ask_why_user_does_not_want_continue" not in st.session_state:
    st.session_state["ask_why_user_does_not_want_continue"] = False
if "answered_user_question" not in st.session_state:
    st.session_state["answered_user_question"] = False
if "chat_value" not in st.session_state:
    st.session_state["chat_value"] = ""
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None
if "user_last_response" not in st.session_state:
    st.session_state["user_last_response"] = ""
if "chatbot_last_response" not in st.session_state:
    st.session_state["chatbot_last_response"] = ""
if "user_background" not in st.session_state:
    st.session_state["user_background"] = ""
if "modified_script" not in st.session_state:
    st.session_state["modified_script"] = ""
if "input_control" not in st.session_state: 
    st.session_state["input_control"] = ""

st.title("👩🏻‍⚕️ GRACE")
st.caption("💉 By Mayo Clinic")


with st.sidebar:
    st.title("User Information")
    if st.session_state.user_info == None:
        
        with st.form(key="load_id_form"):
            user_id = st.text_input("🔑 Enter User ID to get existing information", key="user_id_key")
            check_id = st.form_submit_button("🔍 Search")

        user_data = None
        if check_id:
            user_data = load_user_information(user_id)
            if user_data:
                st.success(f"Found information for user {user_data.name}")
                st.session_state["user_info"] = user_data
                st.session_state["user_background"] = user_data.generate_background_sentence()
                st.session_state["submitted"] = True
                st.session_state["input_control"] = "script"
                st.session_state["messages"] = []
                st.rerun()
            else:
                st.warning("No user information found for this ID. Please fill in the form below to create a new user.")

        st.markdown("---")  # Divider
    
        
        with st.form(key="user_info_form"):
            user_id = st.text_input("New User ID", value=user_data.user_id if user_data else "")
            name = st.text_input("Name", value=user_data.name if user_data else "")
            gender = st.selectbox("Gender", ["Male", "Female"], index=1 if (user_data and user_data.gender == "Female") else 0)
            age = st.text_input("Age", value=user_data.age if user_data else "")
            degree = st.selectbox("Education", ["Bachelor", "Master", "Ph.D"], index=["Bachelor", "Master", "Ph.D"].index(user_data.degree) if user_data else 0)
            num_children = st.number_input("Number of Children", min_value=0, max_value=10, value=user_data.num_children if user_data else 0)
            submitted = st.form_submit_button("✅ Submit Information")

        
        if submitted:
            new_user = User(name, user_id, gender, age, degree, num_children)
            save_user_info(new_user)
            st.success(f"Information saved for user {name}!")
            st.session_state["user_info"] = new_user
            st.session_state["user_background"] = new_user.generate_background_sentence()
            st.session_state["submitted"] = True
            st.session_state["input_control"] = "script"
            st.session_state["messages"] = []
            st.rerun()

    # Display loaded user information
    else:
        user_info = st.session_state["user_info"]
        st.markdown(f"👤 **User Name** : {user_info.name}")
        st.markdown(f"🎂 **User Age** : {user_info.age}")
        st.markdown(f"🚻 **User Gender** : {user_info.gender}")
        st.markdown(f"🎓 **Education** : {user_info.degree}")
        st.markdown(f"👶 **Number of Children** : {user_info.num_children}")


    # Footer
    "[Designed by Tao's lab](https://github.com/Tao-AI-group)"
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://www.mayoclinic.org/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Mayo_Clinic_logo.svg" alt="Mayo Clinic Logo" style="width: 50%;">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! I am GRACE, HPV chatbot, please enter your user id on the left."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


chat_value = st.chat_input("Your response here...")


if chat_value:
    if st.session_state["user_info"] == None:
        st.info("Please add your ID in the sidebar to continue.")
        st.stop()
    st.session_state["chat_value"] = chat_value
    
if st.session_state["input_control"] == "script":
    print("---------------------New scirpt-------------------------")
    print(st.session_state.current_script_index)
    current_index = st.session_state.current_script_index
    user_info = st.session_state["user_info"]

    script = scripts[current_index]
    if "user_name" in script:
        script = script.replace("{user_name}", user_info.name)
    if "dr_name" in script:
        script = script.replace("{dr_name}", "Charlotte")

    
    modified_script = modify_script_based_on_user_background(llm, script, st.session_state["user_background"])
    st.session_state["modified_script"] = modified_script
    st.chat_message("assistant").write(modified_script)
    st.session_state.messages.append({"role": "assistant", "content": modified_script})
    st.session_state["input_control"] = "process_original_user_response"
    if current_index == len(scripts) - 1:
        st.info("The HPV Vaccine Chatbot process is finedsed. Thanks for using!😄")

if st.session_state["input_control"] == "process_original_user_response" or st.session_state["input_control"] == "QA":
    if st.session_state["chat_value"] != "":
        process_user_input_streamlit(llm, st.session_state["modified_script"], retriever)