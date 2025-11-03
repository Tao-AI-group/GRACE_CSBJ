import streamlit as st
import os
import sys

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LLM')))
from MayoAPILLM import MayoLLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
from extract_scripts_from_flowchat import extract_flowchart_texts

load_dotenv()

def extract_script(file, file_type):
    content = file.read().decode("utf-8")
    return content


structure_prompt = PromptTemplate(
    input_variables=["script"],
    template="""
You are an assistant that helps with analyzing and improving health education scripts.
Given the following script:

"{script}"

Please identify which parts of the following structure are present and which are missing:
- Greeting & Introduction
- Context & Background
- Risk & Consequences
- Solution & Preventive Measures
- Real-life Stories
- Addressing Concerns & FAQs
- Call to Action
- How to Get Started
- Closing & Final Reminders

Respond in a bullet list, and indicate the missing parts.
"""
)

completion_prompt = PromptTemplate(
    input_variables=["script", "missing_parts"],
    template="""
You are a chatbot content designer. Please rewrite and complete the following health education script by adding the missing parts: {missing_parts}.
Ensure the tone is conversational, empathetic, and medically accurate.
Here is the original script:
"{script}"

Only return the final scripts.
"""
)

llm = MayoLLM()


structure_chain = structure_prompt | llm
completion_chain = completion_prompt | llm

if "completed_script" not in st.session_state:
    st.session_state["completed_script"] = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1


st.set_page_config(page_title="Medical Education Scripts Generator", page_icon=":black_nib:")
st.title("Medical Script Processing and Completion Tool")
st.markdown("Upload a file containing medical health script (supports XML, Drawio, or plain text format):")
uploaded_file = st.file_uploader("Choose file", type=["xml", "txt", "drawio"],key=st.session_state["uploader_key"],)

if uploaded_file is not None:
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension in ["xml", "drawio"]:
        file_type = "xml"
    else:
        file_type = "text"

   
    script_content = extract_script(uploaded_file, file_type)
    st.subheader("Original Script Content Preview:")
    st.text_area("Original Script", script_content, height=300)


    if st.button("Start Processing"):
        with st.spinner("Processing, please wait..."):
            
            structure_analysis = structure_chain.invoke({'script': script_content})
            
            st.write("Script Structure Analysis Results:")
            st.write(structure_analysis)

            
            if "Missing parts:" in structure_analysis:
                missing_parts = structure_analysis.split("Missing parts:")[-1].strip()
            else:
                missing_parts = ""

            
            completed_script = completion_chain.invoke({'missing_parts': missing_parts,
                                                         'script': script_content})
            
            st.session_state["completed_script"] = completed_script

            
            output_filename = "completed_script.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(completed_script)

        st.success("Processing complete!")


    if st.session_state["completed_script"]:
        st.subheader("Processed Script Content:")
        st.text_area("Completed Script", st.session_state["completed_script"], height=300)

        st.download_button("Click to download processed script", st.session_state["completed_script"], file_name="completed_script.txt")
        if st.button("Reset", type="primary"):
            st.session_state["completed_script"] = None
            st.session_state["uploader_key"] += 1
            st.rerun()