import sys
sys.path.append("..")
from LLM.APILLM import SiliconFlow
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
# Read the text content from the file
bg = '''
Name: Anna  
Age: 81  
Sex: Female
Marital Status: Single  
Children: No  
Education: PhD 
'''
def analyze_tone_to_user(user_background: str, llm) -> str:
    llm = SiliconFlow()
    prompt = f'''
    This is a user's background information and your are a dorctor to introduce the importance of HPV vaccine. Based on the user's background, what tone should you use to talk with the user?. 
    Only return one sentence to summraize tone, no need for explaination.
    User information:{user_background}
    '''
    response = llm.invoke(prompt)
    return response
    
