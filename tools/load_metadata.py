from typing import List
from tools.user_info import User
import json
import os
import streamlit as st


def load_scripts(file_path) -> List[str]:
    # Reading the text file and importing lines into an array, ignoring empty lines
    lines_array = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()  # Remove leading/trailing whitespace
            if stripped_line:  # Only add non-empty lines
                lines_array.append(stripped_line)
    return lines_array

def load_user_information(user_id):
    if user_id == "001":
        return User("Ada", "001", "Female", "23", "Bachelor", "1")
    elif user_id == "002":
        return User("Ella", "002", "Female", "45", "Ph.D", "0")
    elif os.path.exists(f"data/others/user_info/{user_id}.json"):
        with open(f"data/others/user_info/{user_id}.json", "r") as f:
            data = json.load(f)
            return User(**data)
    else:
        return None
    

def save_user_info(user: User):
    with open(f"data/others/user_info/{user.user_id}.json", "w") as f:
        json.dump(user.to_dict(), f, indent=4)

