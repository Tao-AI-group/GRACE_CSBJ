import os
import re
from config import BASE_DIR

def clean_text(content):
    # Remove lines containing dates in formats like "October 26, 2021" or "7/30/22, 3:58 AM"
    date_pattern = r'^(\w+ \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{2} (AM|PM))$'
    
    # Remove lines containing page numbers like "1/3"
    page_number_pattern = r'^\d+/\d+$'
    
    # Remove lines like "--- Page 2 ---"
    page_text_pattern = r'^--- Page \d+ ---'
    
    # Remove lines containing web links (URLs)
    url_pattern = r'^(http|https)://\S+$'

    # Combine lines that are part of the same paragraph but split by newlines
    content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)

    cleaned_lines = []
    for line in content.splitlines():
        if not re.match(date_pattern, line) and \
           not re.match(page_number_pattern, line) and \
           not re.match(page_text_pattern, line) and \
           not re.match(url_pattern, line):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_txt_files_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cleaned_content = clean_text(content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)


if __name__ == "__main__":
    folder_path = BASE_DIR / "data/example_articles"

    clean_txt_files_in_folder(folder_path)
    print("Cleaning complete.")