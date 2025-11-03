import os
import fitz  # PyMuPDF
from config import BASE_DIR

def extract_text_to_txt(pdf_path, txt_output_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    
    # Open the output .txt file in write mode
    with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
        # Loop through each page of the PDF
        for page_num in range(len(document)):
            # Load the page and extract text
            page = document.load_page(page_num)
            text = page.get_text()
            
            # Write the extracted text to the file, add a page separator
            txt_file.write(text)
            txt_file.write('\n')
    
    # Close the document
    document.close()

def extract_pdfs_in_folder(folder_path):
    # Walk through all folders and subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a PDF
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                
                # Create a corresponding .txt file path in the same directory
                txt_output_path = os.path.splitext(pdf_path)[0] + ".txt"
                
                print(f'Extracting text from: {pdf_path}')
                
                # Extract the text from the PDF and save to the .txt file
                extract_text_to_txt(pdf_path, txt_output_path)
                
                print(f'Text saved to: {txt_output_path}')
                
def remove_txt_files_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

# Usage example

folder_path = BASE_DIR / "data/example_articles"  # Replace with the path to the folder containing sub-folders and PDFs
remove_txt_files_in_folder(folder_path)
extract_pdfs_in_folder(folder_path)
