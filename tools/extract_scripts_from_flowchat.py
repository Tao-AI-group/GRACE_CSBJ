import xml.etree.ElementTree as ET
import sys
from bs4 import BeautifulSoup
from config import BASE_DIR
def remove_html_tags(text):
    """
    Remove HTML tags from the provided text.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_flowchart_texts(drawio_file):
    """
    Parse the draw.io file and extract the plain text content from each flowchart box.
    This assumes that each flowchart box corresponds to an XML node like <mxCell vertex="1" ...>
    and that the text is stored in the 'value' attribute.
    """
    try:
        tree = ET.parse(drawio_file)
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)
        
    root = tree.getroot()
    texts = []

    for cell in root.iter('mxCell'):
        # Check if the node is a graphical element and has a value attribute.
        if cell.get('vertex') == "1":
            text = cell.get('value')
            if text and text.strip():
                clean_text = remove_html_tags(text)
                texts.append(clean_text)
    return texts

def extract_flowchart_topo_order(drawio_file):
    tree = ET.parse(drawio_file)
    root = tree.getroot()

    id_to_text = {}

    graph = {}
    
    in_degree = {}

    for cell in root.iter('mxCell'):
        cell_id = cell.get('id')
        
        if cell.get('vertex') == "1":
            raw_text = cell.get('value') or ""
            clean_text = remove_html_tags(raw_text)
            id_to_text[cell_id] = clean_text
            graph[cell_id] = []
            in_degree[cell_id] = 0

    for cell in root.iter('mxCell'):
        if cell.get('edge') == "1":
            source = cell.get('source')
            target = cell.get('target')
            if source and target and source in graph and target in graph:
                graph[source].append(target)
                in_degree[target] += 1

    
    from collections import deque

    queue = deque([node for node in graph if in_degree[node] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    
    result = [id_to_text[node] for node in topo_order if node in id_to_text]
    return result



if __name__ == '__main__':
    drawio_file = BASE_DIR / "data/others/demo.drawio"
    texts = extract_flowchart_topo_order(drawio_file)

    if not texts:
        print("No flowchart box text content found.")
    else:
        for idx, text in enumerate(texts, start=1):
            print(f"Step {idx}: {text}")

