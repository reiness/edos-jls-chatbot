# src/utils.py
import re
from pathlib import Path

def clean_text(text: str) -> str:
    # basic cleaning
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+',' ', text)
    text = text.strip()
    return text

def is_probable_heading(line: str) -> bool:
    # heuristic: short line, Title Case or ALL CAPS
    if len(line) < 200 and len(line.split()) <= 8:
        if line.isupper() or line.istitle():
            return True
    return False
