# src/pdf_text_extractor.py
from typing import List, Dict, Any
import fitz  # PyMuPDF
import re

def _normalize_whitespace(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def extract_pages_with_sections(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts pages from PDF and attempts semantic sectioning using font-size heuristics.
    Returns a list of dicts: {"page_num": int, "text": str, "sections": [{"heading": str, "text": str}], "raw_blocks": [...]}
    """
    doc = fitz.open(pdf_path)
    pages_out = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        # Collect text spans with font size
        spans = []
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "")
                    size = span.get("size", 0)
                    font = span.get("font", "")
                    if not text.strip():
                        continue
                    spans.append({"text": text, "size": float(size), "font": font})
        if not spans:
            # fallback: raw page text
            raw_text = page.get_text("text")
            pages_out.append({
                "page_num": page_idx + 1,
                "text": _normalize_whitespace(raw_text),
                "sections": [{"heading": None, "text": _normalize_whitespace(raw_text)}],
                "raw_blocks": []
            })
            continue

        # Determine typical font sizes and identify candidate headings as spans with size >= (mean + std*0.6)
        sizes = [s["size"] for s in spans if s["size"] > 0]
        mean_size = sum(sizes) / len(sizes) if sizes else 0
        std = (sum((x - mean_size) ** 2 for x in sizes) / len(sizes)) ** 0.5 if sizes else 0
        heading_threshold = mean_size + 0.6 * std

        # Build lines by concatenating spans while preserving relative size changes
        lines = []
        cur_line = {"size": spans[0]["size"], "text": spans[0]["text"]}
        for s in spans[1:]:
            if abs(s["size"] - cur_line["size"]) < 0.1:
                cur_line["text"] += " " + s["text"]
            else:
                lines.append(cur_line)
                cur_line = {"size": s["size"], "text": s["text"]}
        lines.append(cur_line)

        # Identify headings and group lines into sections
        sections = []
        cur_section = {"heading": None, "text": ""}
        for ln in lines:
            txt = ln["text"].strip()
            if not txt:
                continue
            # Heuristic: short line (<=8 words) and larger than threshold -> heading
            if ln["size"] >= heading_threshold and len(txt.split()) <= 10:
                # start new section
                if cur_section["text"].strip():
                    sections.append(cur_section)
                cur_section = {"heading": _normalize_whitespace(txt), "text": ""}
            else:
                # append to current section
                if cur_section["text"]:
                    cur_section["text"] += "\n" + txt
                else:
                    cur_section["text"] = txt
        if cur_section and cur_section["text"].strip():
            sections.append(cur_section)

        full_text = "\n\n".join([s["text"] for s in sections])
        pages_out.append({
            "page_num": page_idx + 1,
            "text": _normalize_whitespace(full_text),
            "sections": [{"heading": s["heading"], "text": _normalize_whitespace(s["text"])} for s in sections],
            "raw_blocks": blocks
        })
    return pages_out
