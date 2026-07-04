from paddleocr import PaddleOCR
import re
import fitz
import cv2
import unicodedata
import numpy as np
from langchain_core.documents import Document

class Preprocessing:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=True,
            lang="en"
        )
    def process_pdf(self, path: str) -> list[Document]:
        document = []
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text().strip()
                if len(text) < 50:
                    pix = page.get_pixmap(dpi = 300)
                    img = np.frombuffer(
                        pix.samples,
                        dtype = np.uint8
                    ).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    result = self.ocr.predict(img)
                    lines = []
                    if result and result[0]["rec_texts"]:
                        text = "\n".join(result[0]["rec_texts"])
                    else:
                        text = ""
                document.append(
                        Document(
                        page_content = text,
                        metadata = {
                        "page": page_num,
                        "page_label": str(page_num + 1),
                        "source": path,
                            }
                        )
                    )
        return document

    def clean_text(self, text):
        cleanned_text = text.copy()
        for i in range(len(text)):
            cleanned_text[i] = unicodedata.normalize("NFKC", cleanned_text[i])
            cleanned_text[i] = re.sub(r'[\u200B-\u200D\uFEFF\uf000-\uf8ff]', '', cleanned_text[i])
            cleanned_text[i] = cleanned_text[i].replace("\r\n", "\n")
            cleanned_text[i] = cleanned_text[i].replace("\r", "\n")
            cleanned_text[i] = re.sub(r"[ \t]+", " ", cleanned_text[i])
            cleanned_text[i] = re.sub(r"^\s*Page\s+\d+\s*$", "", cleanned_text[i], flags=re.MULTILINE)
            cleanned_text[i] = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", cleanned_text[i], flags=re.MULTILINE)
            cleanned_text[i] = re.sub(r"^\s*\d+(\.\d+)?\s*$", "", cleanned_text[i], flags=re.MULTILINE)
            cleanned_text[i] = re.sub(
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM)?\b",
                "",
                cleanned_text[i],
                flags=re.IGNORECASE,
                )
            cleanned_text[i] = re.sub(r"[-_]{3,}", "", cleanned_text[i])
            cleanned_text[i] = re.sub(r"[=]{3,}", "", cleanned_text[i])
            cleanned_text[i] = re.sub(r"(\w)-\n(\w)", r"\1\2", cleanned_text[i])
            lines = [line.strip() for line in cleanned_text[i].split("\n")]
            merged = []
            for line in lines:
                if not line:
                    merged.append("")
                    continue
                if not merged:
                    merged.append(line)
                    continue
                prev = merged[-1]
                if (
                  prev
                  and not prev.endswith((".", "!", "?", ":", ";"))
                  and not line.startswith("•")
                  and not re.match(r"^[A-Z][A-Za-z ]+$", line)
              ):
                  merged[-1] += " " + line
                else:
                  merged.append(line)
            cleanned_text[i] = "\n".join(merged)
            cleanned_text[i] = re.sub(r"\n{3,}", "\n\n", cleanned_text[i])
        return cleanned_text

    def detect_repeated_lines(self, pages, top_n=6, bottom_n=6, threshold=0.8):
        counter = {}
        total_pages = len(pages)
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            for line in lines[:top_n]:
                if line not in counter:
                    counter[line] = 1
                else:
                    counter[line] += 1
            for line in lines[-bottom_n:]:
                if line not in counter:
                    counter[line] = 1
                else:
                    counter[line] += 1
        min_occurrences = max(2, int(total_pages * threshold))
        return {
            line
            for line, count in counter.items()
            if count >= min_occurrences
            }

    def remove_repeated_lines(self, pages, repeated_lines):
        cleaned_pages = []
        for page in pages:
            lines = [
                line
                for line in page.splitlines()
                if line.strip() not in repeated_lines
                ]
            cleaned_pages.append("\n".join(lines).strip())
        return cleaned_pages
