import json
import pickle
from pathlib import Path
from utils import clean_text
from PyPDF2 import PdfReader

with open("model.pkl", "rb") as f:

    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

label_map = {0: "title", 1: "H1", 2: "H2", 3: "H3", 4: "O"}

def extract_text_blocks(pdf_path):
    reader = PdfReader(pdf_path)
    blocks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            lines = text.split("\n")
            for line in lines:
                cleaned = clean_text(line)
                if cleaned:
                    blocks.append({"text": line.strip(), "page": page_num + 1})
    return blocks


pdf_dir = Path("sample_dataset/pdfs")
output_dir = Path("sample_dataset/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

for pdf_path in pdf_dir.glob("*.pdf"):
    blocks = extract_text_blocks(pdf_path)
    texts = [clean_text(block["text"]) for block in blocks]
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    title = None
    outline = []

    for block, pred in zip(blocks, preds):
        label = label_map[pred]
        if label == "title" and title is None:
            title = block["text"]
        elif label != "O":
            outline.append({"text": block["text"], "label": label, "page": block["page"]})

    result = {
        "title": title if title else "",
        "outline": outline
    }

    output_path = output_dir / (pdf_path.stem + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f" Processed {pdf_path} -> {output_path}")
