import fitz  
import csv
import os

def extract_text_features(pdf_path, output_csv):
    doc = fitz.open(pdf_path)
    rows = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                font_sizes = []
                is_bold = False
                for span in line["spans"]:
                    line_text += span["text"].strip()
                    font_sizes.append(span["size"])
                    if "Bold" in span["font"]:
                        is_bold = True
                if not line_text.strip():
                    continue
                avg_font = round(sum(font_sizes) / len(font_sizes), 2)
                x0, y0 = line["bbox"][0], line["bbox"][1]
                rows.append([
                    line_text, avg_font, x0, y0,
                    page_num + 1, is_bold,
                    line_text.isupper(), len(line_text)
                ])

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "text", "font_size", "x0", "y0", "page",
            "is_bold", "is_uppercase", "text_length"
        ])
        writer.writerows(rows)

if __name__ == "__main__":
    input_folder = "sample_dataset/pdfs"
    output_folder = "training_csvs"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace(".pdf", "_features.csv"))
            extract_text_features(pdf_path, output_csv)
            print(f"Extracted features for {filename}")
