# Challenge 1a: PDF Processing Solution

## Overview

This is a **working solution** for Challenge 1a of the Adobe India Hackathon 2025.  
The challenge requires implementing a PDF processing solution that extracts structured data from PDF documents and outputs JSON files.  
The solution is containerized using Docker and meets the performance and resource constraints defined by the challenge.

## Official Challenge Guidelines

### Submission Requirements

- **GitHub Project**: Complete code repository with the working solution
- **Dockerfile**: Present in the root directory and fully functional
- **README.md**: Documentation explaining the solution, models, and libraries used

### Build Command

```bash
docker build --platform linux/amd64 -t <reponame.someidentifier> .

docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/repoidentifier/:/app/output --network none <reponame.someidentifier>


Solution Folder Structure

Challenge_1a/
├── sample_dataset/
│   ├── outputs/
│   ├── pdfs/
│   └── schema/
│       └── output_schema.json
├── Dockerfile
├── requirements.txt
├── process_pdfs.py
└── README.md

Docker Build

docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

Docker Run

docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/input \
  -v $(pwd)/sample_dataset/outputs:/app/output \
  --network none \
  mysolutionname:somerandomidentifier

Example Output:

INFO:__main__:Found 5 PDF files to process
INFO:__main__:Processing: /app/input/file01.pdf
INFO:__main__:Generated: /app/output/file01.json
INFO:__main__:Results: Title=1, H1=3, H2=0, H3=6
...
INFO:__main__:Successfully processed 5/5 files


Libraries Used
pdfplumber for PDF text extraction

spaCy (en_core_web_sm) for NLP-based structure analysis

PyPDF2 for additional parsing

Standard Python libraries for JSON and file handling

```
