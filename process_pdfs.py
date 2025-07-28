
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    import PyPDF2
    import pdfplumber
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
except ImportError:
    print("Required libraries not installed. Please run:")
    print("pip install PyPDF2 pdfplumber pdfminer.six")
    exit(1)

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
except ImportError:
    print("ML libraries not installed. Please run:")
    print("pip install spacy scikit-learn numpy")
    print("python -m spacy download en_core_web_sm")
    exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.heading_patterns = {
            'title': [
                r'^[A-Z][A-Za-z\s]+(?:[:\-][A-Za-z\s]+)*$',  
                r'^[A-Z\s]+$',
            ],
            'h1': [
                r'^\d+\.?\s+[A-Z][A-Za-z\s]+',  
                r'^[A-Z][A-Z\s]{2,}[A-Za-z]*$', 
                r'^Chapter\s+\d+',  
                r'^Section\s+\d+',  
            ],
            'h2': [
                r'^\d+\.\d+\.?\s+[A-Z][A-Za-z\s]+',  
                r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', 
            ],
            'h3': [
                r'^\d+\.\d+\.\d+\.?\s+[A-Z][A-Za-z\s]+', 
                r'^[a-z]\)\s+[A-Z][A-Za-z\s]+', 
                r'^[A-Z][a-z]+:', 
            ]
        }
    
    def extract_text_with_formatting(self, pdf_path: str) -> List[Dict[str, Any]]:
        text_blocks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    chars = page.chars
                    
                    if not chars:
                        continue
                    
                    lines = self._group_chars_into_lines(chars)
                    
                    for line in lines:
                        if line['text'].strip():
                            text_blocks.append({
                                'text': line['text'].strip(),
                                'page': page_num + 1,
                                'font_size': line['avg_font_size'],
                                'font_name': line['font_name'],
                                'is_bold': line['is_bold'],
                                'y_position': line['y_position']
                            })
        
        except Exception as e:
            logger.error(f"Error extracting text with formatting: {e}")
            return self._fallback_text_extraction(pdf_path)
        
        return text_blocks
    
    def _group_chars_into_lines(self, chars: List[Dict]) -> List[Dict[str, Any]]:
        if not chars:
            return []
        
        lines = []
        current_line = []
        current_y = chars[0]['y0']
        
        for char in chars:
            if abs(char['y0'] - current_y) <= 2:
                current_line.append(char)
            else:
                if current_line:
                    lines.append(self._process_line(current_line))
                current_line = [char]
                current_y = char['y0']
        
        if current_line:
            lines.append(self._process_line(current_line))
        
        return lines
    
    def _process_line(self, line_chars: List[Dict]) -> Dict[str, Any]:
        text = ''.join(char['text'] for char in line_chars)
        
        font_sizes = [char.get('size', 12) for char in line_chars if char.get('size')]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        font_names = [char.get('fontname', '') for char in line_chars if char.get('fontname')]
        font_name = max(set(font_names), key=font_names.count) if font_names else ''
        
        is_bold = any('bold' in font_name.lower() or 'black' in font_name.lower() 
                     for font_name in font_names)
        
        return {
            'text': text,
            'avg_font_size': avg_font_size,
            'font_name': font_name,
            'is_bold': is_bold,
            'y_position': line_chars[0]['y0'] if line_chars else 0
        }
    
    def _fallback_text_extraction(self, pdf_path: str) -> List[Dict[str, Any]]:
        text_blocks = []
        
        try:
            text = extract_text(pdf_path)
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                if line.strip():
                    text_blocks.append({
                        'text': line.strip(),
                        'page': 1,
                        'font_size': 12,  
                        'font_name': '',
                        'is_bold': False,
                        'y_position': i
                    })
        
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
        
        return text_blocks
    
    def classify_headings_ml(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        if not text_blocks:
            return {'title': [], 'h1': [], 'h2': [], 'h3': []}
        
        features = []
        texts = []
        
        for block in text_blocks:
            text = block['text']
            texts.append(text)
            
            feature_vector = [
                block['font_size'],
                len(text),
                len(text.split()),
                int(block['is_bold']),
                int(text.isupper()),
                int(text.istitle()),
                int(bool(re.match(r'^\d+\.', text))), 
                int(bool(re.match(r'^[A-Z]', text))),   
                text.count('.'),
                text.count(':'),
                block.get('y_position', 0) / 1000, 
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        headings = self._classify_with_rules_and_clustering(text_blocks, features)
        
        return headings
    
    def _classify_with_rules_and_clustering(self, text_blocks: List[Dict[str, Any]], 
                                          features: np.ndarray) -> Dict[str, List[str]]:
        result = {'title': [], 'h1': [], 'h2': [], 'h3': []}
        
        for block in text_blocks:
            text = block['text']
            classified = False
            
            for heading_type, patterns in self.heading_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        result[heading_type].append(text)
                        classified = True
                        break
                if classified:
                    break
            
            if not classified:
                self._classify_by_heuristics(block, result)
        
        result = self._post_process_headings(result, text_blocks)
        
        return result
    
    def _classify_by_heuristics(self, block: Dict[str, Any], result: Dict[str, List[str]]):
        text = block['text']
        font_size = block['font_size']
        is_bold = block['is_bold']
        
        if len(text) < 3 or len(text.split()) > 20:
            return
        
        if re.match(r'^\d+$', text) or 'page' in text.lower():
            return
        
        if (font_size > 16 or is_bold) and len(result['title']) == 0 and len(text.split()) <= 10:
            result['title'].append(text)
        
        elif font_size > 14 or (is_bold and re.match(r'^[A-Z]', text)):
            result['h1'].append(text)
        
        elif font_size > 12 or text.istitle():
            result['h2'].append(text)
        
        elif re.match(r'^[A-Z]', text) and len(text.split()) <= 8:
            result['h3'].append(text)
    
    def _post_process_headings(self, result: Dict[str, List[str]], 
                              text_blocks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        
        if not result['title'] and text_blocks:
            first_significant = None
            for block in text_blocks[:5]:  
                text = block['text']
                if len(text.split()) >= 2 and len(text.split()) <= 15:
                    first_significant = text
                    break
            
            if first_significant:
                result['title'] = [first_significant]

                for category in ['h1', 'h2', 'h3']:
                    result[category] = [h for h in result[category] if h != first_significant]
        
        
        for category in result:
            seen = set()
            unique_list = []
            for item in result[category]:
                if item not in seen:
                    seen.add(item)
                    unique_list.append(item)
            result[category] = unique_list
        
        
        result['title'] = result['title'][:1] 
        result['h1'] = result['h1'][:10]      
        result['h2'] = result['h2'][:20]      
        result['h3'] = result['h3'][:30]       
        
        return result
    
    def process_pdf(self, pdf_path: str, output_path: str) -> bool:
        try:
            logger.info(f"Processing: {pdf_path}")
            
            text_blocks = self.extract_text_with_formatting(pdf_path)
            
            if not text_blocks:
                logger.warning(f"No text extracted from {pdf_path}")

                result = {'title': [], 'h1': [], 'h2': [], 'h3': []}
            else:
                result = self.classify_headings_ml(text_blocks)
            
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated: {output_path}")
            logger.info(f"Results: Title={len(result['title'])}, H1={len(result['h1'])}, "
                       f"H2={len(result['h2'])}, H3={len(result['h3'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return False

def main():
    
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    
    if not input_dir.exists():
        input_dir = Path("./input")
        output_dir = Path("./output")
    
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    processor = PDFProcessor()
    
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    
    success_count = 0
    for pdf_file in pdf_files:
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        
        
        if processor.process_pdf(str(pdf_file), str(output_file)):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(pdf_files)} files")

if __name__ == "__main__":
    main()