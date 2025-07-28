import os
import json
import time
import fitz  # PyMuPDF
import numpy as np
import faiss
import torch
import sys
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datetime import datetime

# --- CONFIGURATION ---
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
SUMMARIZER_MODEL = 't5-small'
MODEL_CACHE = './model_cache'

# --- 1. MODEL INITIALIZATION ---
def get_models():
    """Loads and caches models for offline use."""
    print("Loading models... (This may take a moment on first run)")
    
    # Load sentence transformer for embeddings
    embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE)
    
    # Load T5 for summarization
    tokenizer = T5Tokenizer.from_pretrained(SUMMARIZER_MODEL, cache_dir=MODEL_CACHE)
    summarizer = T5ForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL, cache_dir=MODEL_CACHE)
    
    print("Models loaded successfully.")
    return embedder, tokenizer, summarizer

# --- 2. IMPROVED HEADING DETECTION ---
def analyze_text_structure(doc):
    """Analyze document structure to identify text patterns."""
    font_info = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=4)["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if not line.get("spans"):
                    continue
                
                # Get line text
                line_text = "".join([span["text"] for span in line["spans"]]).strip()
                if not line_text or len(line_text) < 3:
                    continue
                
                # Analyze first span for formatting
                first_span = line["spans"][0]
                font_info.append({
                    'text': line_text,
                    'size': round(first_span["size"], 1),
                    'font': first_span["font"].lower(),
                    'flags': first_span["flags"],
                    'page': page_num + 1,
                    'line_count': len(line_text.split()),
                    'bbox': line["bbox"]
                })
    
    return font_info

def detect_headings_improved(font_info):
    """Improved heading detection using multiple heuristics."""
    if not font_info:
        return []
    
    # Calculate font size statistics
    sizes = [item['size'] for item in font_info]
    avg_size = np.mean(sizes)
    size_threshold = avg_size + np.std(sizes) * 0.5
    
    # Identify potential headings
    headings = []
    for item in font_info:
        text = item['text']
        
        # Skip if too long (likely paragraph text)
        if len(text) > 100 or item['line_count'] > 15:
            continue
            
        # Check various heading indicators
        is_heading = False
        confidence = 0
        
        # Font size check
        if item['size'] > size_threshold:
            confidence += 2
            
        # Bold check (flags & 16 for bold)
        if item['flags'] & 16:
            confidence += 2
            
        # Font name check
        if any(word in item['font'] for word in ['bold', 'black', 'heavy']):
            confidence += 1
            
        # Text pattern checks
        # All caps (but not too short)
        if text.isupper() and len(text) > 5:
            confidence += 1
            
        # Title case
        if text.istitle() and len(text.split()) <= 8:
            confidence += 1
            
        # Ends with colon
        if text.endswith(':'):
            confidence += 1
            
        # Common heading words
        heading_keywords = ['introduction', 'overview', 'chapter', 'section', 'guide', 
                          'tips', 'tricks', 'conclusion', 'summary', 'background',
                          'activities', 'experiences', 'recommendations', 'highlights']
        if any(keyword in text.lower() for keyword in heading_keywords):
            confidence += 1
            
        # Numbering patterns
        if re.match(r'^\d+[\.\-\s]', text) or re.match(r'^[IVXLCDM]+[\.\-\s]', text):
            confidence += 1
            
        # Short lines that aren't sentences
        if len(text.split()) <= 8 and not text.endswith('.'):
            confidence += 0.5
            
        # Decide if it's a heading
        if confidence >= 2.5:
            is_heading = True
            
        if is_heading:
            headings.append({
                'text': text,
                'page': item['page'],
                'confidence': confidence,
                'size': item['size'],
                'bbox': item['bbox']
            })
    
    # Sort by page and position
    headings.sort(key=lambda x: (x['page'], x['bbox'][1]))
    return headings

def extract_sections_improved(pdf_path: str) -> list:
    """
    Improved section extraction with better heading detection.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    
    # Analyze document structure
    font_info = analyze_text_structure(doc)
    if not font_info:
        doc.close()
        return []
    
    # Detect headings
    headings = detect_headings_improved(font_info)
    
    # If no headings found, create basic sections
    if not headings:
        # Fallback: create sections based on pages
        sections = []
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                sections.append({
                    "title": f"Page {page_num + 1} ({filename})",
                    "content": text[:2000],  # Limit content length
                    "page": page_num + 1,
                    "file": filename
                })
        doc.close()
        return sections
    
    # Extract content between headings
    sections = []
    all_text_blocks = []
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        all_text_blocks.append({
            'text': page_text,
            'page': page_num + 1
        })
    
    # Match headings to content
    for i, heading in enumerate(headings):
        current_heading = heading['text']
        current_page = heading['page']
        
        # Find content for this heading
        content_parts = []
        
        # Get content from current page after heading
        current_page_text = all_text_blocks[current_page - 1]['text']
        heading_pos = current_page_text.find(current_heading)
        
        if heading_pos >= 0:
            after_heading = current_page_text[heading_pos + len(current_heading):]
            
            # Find next heading on same page to limit content
            next_heading_pos = len(after_heading)
            for next_heading in headings[i+1:]:
                if next_heading['page'] == current_page:
                    next_pos = after_heading.find(next_heading['text'])
                    if next_pos >= 0:
                        next_heading_pos = next_pos
                        break
            
            content_parts.append(after_heading[:next_heading_pos])
        
        # Add content from subsequent pages until next heading
        if i < len(headings) - 1:
            next_heading_page = headings[i + 1]['page']
            for page_idx in range(current_page, min(next_heading_page, len(all_text_blocks))):
                if page_idx == current_page - 1:
                    continue  # Already processed
                content_parts.append(all_text_blocks[page_idx]['text'])
        else:
            # Last heading - get remaining pages
            for page_idx in range(current_page, len(all_text_blocks)):
                if page_idx == current_page - 1:
                    continue
                content_parts.append(all_text_blocks[page_idx]['text'])
        
        # Clean and combine content
        full_content = " ".join(content_parts).strip()
        full_content = re.sub(r'\s+', ' ', full_content)  # Normalize whitespace
        
        if full_content and len(full_content) > 50:  # Only include substantial content
            sections.append({
                "title": current_heading,
                "content": full_content[:3000],  # Limit content length
                "page": current_page,
                "file": filename
            })
    
    doc.close()
    return sections

# --- 3. CONTENT REFINEMENT ---
def refine_content_for_task(content: str, task: str, persona: str, tokenizer, summarizer, max_length: 100) -> str:
    """
    Refine content to be task-specific and concise.
    """
    if len(content.split()) <= max_length:
        return content
    
    # Create task-focused prompt
    prompt = f"summarize for {persona} to {task}: {content}"
    
    # Tokenize and generate summary
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        summary_ids = summarizer.generate(
            inputs, 
            max_length=max_length, 
            min_length=20,
            length_penalty=2.0,
            num_beams=2,
            early_stopping=True
        )
    
    refined_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Clean up the refined text
    refined_text = refined_text.replace("summarize for", "").replace(f"{persona} to {task}:", "").strip()
    
    return refined_text if refined_text else content[:500]

# --- 4. MAIN PROCESSING ---
def process_collection(collection_path: str, models: tuple):
    """
    Main pipeline with improved processing.
    """
    # Construct correct paths
    input_dir = os.path.abspath(collection_path)
    input_file = os.path.join(input_dir, 'challenge1b_input.json')
    output_file = os.path.join(os.path.dirname(input_dir), 'challenge1b_output.json')
    TOP_K = 5

    print(f"Looking for input file at: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"challenge1b_input.json not found. Looked in: {input_file}")

    with open(input_file, 'r') as f:
        config = json.load(f)

    # Extract configuration
    persona = config.get('persona', {}).get('role', 'Professional')
    task = config.get('job_to_be_done', {}).get('task', 'Complete analysis')
    pdf_filenames = [doc['filename'] for doc in config.get('documents', [])]

    timestamp = datetime.now().isoformat()

    # Step 1: Extract sections from all PDFs with improved method
    print(f"Extracting sections from {len(pdf_filenames)} PDF(s)...")
    all_sections = []
    for filename in pdf_filenames:
        # The filenames in the JSON already include 'PDFs/', so don't add it again
        pdf_path = os.path.join(collection_path, filename)
        print(f"Looking for PDF at: {pdf_path}")
        if os.path.exists(pdf_path):
            sections = extract_sections_improved(pdf_path)
            all_sections.extend(sections)
            print(f"  Extracted {len(sections)} sections from {filename}")
        else:
            print(f"Warning: File {pdf_path} not found.")

    if not all_sections:
        print("No content extracted from PDFs. Aborting.")
        return

    print(f"Total sections extracted: {len(all_sections)}")

    # Step 2: Embed and rank sections
    print("Embedding content and ranking sections...")
    embedder, tokenizer, summarizer = models

    # Create more specific query
    query = f"As a {persona}, I need to {task}. Show me relevant information about planning, activities, recommendations, and practical tips."
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # Filter out very short sections
    valid_sections = [sec for sec in all_sections if len(sec['content'].split()) > 20]
    if not valid_sections:
        print("No valid sections found. Using all sections.")
        valid_sections = all_sections

    # Group sections by document
    from collections import defaultdict
    doc_to_sections = defaultdict(list)
    for sec in valid_sections:
        doc_to_sections[sec['file']].append(sec)

    # For each document, find the most relevant section
    per_doc_best_sections = []
    for doc, sections in doc_to_sections.items():
        section_contents = [s['content'] for s in sections]
        section_embeddings = embedder.encode(section_contents, convert_to_tensor=True, show_progress_bar=False)
        # Normalize
        section_embeddings = section_embeddings / torch.linalg.norm(section_embeddings, axis=1, keepdims=True)
        # Compute similarity
        sims = torch.matmul(section_embeddings, query_embedding)
        best_idx = int(torch.argmax(sims))
        best_score = float(sims[best_idx])
        per_doc_best_sections.append((sections[best_idx], best_score))

    # Sort all best sections by similarity, take up to TOP_K * num_docs if you want more
    per_doc_best_sections.sort(key=lambda x: x[1], reverse=True)

    # You can change this to get more than one per doc if needed
    extracted_sections_output = []
    subsection_analysis_output = []
    for i, (section, similarity_score) in enumerate(per_doc_best_sections):
        # Build extracted_sections entry
        extracted_sections_output.append({
            "document": section['file'],
            "section_title": section['title'],
            "importance_rank": i + 1,
            "page_number": section['page']
        })
        # Refine content for subsection_analysis
        refined_content = refine_content_for_task(
            section['content'],
            task,
            persona,
            tokenizer,
            summarizer,
            max_length=80
        )
        subsection_analysis_output.append({
            "document": section['file'],
            "refined_text": refined_content,
            "page_number": section['page']
        })

    # Clean filenames for metadata
    clean_filenames = [os.path.basename(f) for f in pdf_filenames]

    output_data = {
        "metadata": {
            "input_documents": clean_filenames,
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }

    # Write results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Successfully created output file at: {output_file}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python challenge1b_final.py input")
        sys.exit(1)
        
    collection_dir = sys.argv[1]
    if not os.path.isdir(collection_dir):
        print(f"Error: Directory not found at '{collection_dir}'")
        sys.exit(1)

    start_time = time.time()
    
    # Load models once
    loaded_models = get_models()
    
    # Process the document collection
    process_collection(collection_dir, loaded_models)
    
    end_time = time.time()
    print(f"\n🚀 Total processing time: {end_time - start_time:.2f} seconds.")