import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Tuple

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor
        
        Args:
            chunk_size: The size of text chunks in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract all text from a PDF file with enhanced error handling"""
        print(f"Extracting text from PDF: {file_path}")
        
        try:
            doc = fitz.open(file_path)
            text = ""
            
            print(f"PDF has {len(doc)} pages")
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    
                    # Try to get text normally first
                    page_text = page.get_text()
                    
                    # If page has no text but might have images/scan
                    if not page_text.strip():
                        print(f"Page {page_num+1} has no readable text, checking for images...")
                        
                        # Alternative: get text from images if page seems empty
                        # This uses simple image-based text extraction
                        # For production, consider using OCR like PyTesseract
                        try:
                            # Try to extract images and treat as text
                            image_list = page.get_images(full=True)
                            if image_list:
                                print(f"Page {page_num+1} has {len(image_list)} images. Text might be in image form.")
                                # Since we don't have OCR set up, just note this
                                page_text = f"[Image-based content on page {page_num+1}]"
                        except Exception as img_err:
                            print(f"Error checking for images on page {page_num+1}: {str(img_err)}")
                    
                    text += page_text
                    print(f"Page {page_num+1}: Extracted {len(page_text)} characters")
                except Exception as e:
                    print(f"Error extracting text from page {page_num+1}: {str(e)}")
            
            if not text.strip():
                print("WARNING: No text extracted from PDF. The file might be scanned or image-based.")
                # Provide a placeholder for completely empty documents
                text = f"[This document appears to be image-based or contains no extractable text: {os.path.basename(file_path)}]"
            else:
                print(f"Successfully extracted {len(text)} characters from PDF")
                
            return text
        except Exception as e:
            print(f"Error opening PDF file: {str(e)}")
            raise
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Chunk text with intelligent splitting at paragraph/section boundaries
        
        Args:
            text: The text to chunk
            filename: The name of the source PDF file
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        print(f"Chunking text for {filename}, total length: {len(text)} characters")
        
        # Clean text, normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is too short, don't chunk it
        if len(text) < self.chunk_size:
            print(f"Text is shorter than chunk size ({len(text)} < {self.chunk_size}), creating single chunk")
            return [{
                "text": text,
                "source": filename,
                "chunk_size": len(text)
            }]
        
        # Split text into paragraphs (approximation)
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If the split doesn't work well (e.g., very few paragraphs), use sentence splitting
        if len(paragraphs) < 3:
            print("Few paragraphs detected, trying sentence-based splitting")
            paragraphs = re.split(r'(?<=[.!?])\s+', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        print(f"Split text into {len(paragraphs)} paragraphs")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size and we already have content
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Store current chunk
                chunks.append({
                    "text": current_chunk,
                    "source": filename,
                    "chunk_size": len(current_chunk)
                })
                
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    # Get last portion of the previous chunk for overlap
                    current_chunk = current_chunk[-self.chunk_overlap:] + " " + para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += " " + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "source": filename,
                "chunk_size": len(current_chunk)
            })
        
        print(f"Created {len(chunks)} chunks from text")
        
        # Print a sample of the first chunk
        if chunks:
            print(f"First chunk sample: {chunks[0]['text'][:100]}...")
            
        # Ensure we always return at least one chunk, even if it's a placeholder
        if not chunks:
            print("WARNING: No chunks created, adding placeholder chunk")
            chunks = [{
                "text": f"[No processable content in document: {filename}]",
                "source": filename,
                "chunk_size": 50
            }]
            
        return chunks
    
    def process_pdf(self, file_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process a PDF file: extract text and create chunks
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, original_text)
        """
        # Get the real filename for storage (without the temp path)
        filename = os.path.basename(file_path)
        print(f"Processing PDF: {filename}")
        
        text = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text, filename)
        
        return chunks, text 