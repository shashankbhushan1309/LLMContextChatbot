import os
import sys
from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_db import VectorDB
import shutil

def main():
    print("\n----- PDF QA SYSTEM DIAGNOSTIC TOOL -----\n")
    
    # Check if vector DB exists and its state
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        print(f"Vector DB exists at {db_path}")
        dirs = os.listdir(db_path)
        print(f"Contents: {dirs}")
        
        # Check if it has data
        vector_db = VectorDB(persist_directory=db_path)
        count = vector_db.collection.count()
        print(f"Number of chunks in database: {count}")
        
        if count > 0:
            # Show sample entries
            try:
                results = vector_db.collection.get(limit=3)
                if results and 'metadatas' in results:
                    print("\nSample entries:")
                    for i, (meta, doc) in enumerate(zip(results['metadatas'], results['documents'])):
                        print(f"{i+1}. Source: {meta.get('source')}")
                        print(f"   Text preview: {doc[:100]}...\n")
                
                # List all sources
                results = vector_db.collection.get()
                if results and 'metadatas' in results:
                    sources = {}
                    for meta in results['metadatas']:
                        source = meta.get('source', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    print("\nSources in database:")
                    for source, count in sources.items():
                        print(f"- {source}: {count} chunks")
            except Exception as e:
                print(f"Error retrieving data: {e}")
    else:
        print(f"Vector DB directory not found at {db_path}")
    
    # Check for PDFs in uploads folder
    uploads_path = "./uploads"
    if not os.path.exists(uploads_path):
        os.makedirs(uploads_path)
        print("\nCreated uploads directory")
    
    # Ask user if they want to test PDF processing
    print("\nDo you want to test PDF processing with a sample file? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        # Ask for PDF path
        print("\nEnter path to a PDF file:")
        pdf_path = input().strip()
        
        if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            # Process the PDF
            print(f"\nProcessing PDF: {pdf_path}")
            processor = PDFProcessor()
            try:
                chunks, text = processor.process_pdf(pdf_path)
                print(f"\nExtracted {len(text)} characters and created {len(chunks)} chunks")
                
                if chunks:
                    print(f"\nFirst chunk preview: {chunks[0]['text'][:200]}...")
                    
                    # Try generating embeddings
                    print("\nGenerating embeddings...")
                    embedding_gen = EmbeddingGenerator()
                    chunks_with_embeddings = embedding_gen.process_chunks(chunks)
                    
                    print(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
                    
                    # Ask if user wants to add to DB
                    print("\nAdd these chunks to the vector database? (y/n)")
                    add_choice = input().strip().lower()
                    
                    if add_choice == 'y':
                        print("\nAdding chunks to vector database...")
                        db = VectorDB()
                        db.add_chunks(chunks_with_embeddings)
                        print("Chunks added successfully")
                else:
                    print("No chunks created. The PDF might be empty or contains only images.")
            except Exception as e:
                print(f"Error processing PDF: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid PDF path or file is not a PDF")
    
    # Ask if user wants to reset the database
    print("\nWould you like to reset the vector database? (WARNING: This will delete all data) (y/n)")
    reset_choice = input().strip().lower()
    
    if reset_choice == 'y':
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
                print("Vector database reset successfully")
            except Exception as e:
                print(f"Error resetting database: {e}")
        else:
            print("No database to reset")
    
    print("\n----- DIAGNOSTIC COMPLETE -----\n")

if __name__ == "__main__":
    main() 