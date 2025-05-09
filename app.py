import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Form
from typing import List, Dict, Optional, Annotated
import shutil
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_db import VectorDB
from llm_module import LLMModule

# Load environment variables from config.env
load_dotenv("config.env")

# Create directories
os.makedirs("./uploads", exist_ok=True)
os.makedirs("./static", exist_ok=True)
os.makedirs("./templates", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="PDF Question Answering System")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize components
pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
embedding_generator = EmbeddingGenerator()
vector_db = VectorDB(persist_directory="./chroma_db")

# Initialize LLM with model from environment variable or use default
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
llm = LLMModule(model_name=model_name)

# Track uploaded files
uploaded_files = {}

# For handling file paths properly
import pathlib

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "uploaded_files": list(uploaded_files.keys())}
    )

@app.post("/upload")
async def upload_pdf(files: Annotated[List[UploadFile], File()]):
    """
    Upload one or more PDF files, process them, and add to the vector database
    """
    print(f"Received {len(files)} files for upload")
    results = []
    
    for file in files:
        # Save file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file_path = temp_file.name
        temp_file.close()  # Close the file handle immediately
        
        try:
            print(f"Processing file: {file.filename}")
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            print(f"Saved {len(contents)} bytes to temporary file: {temp_file_path}")
            
            # Process PDF
            chunks, _ = pdf_processor.process_pdf(temp_file_path)
            
            if not chunks:
                print(f"WARNING: No chunks were created for {file.filename}")
                results.append({
                    "filename": file.filename,
                    "status": "warning",
                    "error": "No text content could be extracted from this PDF.",
                    "num_chunks": 0
                })
                continue
                
            print(f"Generated {len(chunks)} chunks for {file.filename}")
            
            # Generate embeddings for chunks
            chunks_with_embeddings = embedding_generator.process_chunks(chunks)
            
            # Add to vector DB
            vector_db.add_chunks(chunks_with_embeddings)
            
            # Store the original filename for lookup
            file_key = file.filename
            
            # Track the file - IMPORTANT: Use the exact raw filename for storage
            # This ensures the source filter will match exactly
            source_name = file.filename
            
            # Add to our tracked files dictionary
            uploaded_files[file_key] = {
                "filename": file.filename,
                "source_name": source_name,
                "num_chunks": len(chunks)
            }
            
            # Add a success message 
            results.append({
                "filename": file.filename,
                "status": "success",
                "num_chunks": len(chunks),
                "source_name": source_name  # Include the source name in response
            })
            print(f"Successfully processed {file.filename} with {len(chunks)} chunks")
            print(f"Source name for filtering: {source_name}")
            
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {str(e)}")
    
    print(f"Uploaded files: {list(uploaded_files.keys())}")
    return results

@app.post("/ask")
async def ask_question(
    question: Annotated[str, Form()],
    source_file: Annotated[Optional[str], Form()] = None,
    num_results: Annotated[int, Form()] = 5
):
    """
    Answer a question based on the uploaded PDFs
    """
    try:
        print(f"Processing question: {question}")
        print(f"Source file filter: {source_file}")
        print(f"Number of results: {num_results}")
        
        if not uploaded_files:
            return {
                "answer": "Please upload some PDF documents first.",
                "sources": [],
                "num_chunks_used": 0
            }
        
        # Generate query embedding
        print("Generating query embedding...")
        query_embedding = embedding_generator.generate_embeddings(question)
        print(f"Embedding shape: {type(query_embedding)}")
        
        # Fix embedding format - ensure it's a single list of numbers
        if hasattr(query_embedding, 'cpu'):
            # Convert tensor to numpy and then to list
            query_embedding = query_embedding.cpu().numpy()
            # If it's a batch of embeddings (2D array), take the first one
            if len(query_embedding.shape) > 1 and query_embedding.shape[0] == 1:
                query_embedding = query_embedding[0]
            # Convert to list
            query_embedding = query_embedding.tolist()
        
        # Query vector DB
        print("Querying vector database...")
        print(f"Available files: {list(uploaded_files.keys())}")
        
        # Debug source file filtering
        if source_file:
            print(f"Filtering by source: {source_file}")
            # Check if the source file is in the uploaded files
            
            # Try to find the source_file in uploaded_files
            found = False
            actual_source = source_file
            
            for key, info in uploaded_files.items():
                if key == source_file or info.get("filename") == source_file:
                    found = True
                    actual_source = info.get("source_name", source_file)
                    print(f"Found matching file: {key}, using source: {actual_source}")
                    break
                    
            if not found:
                print(f"WARNING: Source file {source_file} not found in uploaded files. Available files: {list(uploaded_files.keys())}")
                print("Will search without source filter")
                actual_source = None
        else:
            actual_source = None
        
        results = vector_db.query(
            query_embedding=query_embedding,
            n_results=num_results,
            filter_source=actual_source
        )
        
        # Extract documents
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        print(f"Found {len(documents)} document chunks")
        
        if metadatas:
            print(f"Sources of retrieved chunks: {[meta.get('source') for meta in metadatas]}")
        
        if not documents:
            return {
                "answer": "No relevant information found in the uploaded documents. Try uploading more PDFs or rephrasing your question.",
                "sources": [],
                "num_chunks_used": 0
            }
        
        # Filter relevant contexts
        print("Filtering relevant contexts...")
        contexts = llm.filter_relevant_contexts(question, [
            {"document": doc, "metadata": metadata}
            for doc, metadata in zip(documents, metadatas)
        ])
        
        print(f"Using {len(contexts)} contexts to generate answer")
        if contexts:
            print(f"First context sample: {contexts[0][:100]}...")
        
        # Generate answer
        print("Generating answer with LLM...")
        answer = llm.generate_answer(question, contexts)
        print(f"Generated answer: {answer}")
        
        # Calculate sources
        sources = [metadata["source"] for metadata in metadatas]
        unique_sources = list(set(sources))
        
        return {
            "answer": answer,
            "sources": unique_sources,
            "num_chunks_used": len(documents)
        }
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "num_chunks_used": 0
        }

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    return {"files": list(uploaded_files.keys())}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 