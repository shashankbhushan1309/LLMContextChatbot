import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Union
import os

class VectorDB:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database with ChromaDB
        
        Args:
            persist_directory: Directory to persist the database
        """
        print(f"Initializing ChromaDB with persistence at {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"ChromaDB initialized with collection 'pdf_documents'")
        print(f"Collection has {self.collection.count()} documents")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the vector database
        
        Args:
            chunks: List of chunks with text, embedding, and metadata
        """
        if not chunks:
            print("No chunks to add to vector DB")
            return
            
        print(f"Adding {len(chunks)} chunks to vector DB")
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}_{hash(chunk['text'])}" for i, chunk in enumerate(chunks)]
        documents = [chunk["text"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        metadatas = [{
            "source": chunk["source"],
            "chunk_size": chunk["chunk_size"]
        } for chunk in chunks]
        
        # Add to collection in batches to avoid memory issues with large uploads
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_ids = ids[i:end_idx]
            batch_documents = documents[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            print(f"Adding batch {i//batch_size + 1}: {len(batch_ids)} chunks")
            
            # Add batch to collection
            self.collection.add(
                ids=batch_ids,
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        print(f"Successfully added chunks to vector DB. Collection now has {self.collection.count()} documents")
    
    def query(
        self, 
        query_embedding: List[float],
        n_results: int = 5,
        filter_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar documents
        
        Args:
            query_embedding: Embedding of the query
            n_results: Number of results to return
            filter_source: Filter results by source (filename)
            
        Returns:
            Dictionary with query results
        """
        print(f"Querying vector DB for {n_results} results")
        if filter_source:
            print(f"Filtering by source: {filter_source}")
            
        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            
            # Add filter if specified
            if filter_source:
                query_params["where"] = {"source": filter_source}
            
            # Execute query
            results = self.collection.query(**query_params)
            
            print(f"Found {len(results['documents'][0])} matching documents")
            return results
            
        except Exception as e:
            print(f"Error querying vector DB: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty results on error
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]]
            }